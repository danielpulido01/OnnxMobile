"""Un experimento por invocación, con checkpoints por época y tracking W&B."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import wandb

from augment import prepare_augmented_dataset, validate_policy
from data import inspect_dataset, make_loader, sha256, SpectrogramDataset
from experiment import (AI_DIR, log_class_metrics, log_provenance, plot_history, profile_model,
                        save_environment, seed_everything, source_fingerprint, write_json,
                        environment, capture_rng, restore_rng, save_checkpoint, save_table)
from metrics import run_epoch, metrics_from_confusion
from models import build_model


def experiment_grid(config):
    trials = config["trials"]
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", t["name"]) for t in trials):
        raise ValueError("Los nombres de trials deben usar letras, números, guiones o guiones bajos")
    if len(trials) < 3 or len({t["name"] for t in trials}) != len(trials):
        raise ValueError("Se necesitan al menos tres trials con nombres únicos")
    settings = [{k: v for k, v in t.items() if k != "name"} for t in trials]
    if len({json.dumps(t, sort_keys=True) for t in settings}) != len(trials):
        raise ValueError("Los trials deben tener hiperparámetros diferentes")
    if config["epochs"] < 1 or config["patience"] < 1 or config["min_delta"] < 0:
        raise ValueError("epochs/patience deben ser positivos y min_delta no negativo")
    if config["class_weighting"] not in ("inverse_frequency", "none"):
        raise ValueError("class_weighting no soportado")
    validate_policy(config["augmentation"])
    experiments = []
    for architecture in ("lenet5", "mobilenet_v1"):
        for variant in ("base", "augmented"):
            for trial in trials:
                if (trial["optimizer"] not in ("adam", "sgd") or trial["lr"] <= 0
                        or trial["batch_size"] < 2 or trial["weight_decay"] < 0):
                    raise ValueError(f"Trial inválido: {trial['name']}")
                experiments.append({**config, **trial, "architecture": architecture, "variant": variant,
                                    "experiment": f"{architecture}-{variant}-{trial['name']}"})
    return experiments


def select_winners(results):
    winners = {}
    for result in results:
        group = f"{result['architecture']}/{result['variant']}"
        # Desempate predefinido: menor loss y finalmente nombre lexicográfico.
        score = lambda r: (-r["best_validation_macro_f1"], r["best_validation_loss"], r["experiment"])
        if group not in winners or score(result) < score(winners[group]):
            winners[group] = result
    if len(winners) != 4:
        raise ValueError("Faltan combinaciones arquitectura/dataset")
    return winners


def fit_one(config, dataset, suite, device, args):
    directory = suite / config["experiment"]
    directory.mkdir(exist_ok=True)
    write_json(directory / "config.json", config)
    seed_everything(config["seed"], args.threads)
    if (directory / "environment.json").exists():
        env = json.loads((directory / "environment.json").read_text(encoding="utf-8"))
        current = environment(device)
        for key in ("python", "platform", "torch", "numpy", "wandb", "cuda", "cudnn", "gpu", "threads"):
            if env[key] != current[key]:
                raise ValueError(f"No se puede reanudar: cambió el entorno ({key})")
    else:
        env = save_environment(directory, device)
    classes = dataset["metadata"]["classes"]
    clean_data = SpectrogramDataset(dataset["root"], "train")
    training_dataset = dataset["augmented"] if config["variant"] == "augmented" else dataset
    train_data = SpectrogramDataset(training_dataset["root"], "train")
    validation_data = SpectrogramDataset(dataset["root"], "validation")
    train_loader = make_loader(train_data, config["batch_size"], config["seed"], args.workers, True)
    clean_loader = make_loader(clean_data, config["batch_size"], config["seed"], args.workers)
    validation_loader = make_loader(validation_data, config["batch_size"], config["seed"], args.workers)
    counts = np.bincount(clean_data.y, minlength=len(classes))
    if (counts == 0).any():
        raise ValueError("Todas las clases deben estar presentes en entrenamiento")
    weights = len(clean_data) / (len(classes) * counts)
    if config["class_weighting"] == "none":
        weights = np.ones(len(classes))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    model = build_model(config["architecture"], len(classes), config["width_mult"]).to(device)
    optimizer_args = {"lr": config["lr"], "weight_decay": config["weight_decay"]}
    optimizer = (torch.optim.Adam(model.parameters(), **optimizer_args) if config["optimizer"] == "adam"
                 else torch.optim.SGD(model.parameters(), momentum=config.get("momentum", 0.9), **optimizer_args))
    augmentation_rng = torch.Generator().manual_seed(config["seed"] + 1000)
    loaders = [train_loader, clean_loader, validation_loader]
    state = None
    if (directory / "last.pt").exists():
        state = torch.load(directory / "last.pt", map_location=device, weights_only=True)
        if (state["config"] != config or state["dataset_fingerprint"] != dataset["fingerprint"]
                or state["training_dataset_fingerprint"] != training_dataset["fingerprint"]
                or state["source_sha256"] != source_fingerprint() or state["classes"] != classes):
            raise ValueError("Checkpoint incompatible con configuración, datos o código")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer_state"])
    history = state["history"] if state else []
    best_checkpoint = state["best_checkpoint"] if state else None
    best = best_checkpoint["validation"]["macro_f1"] if state else -1.0
    best_loss = best_checkpoint["validation"]["loss"] if state else float("inf")
    best_epoch = best_checkpoint["epoch"] if state else 0
    stale = state["stale"] if state else 0
    patience_reference = state["patience_reference"] if state else -1.0
    first_epoch = state["epoch"] + 1 if state else 1
    elapsed = state["training_seconds"] if state else 0.0
    if state:
        # last.pt incluye el mejor modelo: reparar best.pt si el proceso murió al escribirlo.
        save_checkpoint(directory / "best.pt", best_checkpoint)
        save_table(directory / "history.csv", pd.DataFrame(history))
        print(f"Reanudando {config['experiment']} desde época {first_epoch}", flush=True)
    with wandb.init(project=args.project, entity=args.entity, mode=args.wandb_mode, dir=str(suite),
                    name=config["experiment"] + (f"-resume-e{first_epoch}" if state else ""),
                    group=suite.name, job_type="train",
                    config={**config, "dataset_fingerprint": dataset["fingerprint"], "environment": env,
                            "class_weights": weights.tolist(), "loss": "weighted_cross_entropy",
                            "training_dataset_fingerprint": training_dataset["fingerprint"],
                            "training_samples": len(train_data), "original_training_samples": len(clean_data),
                            "selection_metric": "validation/macro_f1", "resume_from_epoch": first_epoch - 1,
                            "previous_run_id": state["wandb_id"] if state else None},
                    tags=[config["architecture"], config["variant"]]) as run:
        run.define_metric("epoch")
        for prefix in ("train", "train_clean", "validation"):
            run.define_metric(f"{prefix}/*", step_metric="epoch")
        log_provenance(run, dataset, directory)
        (directory / "architecture.txt").write_text(str(model), encoding="utf-8")
        if args.watch:
            run.watch(model, log="all", log_freq=100)
        if state:
            restore_rng(state["rng"], augmentation_rng, loaders)
        started = time.perf_counter()
        final_epoch = config["epochs"]
        if args.epochs_per_run is not None:
            final_epoch = min(final_epoch, first_epoch + args.epochs_per_run - 1)
        for epoch in range(first_epoch, final_epoch + 1):
            if stale >= config["patience"]:
                break
            tick = time.perf_counter()
            train_metrics, _ = run_epoch(model, train_loader, class_weights, device, optimizer)
            clean_metrics, _ = run_epoch(model, clean_loader, class_weights, device)
            validation_metrics, _ = run_epoch(model, validation_loader, class_weights, device)
            row = {"epoch": epoch, "epoch_seconds": time.perf_counter() - tick, "lr": optimizer.param_groups[0]["lr"]}
            for prefix, metrics in (("train", train_metrics), ("train_clean", clean_metrics),
                                    ("validation", validation_metrics)):
                row.update({f"{prefix}/{k}": v for k, v in metrics.items() if k != "per_class"})
            row["generalization/macro_f1_gap"] = clean_metrics["macro_f1"] - validation_metrics["macro_f1"]
            history.append(row)
            value, loss = validation_metrics["macro_f1"], validation_metrics["loss"]
            if value > best or (value == best and loss < best_loss):
                best, best_loss, best_epoch = value, loss, epoch
                best_checkpoint = copy.deepcopy({"state_dict": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(), "config": config,
                            "classes": classes, "class_weights": weights.tolist(),
                            "dataset_fingerprint": dataset["fingerprint"], "source_sha256": source_fingerprint(),
                            "training_dataset_fingerprint": training_dataset["fingerprint"],
                            "epoch": epoch, "validation": validation_metrics})
            # min_delta solo controla paciencia; nunca descarta un mejor checkpoint.
            if value > patience_reference + config["min_delta"]:
                patience_reference, stale = value, 0
            else:
                stale += 1
            # Guardar localmente antes de enviar métricas: una falla de red no pierde esta época.
            save_checkpoint(directory / "last.pt", {
                "state_dict": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "config": config, "classes": classes, "dataset_fingerprint": dataset["fingerprint"],
                "training_dataset_fingerprint": training_dataset["fingerprint"],
                "source_sha256": source_fingerprint(), "epoch": epoch, "history": history,
                "best_checkpoint": best_checkpoint, "stale": stale, "patience_reference": patience_reference,
                "training_seconds": elapsed + time.perf_counter() - started,
                "rng": capture_rng(augmentation_rng, loaders), "wandb_id": run.id})
            save_checkpoint(directory / "best.pt", best_checkpoint)
            save_table(directory / "history.csv", pd.DataFrame(history))
            run.log(row)
            print(f"{config['experiment']} epoch={epoch} train_f1={clean_metrics['macro_f1']:.4f} "
                  f"val_f1={value:.4f} val_loss={loss:.4f}", flush=True)
            if stale >= config["patience"]:
                break
        if len(history) < config["epochs"] and stale < config["patience"]:
            run.summary.update({"status": "paused", "last_epoch": history[-1]["epoch"],
                                "best_validation_macro_f1": best})
            run.log({"learning_curves": wandb.Image(str(plot_history(history, directory)))})
            print(f"Pausa: {config['experiment']}, época {len(history)}. Checkpoint: {directory / 'last.pt'}", flush=True)
            return None
        checkpoint = torch.load(directory / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        validation_metrics, confusion = run_epoch(model, validation_loader, class_weights, device)
        clean_metrics, _ = run_epoch(model, clean_loader, class_weights, device)
        log_class_metrics(run, "validation_best", validation_metrics, confusion, classes, directory)
        curves = plot_history(history, directory)
        run.log({"learning_curves": wandb.Image(str(curves))})
        # Baseline de mayoría: la clase se decide con train, se mide en validation.
        baseline_cm = np.zeros_like(confusion)
        baseline_cm[:, counts.argmax()] = np.bincount(validation_data.y, minlength=len(classes))
        baseline = metrics_from_confusion(baseline_cm)
        profile = profile_model(model, device)
        result = {"experiment": config["experiment"], "architecture": config["architecture"],
                  "variant": config["variant"], "trial": config["name"], "seed": config["seed"],
                  "training_dataset_fingerprint": training_dataset["fingerprint"],
                  "training_samples": len(train_data), "original_training_samples": len(clean_data),
                  "optimizer_steps": len(train_loader) * len(history),
                  "optimizer": config["optimizer"], "lr": config["lr"], "batch_size": config["batch_size"],
                  "weight_decay": config["weight_decay"], "best_epoch": best_epoch, "epochs_ran": len(history),
                  "best_validation_macro_f1": best, "best_validation_loss": best_loss,
                  "best_validation_accuracy": validation_metrics["accuracy"],
                  "best_train_clean_macro_f1": clean_metrics["macro_f1"],
                  "macro_f1_gap": clean_metrics["macro_f1"] - best,
                  "majority_validation_macro_f1": baseline["macro_f1"],
                  "majority_validation_accuracy": baseline["accuracy"],
                  "training_seconds": elapsed + time.perf_counter() - started,
                  "checkpoint": str((directory / "best.pt").relative_to(suite)),
                  "wandb_id": run.id, "wandb_url": run.url, **profile}
        result["checkpoint_sha256"] = hashlib.sha256((directory / "best.pt").read_bytes()).hexdigest()
        run.summary.update(result)
        artifact = wandb.Artifact(f"model-{run.id}", type="model", metadata={"selection": "validation_macro_f1"})
        artifact.add_dir(str(directory))
        run.log_artifact(artifact, aliases=["best-validation"])
        write_json(directory / "result.json", result)
    del train_loader, clean_loader, validation_loader, train_data, clean_data, validation_data, model, optimizer
    return result


def completed_results(suite, grid):
    results = []
    for config in grid:
        path = suite / config["experiment"] / "result.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if (result["experiment"] != config["experiment"]
                    or sha256(suite / result["checkpoint"]) != result["checkpoint_sha256"]):
                raise ValueError("Resultado o checkpoint guardado modificado")
            results.append(result)
    return results


def show_progress(suite, grid):
    for config in grid:
        directory = suite / config["experiment"]
        if (directory / "result.json").exists():
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            status = f"complete ({result['epochs_ran']} epochs)"
        elif (directory / "last.pt").exists():
            state = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
            status = f"resumable (epoch {state['epoch']})"
        else:
            status = "pending"
        print(f"{config['experiment']}: {status}", flush=True)


def finalize_selection(suite, results, args, identity):
    winners = select_winners(results)
    ranking = lambda r: (-r["best_validation_macro_f1"], r["best_validation_loss"], r["experiment"])
    selection = {"criterion": "validation macro-F1 desc, loss asc, name asc; test never used",
                 "winners": winners, "best_overall": min(winners.values(), key=ranking)["experiment"],
                 "best_base": min([r for r in winners.values() if r["variant"] == "base"], key=ranking)["experiment"]}
    write_json(suite / "selection.json", selection)
    save_table(suite / "training_results.csv", pd.DataFrame(results))
    if (suite / "comparison.json").exists():
        return
    with wandb.init(project=args.project, entity=args.entity, mode=args.wandb_mode, dir=str(suite),
                    group=suite.name, name=f"{suite.name}-comparison", job_type="comparison", config=identity) as run:
        run.log({"all_training_runs": wandb.Table(dataframe=pd.DataFrame(results)),
                 "selected_models": wandb.Table(dataframe=pd.DataFrame(winners.values()))})
        run.summary.update({"best_overall": selection["best_overall"], "best_base": selection["best_base"]})
        artifact = wandb.Artifact(f"selection-{run.id}", type="selection")
        artifact.add_file(str(suite / "selection.json"))
        artifact.add_file(str(suite / "training_results.csv"))
        run.log_artifact(artifact)
        comparison = {"wandb_id": run.id}
    write_json(suite / "comparison.json", comparison)
    print(f"Selección congelada: {suite / 'selection.json'}\nTest: python ai/src/evaluate.py --suite {suite}")


def execute_suite(args, suite):
    previous = json.loads((suite / "suite.json").read_text(encoding="utf-8")) if (suite / "suite.json").exists() else None
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    else:
        config = previous["config"] if previous else json.loads((AI_DIR / "configs/experiments.json").read_text(encoding="utf-8"))
    grid = experiment_grid(config)
    if args.experiment and args.experiment not in {c["experiment"] for c in grid}:
        raise ValueError("Experimento desconocido. Use --list para ver los nombres disponibles")
    if args.list:
        show_progress(suite, grid)
        return
    for name, default in (("project", "onnxmobile-speech-commands"), ("entity", None),
                          ("wandb_mode", "online"), ("workers", 0), ("threads", 4), ("device", "auto")):
        if getattr(args, name) is None:
            setattr(args, name, previous[name] if previous else default)
    if args.workers < 0 or args.threads < 1 or (args.epochs_per_run is not None and args.epochs_per_run < 1):
        raise ValueError("workers >= 0, threads >= 1 y epochs-per-run >= 1")
    if args.data is None:
        if previous is None:
            raise ValueError("La primera invocación requiere --data")
        args.data = Path(json.loads((suite / "dataset.json").read_text(encoding="utf-8"))["root"])
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    dataset = inspect_dataset(args.data)
    saved_augmented = suite / "augmented_dataset.json"
    if args.augmented_data is None:
        args.augmented_data = (Path(json.loads(saved_augmented.read_text(encoding="utf-8"))["root"])
                               if saved_augmented.exists() else
                               args.data.with_name(args.data.name + "-augmented"))
    # Reutilizar la misma exportación fija en ambas arquitecturas y todos los trials.
    torch.set_num_threads(args.threads)
    augmented = prepare_augmented_dataset(args.augmented_data, dataset, config["augmentation"], config["seed"] + 1000)
    identity = {"config": config, "dataset_fingerprint": dataset["fingerprint"],
                "augmented_dataset_fingerprint": augmented["fingerprint"],
                "source_sha256": source_fingerprint(), "device": str(device),
                "workers": args.workers, "threads": args.threads,
                "project": args.project, "entity": args.entity, "wandb_mode": args.wandb_mode}
    if previous:
        if previous != identity:
            raise ValueError("No se puede reanudar: cambiaron datos, código o configuración. Los archivos se conservan")
    else:
        write_json(suite / "dataset.json", dataset)
        write_json(saved_augmented, augmented)
        write_json(suite / "suite.json", identity)
    dataset["augmented"] = augmented
    results = completed_results(suite, grid)
    completed = {result["experiment"] for result in results}
    pending = [c for c in grid if c["experiment"] not in completed]
    if args.experiment:
        selected = [c for c in pending if c["experiment"] == args.experiment]
    else:
        selected = pending if args.all else pending[:1]
    print(f"Suite: {suite}\nDevice: {device}\nComplete: {len(results)}/{len(grid)}\nThis invocation: {len(selected)}", flush=True)
    print(f"Train samples: base={dataset['counts']['train']}, augmented={augmented['counts']['train']}", flush=True)
    for experiment in selected:
        fit_one(experiment, dataset, suite, device, args)
        # Recolectar TODOS los resultados anteriores, aunque esta invocación entrenó uno solo.
        results = completed_results(suite, grid)
        if results:
            save_table(suite / "training_results.csv", pd.DataFrame(results))
        if not (suite / experiment["experiment"] / "result.json").exists():
            break
    results = completed_results(suite, grid)
    if results:
        save_table(suite / "training_results.csv", pd.DataFrame(results))
    if len(results) == len(grid):
        finalize_selection(suite, results, args, identity)
    else:
        print(f"Saved progress: {len(results)}/{len(grid)} complete. Repeat with --suite {suite}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Exportación con COMPLETED.json; solo requerida al crear la suite")
    parser.add_argument("--augmented-data", type=Path, help="Exportación aumentada; por defecto <data>-augmented")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=AI_DIR / "runs")
    parser.add_argument("--suite", "--resume-suite", dest="suite", type=Path, help="Carpeta compartida para crear o continuar experimentos")
    parser.add_argument("--project")
    parser.add_argument("--entity")
    parser.add_argument("--wandb-mode", choices=["online", "offline"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--workers", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--watch", action="store_true", help="Histogramas de parámetros y gradientes")
    parser.add_argument("--epochs-per-run", type=int, help="Pausar después de este número de épocas completas")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--experiment", help="Entrenar o reanudar un trial por nombre")
    selection.add_argument("--all", action="store_true", help="Optar explícitamente por todos los pendientes")
    selection.add_argument("--list", action="store_true", help="Ver estados sin entrenar ni conectar W&B")
    args = parser.parse_args()
    suite = (args.suite or args.output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).resolve()
    if args.list:
        execute_suite(args, suite)
        return
    suite.mkdir(parents=True, exist_ok=True)
    from experiment import suite_lock
    with suite_lock(suite):
        execute_suite(args, suite)


if __name__ == "__main__":
    main()
