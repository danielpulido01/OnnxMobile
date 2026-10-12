"""Test de los cuatro ganadores ya seleccionados, con registro obligatorio W&B."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch
import wandb

from augment import inspect_augmented_dataset
from data import inspect_dataset, make_loader, sha256, SpectrogramDataset
from experiment import log_class_metrics, profile_model, seed_everything, source_fingerprint, write_json
from metrics import run_epoch
from models import build_model


def write_report(suite, selection, tests):
    training = pd.read_csv(suite / "training_results.csv")
    lines = ["# Comparación de experimentos", "", "La selección se fijó con validación antes de abrir test.",
             "Los datos de esta tabla provienen de las ejecuciones, sin resultados inventados.", "",
             "## Todos los entrenamientos", "",
             "| Experimento | Época elegida | Train F1 limpio | Validación F1 | Validación loss | Brecha F1 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in training.to_dict("records"):
        lines.append(f"| {r['experiment']} | {r['best_epoch']} | {r['best_train_clean_macro_f1']:.4f} | "
                     f"{r['best_validation_macro_f1']:.4f} | {r['best_validation_loss']:.4f} | {r['macro_f1_gap']:.4f} |")
    lines += ["", "## Cuatro ganadores en test", "",
              "| Experimento | Accuracy | Macro-F1 | Balanced accuracy | Parámetros | MACs Conv/Linear | CPU mediana ms |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in tests:
        lines.append(f"| {r['experiment']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | "
                     f"{r['balanced_accuracy']:.4f} | {r['parameters']} | {r['conv_linear_macs']} | "
                     f"{r['forward_median_ms']:.3f} |")
    lines += ["", f"Mejor global por validación: **{selection['best_overall']}**.",
              f"Mejor modelo base por validación: **{selection['best_base']}**.", "",
              "## Efecto del aumento (test, diferencia aumentado menos base)", ""]
    for architecture in ("lenet5", "mobilenet_v1"):
        pair = {r["variant"]: r for r in tests if r["architecture"] == architecture}
        delta = pair["augmented"]["macro_f1"] - pair["base"]["macro_f1"]
        lines.append(f"- {architecture}: Δ macro-F1 = {delta:+.4f}.")
    lines += ["", "## Comparación con los mismos hiperparámetros (validación)", "",
              "| Arquitectura | Trial | Train base | Train aumentado | Δ macro-F1 aumentado − base |",
              "| --- | --- | ---: | ---: | ---: |"]
    for (architecture, trial), group in training.groupby(["architecture", "trial"]):
        pair = group.set_index("variant")
        base, augmented = pair.loc["base"], pair.loc["augmented"]
        delta = augmented.best_validation_macro_f1 - base.best_validation_macro_f1
        lines.append(f"| {architecture} | {trial} | {base.training_samples} | {augmented.training_samples} | {delta:+.4f} |")
    lines += ["", "El conjunto aumentado incluye originales y copias fijas con máscaras. Con el mismo límite de "
              "épocas recorre más ejemplos y realiza más pasos de optimización; training_results.csv registra "
              "training_samples y optimizer_steps. Esta comparación mide la receta completa de aumento, "
              "sin aislar su efecto del presupuesto de cómputo. Los ganadores en test pueden usar distintos "
              "hiperparámetros elegidos exclusivamente por validación."]
    lines += ["", "## Evidencia de ajuste y límites", ""]
    for r in training.to_dict("records"):
        history = pd.read_csv(suite / r["experiment"] / "history.csv")
        tail = history.tail(min(3, len(history)))
        loss_change = tail["validation/loss"].iloc[-1] - tail["validation/loss"].iloc[0]
        lines.append(f"- {r['experiment']}: brecha train limpio − validación = {r['macro_f1_gap']:+.4f}; "
                     f"cambio de loss de validación en las últimas {len(tail)} épocas = {loss_change:+.4f}; "
                     f"macro-F1 de mayoría = {r['majority_validation_macro_f1']:.4f}; "
                     f"[curvas]({r['experiment']}/learning_curves.png).")
    lines += ["", "Una brecha creciente con deterioro de validación sugiere sobreajuste. Rendimiento bajo en "
              "ambas particiones y curvas estancadas sugiere subajuste; se debe interpretar junto con el baseline "
              "y los errores por clase. Estas señales no prueban por sí solas ninguna de las dos condiciones.", "",
              "La pérdida de train puede incluir aumento; para analizar la brecha se usa train_clean, calculado "
              "sin aumento, en eval(), con el mismo checkpoint y la misma pérdida ponderada que validación.", "",
              "La latencia CPU corresponde a forward PyTorch con batch=1 en esta computadora. No mide "
              "preprocesamiento ni latencia ONNX en un teléfono. MACs cuenta únicamente Conv2d y Linear. "
              "Medir memoria, consumo y latencia en el dispositivo objetivo antes de elegir un despliegue.", "",
              "Un entrenamiento por configuración no estima variabilidad entre semillas; los doce ensayos "
              "usan la misma semilla para comparar hiperparámetros. Para intervalos de confianza se requieren "
              "repeticiones adicionales. No ajustar hiperparámetros después de observar test.", ""]
    (suite / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    suite = args.suite.resolve()
    identity = json.loads((suite / "suite.json").read_text(encoding="utf-8"))
    dataset_saved = json.loads((suite / "dataset.json").read_text(encoding="utf-8"))
    selection = json.loads((suite / "selection.json").read_text(encoding="utf-8"))
    if (suite / "test_results.json").exists():
        raise ValueError("Test ya evaluado. Reutilice test_results.json y report.md")
    from train import select_winners
    training = pd.read_csv(suite / "training_results.csv").to_dict("records")
    if source_fingerprint() != identity["source_sha256"]:
        raise ValueError("El código cambió desde entrenamiento; use el snapshot registrado")
    expected_count = 4 * len(identity["config"]["trials"])
    if len(training) != expected_count or len({r["experiment"] for r in training}) != expected_count:
        raise ValueError("La suite de entrenamiento está incompleta")
    expected = select_winners(training)
    if set(selection["winners"]) != set(expected):
        raise ValueError("Selección de grupos inconsistente")
    for group, winner in expected.items():
        selected = selection["winners"][group]
        if selected["experiment"] != winner["experiment"] or selected["checkpoint_sha256"] != winner["checkpoint_sha256"]:
            raise ValueError("La selección no coincide con los ganadores de validación")
    ranking = lambda r: (-r["best_validation_macro_f1"], r["best_validation_loss"], r["experiment"])
    if (selection["best_overall"] != min(expected.values(), key=ranking)["experiment"]
            or selection["best_base"] != min([r for r in expected.values() if r["variant"] == "base"], key=ranking)["experiment"]):
        raise ValueError("La selección global/base no coincide con validación")
    dataset = inspect_dataset(dataset_saved["root"])
    if dataset["fingerprint"] != identity["dataset_fingerprint"]:
        raise ValueError("El dataset cambió desde entrenamiento")
    augmented_saved = json.loads((suite / "augmented_dataset.json").read_text(encoding="utf-8"))
    augmented = inspect_augmented_dataset(augmented_saved["root"], dataset, identity["config"]["augmentation"],
                                          identity["config"]["seed"] + 1000)
    if augmented["fingerprint"] != identity["augmented_dataset_fingerprint"]:
        raise ValueError("El dataset aumentado cambió desde entrenamiento")
    device = torch.device(args.device)
    seed_everything(identity["config"]["seed"], identity["threads"])
    classes = dataset["metadata"]["classes"]
    tests = []
    for group, winner in selection["winners"].items():
        directory = suite / winner["experiment"]
        path = suite / winner["checkpoint"]
        if sha256(path) != winner["checkpoint_sha256"]:
            raise ValueError("Checkpoint modificado desde selección")
        # Permite retomar una evaluación interrumpida sin repetir los tests completos.
        if (directory / "test_result.json").exists():
            cached = json.loads((directory / "test_result.json").read_text(encoding="utf-8"))
            if cached["checkpoint_sha256"] != winner["checkpoint_sha256"]:
                raise ValueError("Resultado test de otro checkpoint")
            tests.append(cached)
            continue
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint["dataset_fingerprint"] != dataset["fingerprint"] or checkpoint["classes"] != classes:
            raise ValueError("Checkpoint incompatible con dataset")
        config = checkpoint["config"]
        training_dataset = augmented if config["variant"] == "augmented" else dataset
        if checkpoint["training_dataset_fingerprint"] != training_dataset["fingerprint"]:
            raise ValueError("Checkpoint incompatible con dataset de entrenamiento")
        model = build_model(config["architecture"], len(classes), config["width_mult"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        loader = make_loader(SpectrogramDataset(dataset["root"], "test"), config["batch_size"], config["seed"])
        with wandb.init(project=identity["project"], entity=identity["entity"], mode=identity["wandb_mode"],
                        dir=str(suite), group=suite.name, name=f"{winner['experiment']}-test", job_type="test",
                        config={"training_run_id": winner["wandb_id"], "checkpoint_sha256": winner["checkpoint_sha256"],
                                "dataset_fingerprint": dataset["fingerprint"], "selection": selection["criterion"],
                                "architecture": config["architecture"], "variant": config["variant"]}) as run:
            weights = torch.tensor(checkpoint["class_weights"], dtype=torch.float32, device=device)
            metrics, confusion = run_epoch(model, loader, weights, device)
            run.log({f"test/{k}": v for k, v in metrics.items() if k != "per_class"})
            log_class_metrics(run, "test", metrics, confusion, classes, directory)
            profile = profile_model(model.cpu(), torch.device("cpu"))
            result = {"experiment": winner["experiment"], "architecture": config["architecture"],
                      "variant": config["variant"], **metrics, **profile,
                      "checkpoint_sha256": winner["checkpoint_sha256"], "wandb_id": run.id,
                      "profile_environment": {"platform": __import__("platform").platform(), "torch": torch.__version__}}
            run.summary.update({k: v for k, v in result.items() if k != "per_class"})
            write_json(directory / "test_result.json", result)
            artifact = wandb.Artifact(f"test-{run.id}", type="evaluation")
            artifact.add_file(str(directory / "test_result.json"))
            artifact.add_file(str(suite / "selection.json"))
            run.log_artifact(artifact)
            tests.append(result)
        del loader, model
    write_json(suite / "test_results.json", tests)
    frame = pd.DataFrame([{k: v for k, v in r.items() if k not in ("per_class", "profile_environment")} for r in tests])
    frame.to_csv(suite / "test_results.csv", index=False)
    write_report(suite, selection, tests)
    with wandb.init(project=identity["project"], entity=identity["entity"], mode=identity["wandb_mode"],
                    dir=str(suite), group=suite.name, name=f"{suite.name}-final-report", job_type="report") as run:
        run.log({"test_comparison": wandb.Table(dataframe=frame),
                 "all_training_runs": wandb.Table(dataframe=pd.DataFrame(training))})
        report = wandb.Artifact(f"report-{run.id}", type="report")
        for filename in ("report.md", "test_results.csv", "test_results.json", "selection.json", "training_results.csv"):
            report.add_file(str(suite / filename))
        run.log_artifact(report)
    print(f"Informe: {suite / 'report.md'}")


if __name__ == "__main__":
    main()
