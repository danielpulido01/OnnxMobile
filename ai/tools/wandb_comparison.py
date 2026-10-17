"""Comparación remota de ejecuciones: sin PyTorch, datasets ni checkpoints locales."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import pandas as pd


METRICS = ("loss", "accuracy", "macro_f1", "balanced_accuracy")
CURVE_COLUMNS = [f"{split}/{metric}" for split in ("train", "train_clean", "validation")
                 for metric in METRICS]


def number(value):
    """Los valores ausentes/no finitos nunca se convierten en resultados cero."""
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def read_history(run):
    # scan_history no muestrea. Sin keys: W&B filtra por intersección de claves,
    # lo que eliminaría épocas de versiones que no registraron train_clean.
    rows = list(run.scan_history(page_size=1000))
    frame = pd.DataFrame(rows)
    if "epoch" not in frame:
        return pd.DataFrame(columns=["epoch", *CURVE_COLUMNS])
    for column in ["epoch", *CURVE_COLUMNS]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # Los logs de imágenes/tablas no son épocas, aunque W&B arrastre el step.
    frame = frame.dropna(subset=["epoch"])
    if "validation/macro_f1" not in frame:
        return pd.DataFrame(columns=["epoch", *CURVE_COLUMNS])
    frame = frame.dropna(subset=["validation/macro_f1"])
    columns = [c for c in ["epoch", *CURVE_COLUMNS] if c in frame]
    return frame[columns].sort_values("epoch", kind="stable").drop_duplicates("epoch", keep="last")


def training_record(run, history):
    config, summary = dict(run.config), dict(run.summary)
    environment = config.get("environment") or {}
    row = {"run_path": "/".join(run.path), "run_id": run.id, "url": run.url,
           "group": run.group, "state": run.state,
           "experiment": config.get("experiment", run.name),
           "architecture": config.get("architecture"), "variant": config.get("variant"),
           "trial": config.get("name"), "seed": config.get("seed"),
           "dataset_fingerprint": config.get("dataset_fingerprint"),
           "host": environment.get("gpu") or environment.get("platform"),
           "source_sha256": environment.get("source_sha256"),
           "loss_definition": config.get("loss"),
           "training_dataset_fingerprint": config.get("training_dataset_fingerprint")}
    for key in ("optimizer", "lr", "batch_size", "epochs", "weight_decay", "width_mult"):
        row[key] = config.get(key)
    for key in ("best_epoch", "epochs_ran", "best_validation_macro_f1", "best_validation_loss",
                "best_validation_accuracy", "best_train_clean_macro_f1", "macro_f1_gap",
                "majority_validation_macro_f1", "majority_validation_accuracy", "parameters",
                "conv_linear_macs", "state_bytes", "training_seconds", "forward_median_ms",
                "forward_p95_ms", "training_samples", "samples_per_epoch", "optimizer_steps"):
        row[key] = number(summary.get(key))
    row["profile_device"] = summary.get("profile_device")
    row["checkpoint_sha256"] = summary.get("checkpoint_sha256")
    # summary validation/* corresponde a la última época, no al mejor checkpoint.
    # Solo completar desde la fila de best_epoch, nunca desde el último summary.
    best_rows = history[history["epoch"] == row["best_epoch"]]
    if not best_rows.empty:
        best = best_rows.iloc[-1]
        for metric in METRICS:
            key = f"best_validation_{metric}"
            if row.get(key) is None:
                row[key] = number(best.get(f"validation/{metric}"))
    if row["epochs_ran"] is None and not history.empty:
        row["epochs_ran"] = number(history["epoch"].max())
    for key in ("last_macro_f1_gap", "tail_validation_loss_change", "tail_train_clean_loss_change"):
        row[key] = None
    if not history.empty:
        last = history.iloc[-1]
        train_f1, val_f1 = number(last.get("train_clean/macro_f1")), number(last.get("validation/macro_f1"))
        if train_f1 is not None and val_f1 is not None:
            row["last_macro_f1_gap"] = train_f1 - val_f1
        tail = history.tail(3)
        if len(tail) >= 2:
            for split in ("validation", "train_clean"):
                column = f"{split}/loss"
                if column in tail and tail[column].notna().all():
                    row[f"tail_{split}_loss_change"] = number(tail[column].iloc[-1] - tail[column].iloc[0])
    reasons = []
    if run.state != "finished":
        reasons.append("ejecución no finalizada")
    expected, actual = number(row["epochs"]), row["epochs_ran"]
    if expected is None or expected < 1 or actual != expected:
        reasons.append("no se verificaron todas las épocas configuradas")
    if (row["best_epoch"] is None or expected is None
            or not 1 <= row["best_epoch"] <= expected):
        reasons.append("best_epoch ausente o inválida")
    if row["best_validation_macro_f1"] is None or not 0 <= row["best_validation_macro_f1"] <= 1:
        reasons.append("macro-F1 del mejor checkpoint ausente o inválido")
    if row["best_validation_loss"] is None or row["best_validation_loss"] < 0:
        reasons.append("loss del mejor checkpoint ausente o inválida")
    if row["architecture"] not in ("lenet5", "mobilenet_v1") or row["variant"] not in ("base", "augmented"):
        reasons.append("arquitectura/variante desconocida")
    weights = config.get("class_weights")
    if (not row["dataset_fingerprint"] or not row["loss_definition"]
            or not isinstance(weights, list) or not weights
            or any(number(w) is None or number(w) <= 0 for w in weights)):
        reasons.append("falta procedencia del dataset o definición de la pérdida")
        row["cohort"] = None
    else:
        signature = {"dataset": row["dataset_fingerprint"], "loss": row["loss_definition"],
                     "class_weights": [float(w) for w in weights]}
        row["cohort"] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    row["eligible"] = not reasons
    row["exclusion_reason"] = "; ".join(reasons)
    return row


def collect_runs(api, sources):
    """Leer proyectos/grupos de ambas laptops, deduplicando por ruta remota."""
    records, tests, histories, remote, warnings = [], [], {}, {}, []
    for source in sources:
        entity, project = source["entity"], source["project"]
        if not entity or not project:
            raise ValueError("Complete entity y project en SOURCES (entity es el usuario o equipo de W&B).")
        filters = {"group": {"$in": source["groups"]}} if source.get("groups") else {}
        for run in api.runs(f"{entity}/{project}", filters=filters, per_page=100):
            path = "/".join(run.path)
            if path in remote:
                continue
            remote[path] = run
            if run.job_type == "test":
                tests.append({"run_path": path, "run": run})
                continue
            if run.job_type != "train":
                continue
            try:
                history = read_history(run)
            except Exception as error:
                warnings.append(f"{path}: historial no disponible ({type(error).__name__}).")
                history = pd.DataFrame(columns=["epoch", *CURVE_COLUMNS])
            histories[path] = history
            records.append(training_record(run, history))
    if not records:
        raise ValueError("No se encontraron runs job_type='train'. Revise proyectos, grupos, acceso y sincronización.")
    return pd.DataFrame(records), tests, histories, remote, warnings


def rank_runs(records, cohort=None, dataset_fingerprint=None):
    candidates = records[records["eligible"]].copy()
    if dataset_fingerprint:
        candidates = candidates[candidates["dataset_fingerprint"] == dataset_fingerprint]
    if cohort:
        candidates = candidates[candidates["cohort"] == cohort]
    if candidates.empty:
        raise ValueError("No hay entrenamientos completos comparables. Revise la tabla de exclusiones y los filtros.")
    if candidates["cohort"].nunique() != 1:
        raise ValueError("Hay distintos datasets o pérdidas. Elija COHORT de la tabla antes de declarar un ganador.")
    ranking = candidates.sort_values(
        ["best_validation_macro_f1", "best_validation_loss", "experiment", "run_path"],
        ascending=[False, True, True, True], kind="stable").reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    winners = ranking.drop_duplicates(["architecture", "variant"])
    return ranking, winners


def match_tests(ranking, tests):
    """Test describe al checkpoint ya seleccionado; nunca modifica el ranking."""
    rows, warnings = [], []
    for item in tests:
        run = item["run"]
        config, summary = dict(run.config), dict(run.summary)
        training_path = "/".join([*run.path[:2], str(config.get("training_run_id", ""))])
        selected = ranking[ranking["run_path"] == training_path]
        if selected.empty:
            continue
        train = selected.iloc[0]
        if (run.state != "finished" or not train["checkpoint_sha256"]
                or config.get("checkpoint_sha256") != train["checkpoint_sha256"]
                or config.get("dataset_fingerprint") != train["dataset_fingerprint"]):
            warnings.append(f"{item['run_path']}: test excluido por estado, dataset o checkpoint incompatible.")
            continue
        row = {"training_run_path": training_path, "test_run_path": item["run_path"],
               "experiment": train["experiment"], "url": run.url}
        for metric in METRICS:
            value = summary.get(f"test/{metric}")
            row[f"test/{metric}"] = number(value if value is not None else summary.get(metric))
        rows.append(row)
    return pd.DataFrame(rows), warnings


def download_media(run, key, root):
    """Descargar solo una imagen o tabla registrada, nunca el artefacto del modelo."""
    metadata = dict(run.summary).get(key)
    if not isinstance(metadata, dict) or not metadata.get("path"):
        return None
    relative = PurePosixPath(metadata["path"])
    # Validar las rutas antes de pasarlas al downloader del SDK.
    if (relative.is_absolute() or ".." in relative.parts or "\\" in str(relative)
            or ":" in str(relative) or relative.parts[0] != "media"):
        raise ValueError("Ruta de media W&B inválida")
    target = Path(root).resolve().joinpath(*run.path)
    target.mkdir(parents=True, exist_ok=True)
    with run.file(str(relative)).download(root=str(target), replace=True) as downloaded:
        path = Path(downloaded.name).resolve()
    if not path.is_relative_to(target.resolve()):
        raise ValueError("Archivo fuera de la carpeta de media")
    return path


def read_class_table(path):
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    return pd.DataFrame(table["data"], columns=table["columns"])
