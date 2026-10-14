"""Reproducibilidad, artefactos W&B y visualizaciones compartidas."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")
AI_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(AI_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("WANDB_CACHE_DIR", str(AI_DIR / ".cache" / "wandb"))
os.environ.setdefault("WANDB_DATA_DIR", str(AI_DIR / ".cache" / "wandb-data"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

def write_json(path, value):
    atomic_write(path, lambda stream: stream.write(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")))


def atomic_write(path, writer):
    """Reemplazar el archivo solo después de completar y vaciar el temporal."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(path, checkpoint):
    atomic_write(path, lambda stream: torch.save(checkpoint, stream))


def save_table(path, frame):
    atomic_write(path, lambda stream: stream.write(frame.to_csv(index=False).encode("utf-8")))


@contextmanager
def suite_lock(suite):
    """Lock del SO: se libera también si el proceso muere, sin borrar checkpoints."""
    with (Path(suite) / ".training.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Ya hay un entrenamiento activo en esta suite") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def capture_rng(augmentation, loaders):
    numpy = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "augmentation": augmentation.get_state(),
            "loaders": [loader.generator.get_state() for loader in loaders]}


def restore_rng(state, augmentation, loaders):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    augmentation.set_state(state["augmentation"].cpu())
    for loader, value in zip(loaders, state["loaders"]):
        loader.generator.set_state(value.cpu())


def seed_everything(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(threads)


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted((AI_DIR / "src").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment(device):
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=AI_DIR, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=AI_DIR, text=True,
                                             stderr=subprocess.DEVNULL))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", None
    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "wandb": wandb.__version__, "numpy": np.__version__, "device": str(device),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "git_revision": revision, "git_dirty": dirty, "source_sha256": source_fingerprint(),
            "deterministic_algorithms": True, "threads": torch.get_num_threads()}


def save_environment(directory, device):
    info = environment(device)
    write_json(directory / "environment.json", info)
    packages = sorted(f"{p.metadata['Name']}=={p.version}" for p in importlib.metadata.distributions())
    (directory / "requirements-lock.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    return info


def log_provenance(run, dataset, directory):
    artifact = wandb.Artifact(f"speech-commands-{dataset['fingerprint'][:12]}", type="dataset-metadata",
                             metadata={"fingerprint": dataset["fingerprint"], "counts": dataset["counts"]})
    for filename in ("manifest.csv", "preprocessing.json", "COMPLETED.json"):
        artifact.add_file(str(Path(dataset["root"]) / filename), name=filename)
    write_json(directory / "dataset.json", dataset)
    artifact.add_file(str(directory / "dataset.json"))
    if "augmented" in dataset:
        augmented = dataset["augmented"]
        for filename in ("manifest.csv", "augmentation.json", "COMPLETED.json", "preview.png"):
            artifact.add_file(str(Path(augmented["root"]) / filename), name=f"augmented/{filename}")
    run.log_artifact(artifact)
    source = wandb.Artifact(f"training-source-{source_fingerprint()[:12]}", type="source")
    for path in sorted((AI_DIR / "src").glob("*.py")):
        source.add_file(str(path), name=f"src/{path.name}")
    source.add_file(str(AI_DIR / "notebooks" / "02_entrenamiento_evaluacion.ipynb"),
                    name="notebooks/02_entrenamiento_evaluacion.ipynb")
    source.add_file(str(directory / "environment.json"))
    source.add_file(str(directory / "requirements-lock.txt"))
    source.add_file(str(directory / "config.json"))
    run.log_artifact(source)


def log_class_metrics(run, prefix, metrics, confusion, classes, directory):
    rows = [[label, *[item[k] for k in ("precision", "recall", "f1", "support")]]
            for label, item in zip(classes, metrics["per_class"])]
    table = wandb.Table(columns=["class", "precision", "recall", "f1", "support"], data=rows)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(confusion, cmap="Blues")
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes,
           yticklabels=classes, xlabel="Predicción", ylabel="Clase real", title=prefix)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    for (row, col), count in np.ndenumerate(confusion):
        ax.text(col, row, str(count), ha="center", va="center", fontsize=7,
                color="white" if count > confusion.max() / 2 else "black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    target = directory / f"{prefix.replace('/', '_')}_confusion.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    np.savetxt(target.with_suffix(".csv"), confusion, fmt="%d", delimiter=",")
    run.log({f"{prefix}/per_class": table, f"{prefix}/confusion": wandb.Image(str(target))})


def plot_history(history, directory):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for prefix, label in (("train", "Entrenamiento (con aumento si aplica)"),
                          ("train_clean", "Entrenamiento sin aumento, eval"), ("validation", "Validación")):
        for ax, metric in zip(axes, ("loss", "macro_f1")):
            ax.plot([r["epoch"] for r in history], [r[f"{prefix}/{metric}"] for r in history], label=label)
            ax.set(xlabel="Época", ylabel=metric)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    path = directory / "learning_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def profile_model(model, device, shape=(1, 1, 64, 101), repetitions=50):
    """MACs Conv/Linear y latencia de forward batch=1; no equivalen a móvil."""
    model.eval()
    sample = torch.zeros(shape, device=device)
    macs = 0
    def hook(layer, inputs, output):
        nonlocal macs
        if isinstance(layer, torch.nn.Conv2d):
            macs += output.numel() * (layer.in_channels // layer.groups) * np.prod(layer.kernel_size)
        elif isinstance(layer, torch.nn.Linear):
            macs += output.numel() * layer.in_features
    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.inference_mode():
        model(sample)
        for handle in handles:
            handle.remove()
        for _ in range(10):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times = []
        for _ in range(repetitions):
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - start) * 1000)
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "conv_linear_macs": int(macs), "forward_median_ms": float(np.median(times)),
            "forward_p95_ms": float(np.percentile(times, 95)), "profile_device": str(device),
            "profile_threads": torch.get_num_threads(), "profile_batch_size": 1,
            "state_bytes": sum(t.numel() * t.element_size() for t in model.state_dict().values())}
