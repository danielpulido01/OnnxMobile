"""Exportar train original + copias SpecAugment; compartir validation/test limpios."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch

from data import inspect_dataset, sha256, spec_augment
from experiment import AI_DIR, write_json


def validate_policy(policy):
    for key in ("frequency_masks", "frequency_width", "time_masks", "time_width", "copies_per_sample"):
        value = policy.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"augmentation.{key} debe ser un entero no negativo")
    if policy["copies_per_sample"] < 1:
        raise ValueError("copies_per_sample debe ser al menos 1")
    if not any(policy[f"{axis}_masks"] and policy[f"{axis}_width"] for axis in ("time", "frequency")):
        raise ValueError("Se requiere al menos una técnica de masking activa")


def _techniques(policy):
    return [f"{axis}_masking" for axis in ("time", "frequency")
            if policy[f"{axis}_masks"] and policy[f"{axis}_width"]]


def _manifest(source_root, policy):
    original = pd.read_csv(Path(source_root) / "manifest.csv", keep_default_na=False)
    original = original.loc[original.split.eq("train")].sort_values("tensor_index").reset_index(drop=True)
    # Los PNG existentes representan el original, nunca una copia enmascarada.
    original = original.rename(columns={"image_path": "source_image_path"})
    original["source_tensor_index"] = original.tensor_index
    blocks = []
    for copy in range(policy["copies_per_sample"] + 1):
        block = original.copy()
        block["tensor_index"] = np.arange(len(original)) + copy * len(original)
        block["augmentation_copy"] = copy
        block["technique"] = "+".join(_techniques(policy)) if copy else "original"
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def inspect_augmented_dataset(root, source, policy, seed):
    """Comprobar bytes y linaje contra el train auditado, sin abrir test para inferencia."""
    validate_policy(policy)
    root = Path(root).resolve()
    metadata = json.loads((root / "augmentation.json").read_text(encoding="utf-8"))
    completion = json.loads((root / "COMPLETED.json").read_text(encoding="utf-8"))
    count = source["counts"]["train"] * (policy["copies_per_sample"] + 1)
    if (metadata["format"] != "onnxmobile-augmented-train-v1"
            or metadata["source_fingerprint"] != source["fingerprint"]
            or metadata["policy"] != policy or metadata["seed"] != seed
            or metadata["classes"] != source["metadata"]["classes"]
            or metadata["counts"] != {**source["counts"], "train": count}
            or metadata["tensor_layout"] != "NHWC" or metadata["image_shape"] != [64, 101, 1]):
        raise ValueError("Dataset aumentado incompatible con datos, política o semilla")
    names = ("X_train.npy", "y_train.npy", "manifest.csv", "augmentation.json", "preview.png")
    hashes = {name: sha256(root / name) for name in names}
    if hashes != completion["hashes"] or completion["counts"] != metadata["counts"]:
        raise ValueError("Dataset aumentado incompleto o modificado")
    actual = pd.read_csv(root / "manifest.csv", keep_default_na=False)
    expected = _manifest(source["root"], policy)
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    except AssertionError as error:
        raise ValueError("Linaje aumentado inconsistente con train original") from error
    x = np.load(root / "X_train.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(root / "y_train.npy", allow_pickle=False)
    if (x.shape != (count, 64, 101, 1) or x.dtype != np.float32
            or y.dtype != np.int64 or not np.array_equal(y, expected.label_id.to_numpy())):
        raise ValueError("Tensores aumentados inválidos")
    del x, y
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"root": str(root), "fingerprint": fingerprint, "hashes": hashes,
            "metadata": metadata, "counts": metadata["counts"]}


def _preview(root, count, copies):
    import matplotlib.pyplot as plt

    x = np.load(root / "X_train.npy", mmap_mode="r", allow_pickle=False)
    # Usar solo un ejemplo de train con energía; nunca elegir ejemplos de test.
    index = next((i for i in range(min(count, 100)) if np.any(x[i] > 0)), 0)
    shown = min(copies, 2)
    figure, axes = plt.subplots(1, shown + 1, figsize=(5 * (shown + 1), 4), squeeze=False)
    for copy, axis in enumerate(axes[0]):
        plot = axis.imshow(x[copy * count + index, :, :, 0], origin="lower", aspect="auto",
                           vmin=0, vmax=1, cmap="magma")
        axis.set(title="Original" if copy == 0 else f"SpecAugment · copia {copy}",
                 xlabel="Tiempo (frames)", ylabel="Frecuencia (bandas Mel)")
    figure.colorbar(plot, ax=axes.ravel().tolist(), label="Intensidad normalizada")
    figure.savefig(root / "preview.png", dpi=120, bbox_inches="tight")
    plt.close(figure)
    del x


def prepare_augmented_dataset(root, source, policy, seed, batch_size=256):
    """Crear una vez, sin sobrescribir; publicar solo una exportación completa."""
    validate_policy(policy)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size debe ser positivo")
    root = Path(root).resolve()
    if root.exists():
        return inspect_augmented_dataset(root, source, policy, seed)
    root.parent.mkdir(parents=True, exist_ok=True)
    count, copies = source["counts"]["train"], policy["copies_per_sample"]
    print(f"Exportando train aumentado: {count} originales + {copies * count} copias -> {root}", flush=True)
    with tempfile.TemporaryDirectory(prefix=f".{root.name}-", dir=root.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        original = np.load(Path(source["root"]) / "X_train.npy", mmap_mode="r", allow_pickle=False)
        generator = torch.Generator().manual_seed(seed)
        try:
            # Escritura secuencial: no mapear los 6+ GB de salida en memoria.
            with (staging / "X_train.npy").open("wb") as output:
                np.lib.format.write_array_header_1_0(output, {
                    "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
                    "fortran_order": False, "shape": (count * (copies + 1), 64, 101, 1)})
                for copy in range(copies + 1):
                    for start in range(0, count, batch_size):
                        stop = min(start + batch_size, count)
                        batch = np.array(original[start:stop], copy=True)
                        if not np.isfinite(batch).all() or batch.min() < 0 or batch.max() > 1:
                            raise ValueError("Espectrograma original inválido en train")
                        if copy:
                            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2)
                            batch = spec_augment(tensor, policy, generator).permute(0, 2, 3, 1).numpy()
                        output.write(batch.tobytes(order="C"))
                    print(f"  Bloque {copy + 1}/{copies + 1} guardado", flush=True)
        finally:
            del original
        labels = np.load(Path(source["root"]) / "y_train.npy", allow_pickle=False)
        np.save(staging / "y_train.npy", np.tile(labels, copies + 1))
        _manifest(source["root"], policy).to_csv(staging / "manifest.csv", index=False)
        metadata = {"format": "onnxmobile-augmented-train-v1", "source_root": source["root"],
                    "source_fingerprint": source["fingerprint"], "source_counts": source["counts"],
                    "classes": source["metadata"]["classes"], "image_shape": [64, 101, 1],
                    "tensor_layout": "NHWC", "policy": policy, "seed": seed,
                    "techniques": _techniques(policy), "includes_originals": True,
                    "counts": {**source["counts"], "train": count * (copies + 1)},
                    "validation_test": "shared unchanged from source_root",
                    "torch": str(torch.__version__), "numpy": np.__version__,
                    "generator_sha256": {name: sha256(Path(__file__).with_name(name))
                                         for name in ("augment.py", "data.py")}}
        write_json(staging / "augmentation.json", metadata)
        _preview(staging, count, copies)
        hashes = {name: sha256(staging / name) for name in
                  ("X_train.npy", "y_train.npy", "manifest.csv", "augmentation.json", "preview.png")}
        write_json(staging / "COMPLETED.json", {"counts": metadata["counts"], "hashes": hashes})
        inspect_augmented_dataset(staging, source, policy, seed)
        staging.rename(root)
    return inspect_augmented_dataset(root, source, policy, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Exportación original con COMPLETED.json")
    parser.add_argument("--output", type=Path, help="Carpeta nueva; se verifica y reutiliza si ya existe")
    parser.add_argument("--config", type=Path, default=AI_DIR / "configs/experiments.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = inspect_dataset(args.data)
    output = args.output or Path(source["root"]).with_name(Path(source["root"]).name + "-augmented")
    torch.set_num_threads(2)
    dataset = prepare_augmented_dataset(output, source, config["augmentation"], config["seed"] + 1000)
    print(f"Dataset original: {source['root']} ({source['counts']['train']} train)")
    print(f"Dataset aumentado: {dataset['root']} ({dataset['counts']['train']} train)")
    print("Validation/test se leen del dataset original sin modificaciones.")


if __name__ == "__main__":
    main()
