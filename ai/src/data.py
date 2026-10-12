"""Contrato del notebook, auditoría de particiones y aumento espectral."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_dataset(root):
    root = Path(root).resolve()
    completion = json.loads((root / "COMPLETED.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
    if sha256(root / "manifest.csv") != completion["manifest_sha256"]:
        raise ValueError("El manifiesto no coincide con COMPLETED.json")
    if metadata["tensor_layout"] != "NHWC" or metadata["image_shape"] != [64, 101, 1]:
        raise ValueError("Se esperan espectrogramas NHWC 64x101x1")
    classes = metadata["classes"]
    if metadata["label_to_id"] != {label: i for i, label in enumerate(classes)}:
        raise ValueError("Orden de clases inconsistente")
    manifest = pd.read_csv(root / "manifest.csv", keep_default_na=False)
    if set(manifest.split) != {"train", "validation", "test"}:
        raise ValueError("Se requieren train, validation y test")
    for column in ("speaker", "file_sha256", "audio_sha256"):
        if column not in manifest:
            raise ValueError(f"Falta la columna de auditoría {column}")
        if manifest.groupby(column).split.nunique().gt(1).any():
            raise ValueError(f"Fuga entre particiones: {column}")
    hashes = {p: sha256(root / p) for p in ("manifest.csv", "preprocessing.json", "COMPLETED.json")}
    for split in ("train", "validation", "test"):
        rows = manifest.loc[manifest.split.eq(split)].sort_values("tensor_index")
        x = np.load(root / f"X_{split}.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(root / f"y_{split}.npy", mmap_mode="r", allow_pickle=False)
        if (x.shape != (len(rows), 64, 101, 1) or x.dtype != np.float32
                or y.shape != (len(rows),) or y.dtype != np.int64 or len(rows) == 0):
            raise ValueError(f"Forma o dtype inválido en {split}")
        if len(rows) != completion["counts"][split]:
            raise ValueError(f"Conteo inconsistente en {split}")
        if not np.array_equal(rows.tensor_index, np.arange(len(rows))) or not np.array_equal(y, rows.label_id):
            raise ValueError(f"Índices/etiquetas inconsistentes en {split}")
        if (y < 0).any() or (y >= len(classes)).any():
            raise ValueError("Etiqueta fuera del orden de clases")
        if not np.array_equal(rows.label, np.asarray(classes)[y]):
            raise ValueError("Nombres de clase inconsistentes")
        del x, y
        # Hash de bytes, sin inferencia ni selección usando test.
        for prefix in ("X", "y"):
            name = f"{prefix}_{split}.npy"
            hashes[name] = sha256(root / name)
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"root": str(root), "metadata": metadata, "hashes": hashes, "fingerprint": fingerprint,
            "counts": completion["counts"]}


class SpectrogramDataset(Dataset):
    def __init__(self, root, split):
        self.root, self.split = Path(root), split
        self.y = np.load(self.root / f"y_{split}.npy", allow_pickle=False)
        self.x = None  # Abrir mmap en cada proceso (también con spawn de Windows).

    def __len__(self):
        return len(self.y)

    def __getstate__(self):
        return {**self.__dict__, "x": None}

    def __getitem__(self, index):
        if self.x is None:
            self.x = np.load(self.root / f"X_{self.split}.npy", mmap_mode="r", allow_pickle=False)
        x = torch.from_numpy(np.array(self.x[index], copy=True)).permute(2, 0, 1)
        if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError(f"Espectrograma inválido: {self.split}:{index}")
        return x, int(self.y[index])


def spec_augment(batch, config, generator):
    """Time/frequency masking inspirado en SpecAugment; no time warping.

    RNG independiente: no altera orden de batches ni inicialización del modelo.
    Aplicar en CPU, exclusivamente a ejemplos de entrenamiento.
    Los anchos activos son positivos para evitar copias sin ninguna máscara.
    """
    result = batch.clone()
    for sample in result:
        for dimension, prefix in ((1, "frequency"), (2, "time")):
            maximum = min(config[f"{prefix}_width"], sample.shape[dimension] - 1)
            if maximum == 0:
                continue
            for _ in range(config[f"{prefix}_masks"]):
                width = int(torch.randint(1, maximum + 1, (), generator=generator))
                start = int(torch.randint(sample.shape[dimension] - width + 1, (), generator=generator))
                slices = [slice(None)] * 3
                slices[dimension] = slice(start, start + width)
                sample[tuple(slices)] = 0.0  # Piso espectral normalizado: -80 dB.
    return result


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, batch_size, seed, workers=0, shuffle=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      worker_init_fn=seed_worker, generator=generator)
