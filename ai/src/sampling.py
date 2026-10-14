"""Muestreo de train con cuotas por clase y diversidad de palabras unknown."""
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

from data import make_loader, seed_worker


def validate_sampling(config):
    policy = config.get("sampling", {"strategy": "none"})
    if policy.get("strategy") not in ("none", "controlled"):
        raise ValueError("sampling.strategy debe ser none o controlled")
    if config["class_weighting"] not in ("none", "inverse_frequency"):
        raise ValueError("class_weighting no soportado")
    if policy["strategy"] == "controlled":
        fraction = policy.get("unknown_fraction")
        if (type(fraction) not in (int, float) or not math.isfinite(fraction)
                or not 0 < fraction < 1):
            raise ValueError("sampling.unknown_fraction debe estar en (0, 1)")
        if config["class_weighting"] != "none":
            raise ValueError("Muestreo controlled requiere class_weighting=none para evitar doble corrección")
    return policy


class ControlledSampler(Sampler):
    """Cuotas exactas por época; permutaciones nuevas con RNG del DataLoader.

    Cada comando alcanza el tamaño del comando mayor. Se recorren todos sus
    índices antes de repetir para completar la cuota. Unknown se estratifica
    proporcionalmente por palabra original, usando solo el manifiesto de train.
    Las cuotas son por época: los minibatches se mezclan aleatoriamente.
    """
    def __init__(self, dataset, classes, unknown_fraction, generator):
        if dataset.split != "train":
            raise ValueError("El muestreo controlado solo se aplica a train")
        validate_sampling({"sampling": {"strategy": "controlled", "unknown_fraction": unknown_fraction},
                           "class_weighting": "none"})
        if "unknown" not in classes or len(classes) < 2:
            raise ValueError("Se requieren comandos y la clase unknown")
        self.generator = generator
        self.classes = list(classes)
        self.unknown_id = classes.index("unknown")
        labels = np.asarray(dataset.y)
        counts = np.bincount(labels, minlength=len(classes))
        if len(counts) != len(classes) or (counts == 0).any():
            raise ValueError("Todas las clases deben estar presentes en train")
        commands = [i for i in range(len(classes)) if i != self.unknown_id]
        command_quota = int(counts[commands].max())
        self.quotas = np.full(len(classes), command_quota, dtype=np.int64)
        self.quotas[self.unknown_id] = max(1, round(
            command_quota * len(commands) * unknown_fraction / (1 - unknown_fraction)))
        self.pools = [(torch.from_numpy(np.flatnonzero(labels == i)), command_quota) for i in commands]
        manifest = pd.read_csv(dataset.root / "manifest.csv", keep_default_na=False)
        rows = manifest.loc[manifest.split.eq("train")].sort_values("tensor_index")
        if (not np.array_equal(rows.tensor_index, np.arange(len(labels)))
                or not np.array_equal(rows.label_id, labels)):
            raise ValueError("Manifiesto de train desalineado para muestreo")
        unknown = rows.loc[rows.label_id.eq(self.unknown_id)]
        groups = (list(unknown.groupby("source_label", sort=True)) if "source_label" in unknown
                  else [("unknown", unknown)])
        # Mayor resto: conserva proporciones y la suma exacta de la cuota.
        expected = np.array([len(group) for _, group in groups]) * (
            self.quotas[self.unknown_id] / counts[self.unknown_id])
        quotas = np.floor(expected).astype(np.int64)
        remainder = int(self.quotas[self.unknown_id] - quotas.sum())
        quotas[np.argsort(-(expected - quotas), kind="stable")[:remainder]] += 1
        self.unknown_word_quotas = {str(word): int(quota) for (word, _), quota in zip(groups, quotas)}
        self.pools.extend((torch.tensor(group.tensor_index.to_numpy()), int(quota))
                          for (_, group), quota in zip(groups, quotas))

    def __len__(self):
        return int(self.quotas.sum())

    def __iter__(self):
        selected = []
        for pool, quota in self.pools:
            while quota:
                take = min(quota, len(pool))
                selected.append(pool[torch.randperm(len(pool), generator=self.generator)[:take]])
                quota -= take
        indices = torch.cat(selected)
        return iter(indices[torch.randperm(len(indices), generator=self.generator)].tolist())

    def summary(self):
        return {"strategy": "controlled", "samples_per_epoch": len(self),
                "unknown_fraction_actual": float(self.quotas[self.unknown_id] / len(self)),
                "class_quotas": dict(zip(self.classes, self.quotas.tolist())),
                "unknown_word_quotas": self.unknown_word_quotas}


def make_training_loader(dataset, classes, config, workers=0):
    policy = validate_sampling(config)
    if policy["strategy"] == "none":
        return make_loader(dataset, config["batch_size"], config["seed"], workers, True)
    generator = torch.Generator().manual_seed(config["seed"])
    sampler = ControlledSampler(dataset, classes, policy["unknown_fraction"], generator)
    return DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler,
                      num_workers=workers, worker_init_fn=seed_worker, generator=generator)
