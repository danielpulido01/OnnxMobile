"""Validación real con datos sintéticos locales; no son resultados académicos."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_DIR / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch

from data import inspect_dataset, sha256, spec_augment
from experiment import seed_everything, write_json, atomic_write, save_checkpoint
from metrics import metrics_from_confusion, run_epoch
from models import build_model, DepthwiseSeparable, LeNet5, PartialC3
from train import experiment_grid, select_winners


def synthetic_dataset(root):
    root.mkdir()
    rng = np.random.default_rng(123)
    classes = ["yes", "no", "unknown"]
    rows, counts = [], {}
    for split, per_class in (("train", 4), ("validation", 2), ("test", 2)):
        labels = np.tile(np.arange(3, dtype=np.int64), per_class)
        features = rng.random((len(labels), 64, 101, 1), dtype=np.float32)
        np.save(root / f"X_{split}.npy", features)
        np.save(root / f"y_{split}.npy", labels)
        counts[split] = len(labels)
        for i, label in enumerate(labels):
            rows.append({"split": split, "tensor_index": i, "label_id": int(label),
                         "label": classes[label], "speaker": f"{split}-{i}",
                         "file_sha256": f"file-{split}-{i}", "audio_sha256": f"audio-{split}-{i}"})
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    write_json(root / "preprocessing.json", {"classes": classes, "label_to_id": {c: i for i, c in enumerate(classes)},
                                             "tensor_layout": "NHWC", "image_shape": [64, 101, 1]})
    write_json(root / "COMPLETED.json", {"counts": counts, "manifest_sha256": sha256(root / "manifest.csv")})


class TrainingTest(unittest.TestCase):
    def setUp(self):
        seed_everything(42, 2)

    def test_architectures_and_gradients(self):
        self.assertEqual(sum(p.numel() for p in PartialC3().parameters()), 1516)
        self.assertEqual(sum(p.numel() for p in LeNet5(11).parameters()), 60000)
        for architecture in ("lenet5", "mobilenet_v1"):
            model = build_model(architecture, 11)
            inputs = torch.rand(2, 1, 64, 101)
            logits = model(inputs)
            self.assertEqual(tuple(logits.shape), (2, 11))
            torch.nn.functional.cross_entropy(logits, torch.tensor([0, 10])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            model.eval()
            self.assertEqual(tuple(model(inputs[:1]).shape), (1, 11))
            if architecture == "mobilenet_v1":
                blocks = [m for m in model.modules() if isinstance(m, DepthwiseSeparable)]
                self.assertEqual(len(blocks), 13)
                self.assertTrue(all(b[0][0].groups == b[0][0].in_channels for b in blocks))

    def test_augmentation_reproducibility_and_isolation(self):
        original = torch.ones(4, 1, 64, 101)
        policy = {"frequency_masks": 2, "frequency_width": 8, "time_masks": 2, "time_width": 10}
        a = spec_augment(original, policy, torch.Generator().manual_seed(42))
        b = spec_augment(original, policy, torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(original.eq(1).all())
        self.assertTrue(a.eq(0).any())
        model = build_model("lenet5", 3)
        loader = [(original[:2], torch.tensor([0, 1]))]
        before = copy.deepcopy(model.state_dict())
        clean, cm1 = run_epoch(model, loader, torch.ones(3), torch.device("cpu"))
        other, cm2 = run_epoch(model, loader, torch.ones(3), torch.device("cpu"), augmentation=policy)
        self.assertEqual(clean, other)  # No se aplica aumento en validación/test.
        np.testing.assert_array_equal(cm1, cm2)
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))

    def test_metric_math_and_weighted_loss(self):
        metrics = metrics_from_confusion([[2, 1], [0, 1]])
        self.assertAlmostEqual(metrics["accuracy"], .75)
        self.assertAlmostEqual(metrics["macro_f1"], (.8 + 2/3) / 2)
        self.assertAlmostEqual(metrics["balanced_accuracy"], (2/3 + 1) / 2)
        logits = torch.tensor([[3., 0.], [0., 2.], [1., 0.]])
        labels, weights = torch.tensor([0, 1, 1]), torch.tensor([1., 5.])
        model = torch.nn.Identity()
        actual, _ = run_epoch(model, [(logits[:2], labels[:2]), (logits[2:], labels[2:])], weights, torch.device("cpu"))
        expected = torch.nn.functional.cross_entropy(logits, labels, weight=weights).item()
        self.assertAlmostEqual(actual["loss"], expected, places=6)

    def test_repeatable_training_step(self):
        states = []
        for _ in range(2):
            seed_everything(42, 2)
            model = build_model("lenet5", 3)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            batch = [(torch.rand(4, 1, 64, 101), torch.tensor([0, 1, 2, 0]))]
            run_epoch(model, batch, torch.ones(3), torch.device("cpu"), optimizer)
            states.append(copy.deepcopy(model.state_dict()))
        self.assertTrue(all(torch.equal(v, states[1][k]) for k, v in states[0].items()))

    def test_integrity_and_speaker_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "data"
            synthetic_dataset(root)
            inspect_dataset(root)
            frame = pd.read_csv(root / "manifest.csv")
            frame.loc[frame.split.eq("validation"), "speaker"] = "train-0"
            frame.to_csv(root / "manifest.csv", index=False)
            completion = json.loads((root / "COMPLETED.json").read_text())
            completion["manifest_sha256"] = sha256(root / "manifest.csv")
            write_json(root / "COMPLETED.json", completion)
            with self.assertRaisesRegex(ValueError, "Fuga"):
                inspect_dataset(root)

    def test_grid_and_validation_only_selection(self):
        config = json.loads((AI_DIR / "configs" / "experiments.json").read_text())
        grid = experiment_grid(config)
        self.assertEqual(len(grid), 12)
        results = [{**c, "best_validation_macro_f1": i % 3 / 10, "best_validation_loss": 1.,
                    "test_macro_f1": 1 - i % 3 / 10} for i, c in enumerate(grid)]
        winners = select_winners(results)
        self.assertEqual(len(winners), 4)
        self.assertTrue(all(w["name"] == "sgd_1e-2" for w in winners.values()))
        config["trials"][1] = {**config["trials"][0], "name": "duplicate"}
        with self.assertRaisesRegex(ValueError, "diferentes"):
            experiment_grid(config)

    def test_full_offline_suite(self):
        """Doce trials de UNA época sintética + test de cuatro ganadores en W&B offline."""
        cache = AI_DIR / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            root = Path(temp)
            data = root / "data"
            synthetic_dataset(data)
            config = json.loads((AI_DIR / "configs" / "experiments.json").read_text())
            config.update(epochs=1, patience=1)
            for trial in config["trials"]:
                trial["batch_size"] = 4
            write_json(root / "config.json", config)
            env = {**os.environ, "WANDB_SILENT": "true", "WANDB_CONSOLE": "off"}
            command = [sys.executable, str(AI_DIR / "src" / "train.py"), "--data", str(data),
                       "--config", str(root / "config.json"), "--output", str(root / "runs"),
                       "--device", "cpu", "--threads", "2", "--wandb-mode", "offline", "--all"]
            trained = subprocess.run(command, capture_output=True, text=True, env=env, timeout=240)
            self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)
            suite = next((root / "runs").iterdir())
            frame = pd.read_csv(suite / "training_results.csv")
            self.assertEqual(len(frame), 12)
            self.assertEqual(len(json.loads((suite / "selection.json").read_text())["winners"]), 4)
            self.assertFalse((suite / "test_results.json").exists())
            self.assertEqual(len(list((suite / "wandb").glob("offline-run-*"))), 13)
            evaluated = subprocess.run([sys.executable, str(AI_DIR / "src" / "evaluate.py"), "--suite", str(suite)],
                                       capture_output=True, text=True, env=env, timeout=120)
            self.assertEqual(evaluated.returncode, 0, evaluated.stdout + evaluated.stderr)
            self.assertEqual(len(pd.read_csv(suite / "test_results.csv")), 4)
            self.assertTrue((suite / "report.md").exists())
            self.assertEqual(len(list((suite / "wandb").glob("offline-run-*"))), 18)

    def test_atomic_checkpoint_preserves_previous_on_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "last.pt"
            save_checkpoint(path, {"epoch": 1, "weights": torch.tensor([1., 2.])})
            before = path.read_bytes()
            def fail(stream):
                stream.write(b"incomplete")
                raise OSError("simulated write failure")
            with self.assertRaisesRegex(OSError, "simulated"):
                atomic_write(path, fail)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(torch.load(path, weights_only=True)["epoch"], 1)
            self.assertEqual(list(Path(temp).glob("*.tmp")), [])

    def test_single_experiments_and_exact_epoch_resume(self):
        cache = AI_DIR / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            root = Path(temp)
            synthetic_dataset(root / "data")
            config = json.loads((AI_DIR / "configs/experiments.json").read_text())
            config.update(epochs=2, patience=3)
            for trial in config["trials"]:
                trial["batch_size"] = 4
            write_json(root / "config.json", config)
            env = {**os.environ, "WANDB_SILENT": "true", "WANDB_CONSOLE": "off"}
            base = [sys.executable, str(AI_DIR / "src/train.py"), "--data", str(root / "data"),
                    "--config", str(root / "config.json"), "--device", "cpu", "--threads", "2", "--wandb-mode", "offline"]
            def execute(command):
                result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result
            for architecture in ("lenet5", "mobilenet_v1"):
                experiment = f"{architecture}-augmented-adam_1e-3"
                suite = root / f"resumed-{architecture}"
                target = suite / experiment
                execute(base + ["--suite", str(suite), "--experiment", experiment, "--epochs-per-run", "1"])
                state = torch.load(target / "last.pt", weights_only=True)
                self.assertEqual(state["epoch"], 1)
                self.assertFalse((target / "result.json").exists())
                self.assertFalse((suite / "selection.json").exists())
                # Reanudar infiere config, datos, entorno y cuenta de la suite.
                execute([sys.executable, str(AI_DIR / "src/train.py"), "--suite", str(suite), "--experiment", experiment])
                self.assertEqual(len(pd.read_csv(suite / "training_results.csv")), 1)
                resumed = torch.load(target / "last.pt", weights_only=True)
                self.assertEqual(resumed["epoch"], 2)
                self.assertEqual([row["epoch"] for row in resumed["history"]], [1, 2])
                reference = root / f"uninterrupted-{architecture}"
                execute(base + ["--suite", str(reference), "--experiment", experiment])
                expected = torch.load(reference / experiment / "last.pt", weights_only=True)
                for key, value in resumed["state_dict"].items():
                    self.assertTrue(torch.equal(value, expected["state_dict"][key]), f"{architecture}:{key}")
                for a, b in zip(resumed["history"], expected["history"]):
                    for key in a:
                        if key != "epoch_seconds":
                            self.assertEqual(a[key], b[key], f"{architecture}:{key}")
                checkpoint_hash = sha256(target / "best.pt")
                run_count = len(list((suite / "wandb").glob("offline-run-*")))
                execute([sys.executable, str(AI_DIR / "src/train.py"), "--suite", str(suite), "--experiment", experiment])
                self.assertEqual(len(list((suite / "wandb").glob("offline-run-*"))), run_count)
                self.assertEqual(sha256(target / "best.pt"), checkpoint_hash)
                # Sin --experiment/--all: exactamente un trial pendiente, manteniendo el previo.
                execute([sys.executable, str(AI_DIR / "src/train.py"), "--suite", str(suite)])
                self.assertEqual(len(pd.read_csv(suite / "training_results.csv")), 2)
                self.assertEqual(sha256(target / "best.pt"), checkpoint_hash)
                self.assertFalse((suite / "selection.json").exists())
                status = execute([sys.executable, str(AI_DIR / "src/train.py"), "--suite", str(suite), "--list"])
                self.assertIn("complete", status.stdout)
                self.assertIn("pending", status.stdout)


if __name__ == "__main__":
    unittest.main()
