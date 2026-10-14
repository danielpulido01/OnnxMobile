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

from data import inspect_dataset, sha256, spec_augment, SpectrogramDataset
from augment import inspect_augmented_dataset, prepare_augmented_dataset, validate_policy
from experiment import seed_everything, write_json, atomic_write, save_checkpoint, capture_rng, restore_rng
from metrics import metrics_from_confusion, run_epoch
from models import build_model, DepthwiseSeparable, LeNet5, PartialC3
from train import experiment_grid, select_winners
from sampling import make_training_loader


def synthetic_dataset(root, extra_unknown=0):
    root.mkdir()
    rng = np.random.default_rng(123)
    classes = ["yes", "no", "unknown"]
    rows, counts = [], {}
    for split, per_class in (("train", 4), ("validation", 2), ("test", 2)):
        labels = np.tile(np.arange(3, dtype=np.int64), per_class)
        if split == "train":
            labels = np.concatenate([labels, np.full(extra_unknown, 2, dtype=np.int64)])
        features = rng.random((len(labels), 64, 101, 1), dtype=np.float32)
        np.save(root / f"X_{split}.npy", features)
        np.save(root / f"y_{split}.npy", labels)
        counts[split] = len(labels)
        for i, label in enumerate(labels):
            rows.append({"split": split, "tensor_index": i, "label_id": int(label),
                         "label": classes[label], "speaker": f"{split}-{i}",
                         "source_label": ("bed" if i % 2 else "cat") if label == 2 else classes[label],
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

    def test_controlled_sampling_quotas_diversity_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "data"
            synthetic_dataset(root, extra_unknown=36)
            dataset = SpectrogramDataset(root, "train")
            classes = ["yes", "no", "unknown"]
            config = {"sampling": {"strategy": "controlled", "unknown_fraction": .2},
                      "class_weighting": "none", "batch_size": 4, "seed": 42}
            loader = make_training_loader(dataset, classes, config)
            first = list(loader.sampler)
            np.testing.assert_array_equal(np.bincount(dataset.y[first]), [4, 4, 2])
            self.assertEqual(len(set(first)), 10)
            self.assertTrue(set(range(12)) - {2, 5, 8, 11} <= set(first))
            self.assertEqual(loader.sampler.summary()["unknown_word_quotas"], {"bed": 1, "cat": 1})
            fresh = make_training_loader(dataset, classes, config)
            self.assertEqual(first, list(fresh.sampler))
            augmentation = torch.Generator().manual_seed(1042)
            saved = capture_rng(augmentation, [loader])
            second = list(loader.sampler)
            self.assertNotEqual(first, second)
            self.assertNotEqual(set(first) & set(np.flatnonzero(dataset.y == 2)),
                                set(second) & set(np.flatnonzero(dataset.y == 2)))
            restore_rng(saved, augmentation, [fresh])
            self.assertEqual(second, list(fresh.sampler))
            before = sha256(root / "y_validation.npy")
            with self.assertRaisesRegex(ValueError, "solo.*train"):
                make_training_loader(SpectrogramDataset(root, "validation"), classes, config)
            self.assertEqual(before, sha256(root / "y_validation.npy"))
            legacy = make_training_loader(dataset, classes, {**config, "sampling": {"strategy": "none"}})
            self.assertEqual(len(list(legacy.sampler)), len(dataset))

    def test_controlled_sampling_repeats_only_after_command_pool_exhaustion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "data"
            synthetic_dataset(root, extra_unknown=36)
            labels = np.load(root / "y_train.npy")
            labels[0] = 1  # yes=3, no=5: completar cada comando hasta cinco.
            np.save(root / "y_train.npy", labels)
            manifest = pd.read_csv(root / "manifest.csv")
            manifest.loc[manifest.split.eq("train") & manifest.tensor_index.eq(0), "label_id"] = 1
            manifest.to_csv(root / "manifest.csv", index=False)
            dataset = SpectrogramDataset(root, "train")
            config = {"sampling": {"strategy": "controlled", "unknown_fraction": .2},
                      "class_weighting": "none", "batch_size": 4, "seed": 42}
            loader = make_training_loader(dataset, ["yes", "no", "unknown"], config)
            indices = np.array(list(loader.sampler))
            np.testing.assert_array_equal(np.bincount(labels[indices]), [5, 5, 2])
            yes_indices = indices[labels[indices] == 0]
            self.assertEqual(set(yes_indices), set(np.flatnonzero(labels == 0)))
            self.assertLessEqual(np.unique(yes_indices, return_counts=True)[1].max(), 2)

    def test_invalid_sampling_and_double_correction_rejected(self):
        original = json.loads((AI_DIR / "configs/experiments.json").read_text())
        for fraction in (0, 1, -.2, True, float("nan"), float("inf")):
            with self.subTest(fraction=fraction):
                config = copy.deepcopy(original)
                config["trials"][0]["sampling"] = {"strategy": "controlled", "unknown_fraction": fraction}
                with self.assertRaisesRegex(ValueError, "unknown_fraction"):
                    experiment_grid(config)
        original["trials"][0]["class_weighting"] = "inverse_frequency"
        with self.assertRaisesRegex(ValueError, "doble corrección"):
            experiment_grid(original)

    def test_notebook_sampling_preview_and_error_diagnostics(self):
        import nbformat
        import matplotlib.pyplot as plt
        notebook = nbformat.read(AI_DIR / "notebooks/02_entrenamiento_evaluacion.ipynb", as_version=4)
        nbformat.validate(notebook)
        markdown = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
        self.assertIn("Gradient-Based Learning Applied to Document Recognition", markdown)
        self.assertIn("Instalar y conectar W&B", markdown)
        self.assertIn("Checkpoints, pausas y recuperación", markdown)
        self.assertIn("muestreo controlado", markdown)
        self.assertNotIn("architectures.md", markdown)
        self.assertNotIn("training.md", markdown)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, f"notebook:{cell.id}", "exec")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            synthetic_dataset(root / "data", extra_unknown=36)
            config = json.loads((AI_DIR / "configs/experiments.json").read_text())
            grid = experiment_grid(config)
            namespace = {"json": json, "pd": pd, "display": lambda *args: None,
                         "DATA_DIR": root / "data", "AUGMENTED_DIR": root / "data",
                         "SUITE": root / "suite", "grid": grid}
            cells = {cell.id: cell for cell in notebook.cells}
            try:
                exec(cells["controlled-sampling-preview"].source, namespace)
                self.assertEqual(namespace["sampling_preview"].base_por_epoca.tolist(), [4, 4, 2])
                target = root / "suite" / grid[0]["experiment"]
                target.mkdir(parents=True)
                np.savetxt(target / "validation_best_confusion.csv",
                           [[2, 0, 1], [0, 3, 0], [1, 1, 2]], delimiter=",", fmt="%d")
                exec(cells["unknown-errors-diagnostics"].source, namespace)
                self.assertEqual(len(namespace["error_rows"]), 1)
                result = namespace["error_rows"][0]
                self.assertEqual(result["split"], "validation")
                self.assertEqual(result["unknown_to_command"], .5)
                self.assertEqual(result["command_to_unknown"], 1 / 6)
            finally:
                plt.close("all")

    def test_mask_axes_and_disabled_masking(self):
        original = torch.ones(2, 1, 64, 101)
        for axis, dimension in (("time", 2), ("frequency", 1)):
            policy = {"time_masks": 0, "time_width": 0, "frequency_masks": 0, "frequency_width": 0}
            policy.update({f"{axis}_masks": 1, f"{axis}_width": 8})
            masked = spec_augment(original, policy, torch.Generator().manual_seed(42))
            self.assertTrue(masked.eq(0).any())
            # Toda la franja cubre el eje opuesto: columnas para tiempo, filas para frecuencia.
            opposite = 1 if dimension == 2 else 2
            self.assertTrue(torch.equal(masked[0, 0].eq(0).any(dim=opposite - 1),
                                        masked[0, 0].eq(0).all(dim=opposite - 1)))
        policy = {"time_masks": 1, "time_width": 0, "frequency_masks": 1, "frequency_width": 0}
        self.assertTrue(torch.equal(original, spec_augment(original, policy, torch.Generator())))

    def test_saved_augmented_dataset_lineage_and_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            synthetic_dataset(root / "base")
            source = inspect_dataset(root / "base")
            policy = json.loads((AI_DIR / "configs/experiments.json").read_text())["augmentation"]
            first = prepare_augmented_dataset(root / "augmented", source, policy, 1042, batch_size=5)
            second = prepare_augmented_dataset(root / "repeat", source, policy, 1042, batch_size=7)
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            self.assertEqual(inspect_dataset(root / "base")["fingerprint"], source["fingerprint"])
            self.assertEqual(first["counts"], {**source["counts"], "train": 36})
            original = np.load(root / "base/X_train.npy")
            expanded = np.load(root / "augmented/X_train.npy")
            np.testing.assert_array_equal(expanded[:12], original)
            self.assertTrue(np.all(expanded[12:24] <= original))
            self.assertTrue(np.all(expanded[24:] <= original))
            self.assertTrue(np.all(np.any(expanded[12:24] != original, axis=(1, 2, 3))))
            self.assertTrue(np.isfinite(expanded).all())
            self.assertGreaterEqual(expanded.min(), 0)
            self.assertLessEqual(expanded.max(), 1)
            self.assertEqual(expanded.dtype, np.float32)
            labels = np.load(root / "augmented/y_train.npy")
            np.testing.assert_array_equal(labels, np.tile(np.load(root / "base/y_train.npy"), 3))
            manifest = pd.read_csv(root / "augmented/manifest.csv")
            self.assertEqual(set(manifest.split), {"train"})
            self.assertTrue(manifest.groupby("source_tensor_index").size().eq(3).all())
            for column in ("speaker", "audio_sha256", "file_sha256", "label_id"):
                self.assertTrue(manifest.groupby("source_tensor_index")[column].nunique().eq(1).all())
            self.assertFalse((root / "augmented/X_test.npy").exists())
            self.assertFalse((root / "augmented/X_validation.npy").exists())
            reused = prepare_augmented_dataset(root / "augmented", source, policy, 1042)
            self.assertEqual(reused["fingerprint"], first["fingerprint"])
            with self.assertRaisesRegex(ValueError, "incompatible"):
                inspect_augmented_dataset(root / "augmented", source, policy, 42)
            # Una exportación incompleta nunca se reutiliza ni se sobrescribe.
            (root / "incomplete").mkdir()
            with self.assertRaises(FileNotFoundError):
                prepare_augmented_dataset(root / "incomplete", source, policy, 1042)
            expanded[12, 0, 0, 0] = .123
            np.save(root / "augmented/X_train.npy", expanded)
            with self.assertRaisesRegex(ValueError, "modificado"):
                inspect_augmented_dataset(root / "augmented", source, policy, 1042)

    def test_augmentation_requires_an_active_technique(self):
        policy = {"time_masks": 0, "time_width": 10, "frequency_masks": 0, "frequency_width": 8,
                  "copies_per_sample": 2}
        with self.assertRaisesRegex(ValueError, "activa"):
            validate_policy(policy)
        policy["time_masks"] = 1
        validate_policy(policy)
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_policy({**policy, "copies_per_sample": value})

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
        self.assertTrue(all(w["name"] == config["trials"][-1]["name"] for w in winners.values()))
        config["trials"][1] = {**config["trials"][0], "name": "duplicate"}
        with self.assertRaisesRegex(ValueError, "diferentes"):
            experiment_grid(config)

    def test_trial_budgets_and_matched_dataset_settings(self):
        config = json.loads((AI_DIR / "configs/experiments.json").read_text())
        grid = experiment_grid(config)
        self.assertEqual({c["batch_size"] for c in grid}, {32, 64, 128})
        self.assertEqual({c["epochs"] for c in grid}, {15, 30, 45})
        for architecture in ("lenet5", "mobilenet_v1"):
            for trial in config["trials"]:
                pair = [c for c in grid if c["architecture"] == architecture and c["name"] == trial["name"]]
                self.assertEqual({c["variant"] for c in pair}, {"base", "augmented"})
                self.assertEqual({c["epochs"] for c in pair}, {trial["epochs"]})
                self.assertEqual({c["patience"] for c in pair}, {trial["patience"]})
                self.assertEqual({k: v for k, v in pair[0].items() if k not in ("variant", "experiment")},
                                 {k: v for k, v in pair[1].items() if k not in ("variant", "experiment")})
        # Los trials sin overrides siguen heredando el presupuesto global.
        del config["trials"][0]["epochs"], config["trials"][0]["patience"]
        inherited = experiment_grid(config)[0]
        self.assertEqual(inherited["epochs"], config["epochs"])
        self.assertEqual(inherited["patience"], config["patience"])

    def test_invalid_trial_overrides_rejected(self):
        original = json.loads((AI_DIR / "configs/experiments.json").read_text())
        cases = (("epochs", 0), ("epochs", 1.5), ("patience", -1), ("patience", True),
                 ("batch_size", 1), ("batch_size", 32.5), ("lr", 0), ("lr", float("nan")),
                 ("weight_decay", -1), ("min_delta", float("inf")), ("momentum", 1.1))
        for key, value in cases:
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(original)
                config["trials"][-1][key] = value
                with self.assertRaisesRegex(ValueError, key):
                    experiment_grid(config)

    def test_full_offline_suite(self):
        """Doce trials de 1–2 épocas sintéticas + test de cuatro ganadores en W&B offline."""
        cache = AI_DIR / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            root = Path(temp)
            data = root / "data"
            synthetic_dataset(data)
            config = json.loads((AI_DIR / "configs" / "experiments.json").read_text())
            config.update(epochs=1, patience=1)
            for index, trial in enumerate(config["trials"]):
                trial.update(batch_size=4, epochs=1 + index % 2, patience=3)
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
            self.assertEqual(set(frame.epochs), {1, 2})
            self.assertTrue(frame.epochs_ran.eq(frame.epochs).all())
            self.assertTrue(frame.patience.eq(3).all())
            self.assertTrue(frame.loc[frame.variant.eq("base"), "training_samples"].eq(12).all())
            self.assertTrue(frame.loc[frame.variant.eq("augmented"), "training_samples"].eq(36).all())
            self.assertTrue(frame.original_training_samples.eq(12).all())
            self.assertTrue(frame.loc[frame.variant.eq("base"), "samples_per_epoch"].eq(10).all())
            self.assertTrue(frame.loc[frame.variant.eq("augmented"), "samples_per_epoch"].eq(30).all())
            self.assertTrue(frame.sampling_strategy.eq("controlled").all())
            augmented_runs = frame.loc[frame.variant.eq("augmented")]
            self.assertTrue(augmented_runs.optimizer_steps.eq(8 * augmented_runs.epochs_ran).all())
            for result in frame.to_dict("records"):
                sampling = json.loads((suite / result["experiment"] / "sampling.json").read_text())
                self.assertEqual(sampling["unknown_fraction_actual"], .2)
                checkpoint = torch.load(suite / result["experiment"] / "best.pt", weights_only=True)
                self.assertEqual(checkpoint["class_weights"], [1., 1., 1.])
            saved = json.loads((suite / "augmented_dataset.json").read_text())
            self.assertEqual(saved["counts"]["train"], 36)
            self.assertEqual(len(json.loads((suite / "selection.json").read_text())["winners"]), 4)
            self.assertFalse((suite / "test_results.json").exists())
            self.assertEqual(len(list((suite / "wandb").glob("offline-run-*"))), 13)
            evaluated = subprocess.run([sys.executable, str(AI_DIR / "src" / "evaluate.py"), "--suite", str(suite)],
                                       capture_output=True, text=True, env=env, timeout=120)
            self.assertEqual(evaluated.returncode, 0, evaluated.stdout + evaluated.stderr)
            self.assertEqual(len(pd.read_csv(suite / "test_results.csv")), 4)
            import nbformat
            report = nbformat.read(suite / "report.ipynb", as_version=4)
            nbformat.validate(report)
            self.assertTrue(all(cell.cell_type == "markdown" for cell in report.cells))
            self.assertIn("Cuatro ganadores en test", report.cells[0].source)
            self.assertFalse((suite / "report.md").exists())
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
                trial.update(batch_size=4, epochs=2, patience=3)
            write_json(root / "config.json", config)
            env = {**os.environ, "WANDB_SILENT": "true", "WANDB_CONSOLE": "off"}
            base = [sys.executable, str(AI_DIR / "src/train.py"), "--data", str(root / "data"),
                    "--config", str(root / "config.json"), "--device", "cpu", "--threads", "2", "--wandb-mode", "offline"]
            def execute(command):
                result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result
            for architecture in ("lenet5", "mobilenet_v1"):
                experiment = f"{architecture}-augmented-{config['trials'][0]['name']}"
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
