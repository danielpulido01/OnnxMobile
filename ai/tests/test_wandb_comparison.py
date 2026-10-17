"""Pruebas sin red: procedencia, selección y notebook con W&B simulado."""
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import nbformat
import pandas as pd

AI = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(AI / ".cache/matplotlib"))
sys.path.insert(0, str(AI / "tools"))
from wandb_comparison import (collect_runs, download_media, match_tests, rank_runs,
                              read_history, training_record)


class FakeRun:
    def __init__(self, run_id="one", entity="team", project="speech", experiment=None,
                 architecture="lenet5", variant="base", state="finished", dataset="dataset-a"):
        self.id = run_id
        self.path = [entity, project, run_id]
        self.name = experiment or f"{architecture}-{variant}-adam_b32_e15"
        self.url = "https://wandb.ai/" + "/".join(self.path)
        self.group = "laptop-1" if run_id == "one" else "laptop-2"
        self.job_type = "train"
        self.state = state
        self.config = {"experiment": self.name, "name": "adam_b32_e15", "architecture": architecture,
                       "variant": variant, "seed": 42, "epochs": 3, "optimizer": "adam",
                       "lr": .001, "batch_size": 32, "dataset_fingerprint": dataset,
                       "training_dataset_fingerprint": dataset + "-" + variant,
                       "loss": "cross_entropy", "class_weights": [1., 1.],
                       "environment": {"source_sha256": "code", "gpu": "fake-gpu"}}
        self.rows = [{"epoch": 1, "validation/macro_f1": .7, "validation/loss": .5,
                      "validation/accuracy": .72, "train_clean/macro_f1": .75, "train_clean/loss": .4},
                     {"epoch": 2, "validation/macro_f1": .8, "validation/loss": .3,
                      "validation/accuracy": .81, "train_clean/macro_f1": .9, "train_clean/loss": .2},
                     {"epoch": 3, "validation/macro_f1": .75, "validation/loss": .6,
                      "validation/accuracy": .76, "train_clean/macro_f1": .99, "train_clean/loss": .1},
                     {"epoch": 3, "validation_best/confusion": {"_type": "image-file"}}]
        self.summary = {"best_epoch": 2, "epochs_ran": 3, "best_validation_macro_f1": .8,
                        "best_validation_loss": .3, "validation/macro_f1": .75,
                        "validation/accuracy": .76, "checkpoint_sha256": "hash-" + run_id,
                        "macro_f1_gap": .1, "parameters": 20, "conv_linear_macs": 50}
        self.scanned = False

    def scan_history(self, **kwargs):
        assert "keys" not in kwargs, "Intersection filtering would drop sparse history"
        self.scanned = True
        return iter(self.rows)

    def file(self, name):
        def download(root, replace):
            target = Path(root) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith(".png"):
                target.write_bytes(base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWZkAAAAASUVORK5CYII="))
            else:
                target.write_text(json.dumps({"columns": ["class", "precision", "recall", "f1", "support"],
                                              "data": [["yes", .8, .8, .8, 10]]}), encoding="utf-8")
            return target.open()
        return Mock(download=download)


def fake_api(runs):
    def query(path, filters, per_page):
        groups = filters.get("group", {}).get("$in")
        return [r for r in runs if "/".join(r.path[:2]) == path and (not groups or r.group in groups)]
    return Mock(runs=Mock(side_effect=query))


class ComparisonTest(unittest.TestCase):
    def test_sparse_history_and_best_checkpoint_not_last_epoch(self):
        run = FakeRun()
        history = read_history(run)
        self.assertEqual(history.epoch.tolist(), [1, 2, 3])
        row = training_record(run, history)
        self.assertTrue(row["eligible"])
        self.assertEqual(row["best_validation_accuracy"], .81)
        self.assertAlmostEqual(row["last_macro_f1_gap"], .24)
        self.assertGreater(row["tail_validation_loss_change"], 0)
        self.assertLess(row["tail_train_clean_loss_change"], 0)

    def test_two_projects_same_name_and_overlapping_sources(self):
        runs = [FakeRun(), FakeRun(entity="other", project="second")]
        sources = [{"entity": "team", "project": "speech"},
                   {"entity": "other", "project": "second"},
                   {"entity": "team", "project": "speech", "groups": ["laptop-1"]}]
        frame, _, histories, _, _ = collect_runs(fake_api(runs), sources)
        self.assertEqual(len(frame), 2)
        self.assertEqual(len(histories), 2)
        ranking, winners = rank_runs(frame)
        self.assertEqual(len(ranking), 2)
        self.assertEqual(len(winners), 1)

    def test_incomplete_and_nonfinite_runs_are_excluded(self):
        runs = [FakeRun(), FakeRun("running", state="running"), FakeRun("partial"),
                FakeRun("nan"), FakeRun("missing-provenance")]
        runs[2].summary["epochs_ran"] = 2
        runs[3].summary["best_validation_macro_f1"] = float("nan")
        # No fallback F1 at best_epoch either.
        runs[3].rows = []
        del runs[4].config["class_weights"]
        frame, *_ = collect_runs(fake_api(runs), [{"entity": "team", "project": "speech"}])
        ranking, _ = rank_runs(frame)
        self.assertEqual(ranking.run_id.tolist(), ["one"])
        self.assertTrue(frame.loc[~frame.eligible, "exclusion_reason"].str.len().gt(0).all())

    def test_cohorts_require_explicit_choice_and_weights_must_match(self):
        runs = [FakeRun(), FakeRun("different-data", dataset="dataset-b"), FakeRun("different-loss")]
        runs[2].config["class_weights"] = [1., 2.]
        frame, *_ = collect_runs(fake_api(runs), [{"entity": "team", "project": "speech"}])
        self.assertEqual(frame.cohort.nunique(), 3)
        with self.assertRaisesRegex(ValueError, "COHORT"):
            rank_runs(frame)
        ranking, _ = rank_runs(frame, cohort=frame.iloc[0].cohort)
        self.assertEqual(ranking.run_id.tolist(), ["one"])

    def test_test_metrics_never_select_winner_and_require_checkpoint(self):
        runs = [FakeRun(), FakeRun("two", architecture="mobilenet_v1")]
        runs[1].summary["best_validation_macro_f1"] = .9
        evaluation = FakeRun("test-one")
        evaluation.job_type = "test"
        evaluation.config.update(training_run_id="one", checkpoint_sha256="hash-one")
        evaluation.summary = {"test/macro_f1": .99, "test/accuracy": .99}
        runs.append(evaluation)
        frame, tests, *_ = collect_runs(fake_api(runs), [{"entity": "team", "project": "speech"}])
        ranking, _ = rank_runs(frame)
        result, _ = match_tests(ranking, tests)
        self.assertEqual(ranking.iloc[0].run_id, "two")
        self.assertEqual(result.iloc[0]["test/macro_f1"], .99)
        evaluation.config["checkpoint_sha256"] = "wrong"
        result, warnings = match_tests(ranking, tests)
        self.assertTrue(result.empty)
        self.assertEqual(len(warnings), 1)

    def test_missing_history_does_not_invent_metrics(self):
        run = FakeRun()
        run.scan_history = Mock(side_effect=RuntimeError("offline"))
        frame, _, _, _, warnings = collect_runs(fake_api([run]), [{"entity": "team", "project": "speech"}])
        self.assertEqual(len(warnings), 1)
        self.assertTrue(frame.iloc[0].eligible)
        self.assertIsNone(frame.iloc[0].best_validation_accuracy)

    def test_media_download_is_selective_and_confined(self):
        run = FakeRun()
        run.summary["validation_best/confusion"] = {"path": "media/images/validation_best/confusion.png"}
        with tempfile.TemporaryDirectory() as temp:
            path = download_media(run, "validation_best/confusion", temp)
            self.assertTrue(path.is_file())
            self.assertTrue(path.is_relative_to(Path(temp)))
            run.summary["validation_best/confusion"] = {"path": "../model.pt"}
            with self.assertRaises(ValueError):
                download_media(run, "validation_best/confusion", temp)

    def test_full_notebook_offline_with_publication(self):
        notebook = nbformat.read(AI / "notebooks/03_comparacion_modelos_wandb.ipynb", as_version=4)
        nbformat.validate(notebook)
        config = json.loads((AI / "configs/experiments.json").read_text(encoding="utf-8"))
        runs = []
        for architecture in ("lenet5", "mobilenet_v1"):
            for variant in ("base", "augmented"):
                for trial in config["trials"]:
                    run = FakeRun(str(len(runs)), experiment=f"{architecture}-{variant}-{trial['name']}",
                                  architecture=architecture, variant=variant)
                    run.config["name"] = trial["name"]
                    run.summary["validation_best/confusion"] = {"path": "media/images/confusion.png"}
                    run.summary["validation_best/per_class"] = {"path": "media/table/per_class.table.json"}
                    runs.append(run)
        run = FakeRun("test")
        run.job_type = "test"
        run.config.update(training_run_id="0", checkpoint_sha256="hash-0")
        run.summary = {"test/macro_f1": .7, "test/accuracy": .8}
        runs.append(run)
        import matplotlib.pyplot as plt
        comparison = Mock(id="comparison", url="https://wandb.ai/team/speech/runs/comparison")
        manager = Mock()
        manager.__enter__ = Mock(return_value=comparison)
        manager.__exit__ = Mock(return_value=False)
        with (tempfile.TemporaryDirectory() as temp,
              patch.dict(os.environ, {"WANDB_ENTITY": "team", "WANDB_PROJECT": "speech"}),
              patch("wandb.login") as login, patch("wandb.Api", return_value=fake_api(runs)),
              patch("wandb.init", return_value=manager) as init,
              patch("wandb.Table"), patch("wandb.Image"), patch("wandb.Artifact"),
              patch("IPython.display.display"), patch("matplotlib.pyplot.show", side_effect=lambda: plt.close("all")),
              contextlib.redirect_stdout(io.StringIO())):
            namespace = {"__name__": "__main__"}
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                source = cell.source.replace("OUTPUT = AI / 'runs' / 'wandb_comparison'", f"OUTPUT = Path({temp!r})")
                source = source.replace("PUBLISH_COMPARISON = False", "PUBLISH_COMPARISON = True")
                exec(compile(source, f"comparison_cell_{index}", "exec"), namespace)
            login.assert_called_once()
            init.assert_called_once()
            self.assertFalse(namespace["provisional"])
            self.assertEqual(len(namespace["ranking"]), 12)
            self.assertEqual(len(namespace["winners"]), 4)
            self.assertEqual(len(namespace["test_results"]), 1)
            self.assertEqual(len(namespace["class_tables"]), 4)
            self.assertTrue((Path(temp) / "report.md").is_file())
            selection = json.loads((Path(temp) / "selection.json").read_text(encoding="utf-8"))
            self.assertEqual(len(selection["winner_run_paths"]), 4)
            self.assertEqual(len(list(Path(temp).glob("curves_*.png"))), 12)
            comparison.log_artifact.assert_called_once()


if __name__ == "__main__":
    unittest.main()
