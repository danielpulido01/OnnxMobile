"""Prueba integral sin red con audio sintético; no representa resultados del dataset."""
import contextlib
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest

import nbformat
import numpy as np
import soundfile as sf

AI_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(AI_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(AI_DIR / ".cache" / "numba"))
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt


class NotebookTest(unittest.TestCase):
    def test_pipeline_and_edge_cases(self):
        notebook = nbformat.read(AI_DIR / "notebooks" / "01_exploracion_speech_commands.ipynb", as_version=4)
        nbformat.validate(notebook)
        commands = ["yes", "no", "left", "right", "up", "down", "on", "off", "stop", "go"]
        source_words = commands + ["bed", "cat"]
        cache = AI_DIR / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            root = Path(temp)
            raw = root / "raw"
            split_speakers = {"train": [], "validation": [], "test": []}
            for number in range(200):
                speaker = f"speaker{number:03d}"
                score = int.from_bytes(hashlib.sha256(f"42:{speaker}".encode()).digest()[:8], "big") / 2**64
                split = "test" if score < .1 else "validation" if score < .2 else "train"
                if len(split_speakers[split]) < 2:
                    split_speakers[split].append(speaker)
            speakers = [s for group in split_speakers.values() for s in group]
            for label_index, label in enumerate(source_words):
                folder = raw / label
                folder.mkdir(parents=True)
                for speaker_index, speaker in enumerate(speakers):
                    sr = 8000 if speaker_index == 0 else 16000
                    duration = [.5, 1.0, 1.3][speaker_index % 3]
                    time = np.arange(int(sr * duration)) / sr
                    signal = .3 * np.sin(2 * np.pi * (200 + 23 * label_index + 7 * speaker_index) * time)
                    if speaker_index == 1:
                        signal = np.stack([signal, signal * .8], axis=1)
                    sf.write(folder / f"{speaker}_nohash_0.wav", signal, sr)
            original = raw / "yes" / f"{speakers[0]}_nohash_0.wav"
            (raw / "yes" / f"{speakers[0]}_nohash_1.wav").write_bytes(original.read_bytes())
            sf.write(raw / "yes" / "silent_nohash_0.wav", np.zeros(16000), 16000)
            (raw / "yes" / "broken_nohash_0.wav").write_bytes(b"invalid WAV")
            conflict = np.linspace(-.1, .1, 16000)
            for label in ["yes", "no"]:
                sf.write(raw / label / "conflict_nohash_0.wav", conflict, 16000)
            # Mismo nombre WAV en bed/cat: sus imágenes no deben sobrescribirse.
            # Contradicciones entre palabras unknown también se deben detectar.
            for label in ["bed", "cat"]:
                sf.write(raw / label / "otherconflict_nohash_0.wav", conflict * .5, 16000)
            noise = raw / "_background_noise_"
            noise.mkdir()
            sf.write(noise / "background.wav", np.ones(32000) * .01, 16000)
            namespace = {"__name__": "__main__"}
            old_dir = Path.cwd()
            keys = ["SPEECH_COMMANDS_DATA_DIR", "SPEECH_COMMANDS_OUTPUT_DIR"]
            previous_env = {k: os.environ.get(k) for k in keys}
            os.environ[keys[0]] = str(raw)
            os.environ[keys[1]] = str(root / "processed")
            try:
                os.chdir(AI_DIR)
                with contextlib.redirect_stdout(io.StringIO()):
                    for cell in notebook.cells:
                        if cell.cell_type == "code":
                            exec(compile(cell.source, f"notebook:{cell.id}", "exec"), namespace)
                            plt.close("all")
                manifest = namespace["manifest"]
                self.assertEqual(len(manifest), 72)
                self.assertTrue(manifest.image_path.is_unique)
                self.assertEqual(namespace["CLASSES"], commands + ["unknown"])
                self.assertEqual(namespace["LABEL_TO_ID"]["unknown"], 10)
                unknown = manifest.loc[manifest.label.eq("unknown")]
                self.assertEqual(set(unknown.source_label), {"bed", "cat"})
                self.assertEqual(len(unknown), 12)
                self.assertEqual(set(unknown.label_id), {10})
                self.assertEqual(unknown.groupby("split").size().to_dict(),
                                 {"train": 4, "validation": 4, "test": 4})
                self.assertFalse(manifest.relative_path.str.startswith("_background_noise_").any())
                for row in unknown.itertuples(index=False):
                    self.assertEqual(Path(row.image_path).name,
                                     f"{row.source_label}__{Path(row.relative_path).stem}.png")
                exclusions = namespace["audit"].exclusion_reason.value_counts()
                self.assertEqual(exclusions["audio_duplicado"], 1)
                self.assertEqual(exclusions["silencio_digital"], 1)
                self.assertEqual(exclusions["etiquetas_contradictorias"], 4)
                self.assertEqual(namespace["split_method"], "speaker_sha256_80_10_10")
                image_fn = namespace["audio_to_image"]
                self.assertTrue(np.array_equal(image_fn(np.zeros(16000, dtype=np.float32)), np.zeros((64, 101))))
                with self.assertRaises(ValueError):
                    image_fn(np.full(16000, np.nan))
                with self.assertRaises(ValueError):
                    image_fn(np.zeros(8000))
                # Segunda ruta: listas oficiales, seguida por detección de fuga de hablantes.
                for split, filename in [("validation", "validation_list.txt"), ("test", "testing_list.txt")]:
                    entries = [f"{label}/{speaker}_nohash_0.wav"
                               for label in source_words for speaker in split_speakers[split]]
                    (raw / filename).write_text("\n".join(entries), encoding="utf-8")
                split_cell = next(c for c in notebook.cells if c.cell_type == "code" and "def locate_list" in c.source)
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(split_cell.source, namespace)
                self.assertEqual(namespace["split_method"], "official_lists")
                validation = raw / "validation_list.txt"
                validation.write_text(validation.read_text() + f"\nbed/{split_speakers['train'][0]}_nohash_0.wav")
                with self.assertRaisesRegex(AssertionError, "Fuga de hablantes"):
                    exec(split_cell.source, namespace)
                # Liberar memory maps antes de borrar el directorio temporal en Windows.
                del namespace["X_train"]
            finally:
                namespace.clear()
                plt.close("all")
                os.chdir(old_dir)
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
