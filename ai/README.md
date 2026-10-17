# AI notebooks

- [Exploration, preprocessing and dataset sharing](notebooks/01_exploracion_speech_commands.ipynb)
- [Architectures, class balancing, training and evaluation](notebooks/02_entrenamiento_evaluacion.ipynb)
- [Compare both laptops directly from W&B](notebooks/03_comparacion_modelos_wandb.ipynb)

The comparison notebook reads completed training runs from one or more W&B
projects/groups without local datasets, checkpoints, or a GPU. Set `SOURCES`
with the W&B entity/project used by each laptop (one entry suffices for a shared
project), then run the notebook. Your W&B account must have read access to every
source. Authenticate with `wandb.login()` or `WANDB_API_KEY`; never save API keys
in notebook cells.

It ranks comparable runs by best-checkpoint validation macro-F1, then validation
loss, and shows accuracy, learning curves, generalization gaps, confusion
matrices, and per-class metrics. Different dataset fingerprints or loss/weight
definitions require an explicit cohort selection. Missing/incomplete runs are
listed, and an incomplete experiment grid produces a provisional conclusion.
Previously logged test evaluations are matched by training run, dataset, and
checkpoint hash; test never determines the ranking. Laptop latency measurements
are shown as context and are not used to rank models.

For offline runs, synchronize from **each original laptop** with
`wandb sync <path-to-offline-run>` for each `offline-run-*` folder inside
`ai/runs/<suite>/wandb`, then refresh the notebook. The notebook includes a
PowerShell command to iterate over these folders.
Only synchronized metrics/media can be recovered. New test evaluations still
require the weights and dataset on the machine performing that evaluation.
The notebook exports its evidence to `ai/runs/wandb_comparison`; optionally set
`PUBLISH_COMPARISON=True` to create a new W&B comparison run with the report.
The training runs remain unchanged.

Run the comparison checks without W&B credentials or network access:
`python -m unittest discover -s ai/tests -p test_wandb_comparison.py`.

The [latest processed dataset release (2026-10-14)](https://github.com/danielpulido01/OnnxMobile/releases/tag/speech-commands-20261014-v1)
includes both the base export and the saved augmented training dataset. Download
the attached README and extraction script for checksum-verified setup on another
laptop without repeating preprocessing or augmentation. Assign separate experiment
names to each laptop; automatic next-pending selection checks local progress only.

Each training invocation completes one trial for all its configured epochs
(15, 30 or 45), without early stopping, in a single W&B run. The best validation
weights stay in memory and are saved as `best.pt` only after training finishes.
Interrupted trials restart from
epoch one; completed trials are skipped. Use the notebook's new
`speech_commands_gpu_fixed_epochs_v5` suite for this workflow; existing suites
and their checkpoints are preserved.
