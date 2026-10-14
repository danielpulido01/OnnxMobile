# AI notebooks

- [Exploration, preprocessing and dataset sharing](notebooks/01_exploracion_speech_commands.ipynb)
- [Architectures, class balancing, training and evaluation](notebooks/02_entrenamiento_evaluacion.ipynb)

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
