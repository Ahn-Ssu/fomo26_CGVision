# FOMO26 Challenge — CGVision

Code for our FOMO26 challenge submissions: self-supervised pretraining on brain
MRI (`pretraining/`), a shared finetuning framework (`common/asparagus/`), and
per-task finetuning/submission code (`task1/`, `task3/`, `task5/`, `task6_7/`).

**Pretrained/finetuned model weights are intentionally excluded from this
repository** (checkpoints range from ~120MB to ~3GB each). Each task
directory's `submission/` folder contains the exact `predict.py`/
`preprocess.py`/model-architecture code used for our container submissions,
minus the bundled weight files (`model/fold_checkpoints/*.pt`,
`model/hdbet_weights/`, `model/checkpoint.pt`) and the MNI template NIfTI
(`model/template/`, obtainable via
[TemplateFlow](https://www.templateflow.org/), `MNI152NLin2009cAsym`).

## Structure

```
pretraining/        Self-supervised pretraining pipeline (multi-teacher
                     distillation onto a 3D ResEncUNet student, ver5 checkpoint)
common/asparagus/    Shared finetuning framework (Hydra configs, Lightning
                     modules, network architectures, transform presets) used
                     across all downstream tasks
task1/               Infarct classification
task3/               Brain age regression
task5/               Polymicrogyria (PMG) classification
task6_7/              Linear probing / fairness embeddings
```

Each task directory follows the same convention:
- `submission/` — the container's actual `predict.py`, `preprocess.py`,
  `model/*.py` architecture code, `Apptainer.def`, `requirements.txt`
- `training/` — representative Hydra-orchestration scripts showing how the
  submitted checkpoint(s) were produced (not every intermediate experiment —
  see each task's own README for the full method summary)
- `analysis/` (where applicable) — diagnostic scripts that shaped a
  methodological decision (e.g. Task5's ventricle-masking augmentation)

Note: Task2 and Task4 submissions were developed on a different machine in
our team and aren't included in this snapshot.

## Shared infrastructure

All finetuning tasks build on the same pretrained backbone
(`pretraining/`, self-supervised multi-teacher distillation, "ver5"
checkpoint) via `common/asparagus/`, a Hydra + PyTorch Lightning framework
that handles data loading, augmentation presets, and the network
architecture (`asparagus/modules/networks/task1_arch18.py`:
`FomoTask1Arch18Net` — modality-specific encoder + multistage feature
concatenation + GeM pooling classification head, the winning configuration
from our architecture sweep, reused for Task1 and Task5).

Submission containers themselves do **not** depend on `common/asparagus/`
(or on Lightning/Hydra/wandb) at runtime — each `submission/model/net.py` is
a standalone, dependency-free port of the exact architecture, so the actual
inference container stays minimal and has no path to accidentally reach the
network at runtime.
