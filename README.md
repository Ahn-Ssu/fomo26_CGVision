# FOMO26 Challenge — CGVision

Code for our FOMO26 challenge submissions: self-supervised pretraining on brain
MRI (`pretraining/`), a shared finetuning framework (`common/asparagus/`), and
per-task finetuning/submission code (`task1/`, `task2/`, `task3/`, `task4/`,
`task5/`, `task6_7/`).

**Pretrained/finetuned model weights are intentionally excluded from this
repository** (checkpoints range from ~120MB to ~3GB each). Each task
directory's `submission/` folder contains the exact `predict.py`/
`preprocess.py`/model-architecture code used for our container submissions,
minus the bundled weight files (`model/fold_checkpoints/*.pt`,
`model/hdbet_weights/`, `model/checkpoint.pt`, or for Task4,
`checkpoints/*.ckpt` + `checkpoints/pretrain_step_280000.pt` +
`hd-bet_params/`, or for Task2, `checkpoints/model_fold*.ckpt` +
`checkpoints/pretrain_arch.pt`) and the MNI template NIfTI (`model/template/`, obtainable
via [TemplateFlow](https://www.templateflow.org/), `MNI152NLin2009cAsym`).
The one exception is Task4's `submission/assets/roi_reference_iso05.nii.gz`
(73KB) — a small custom-built ROI reference grid (not a model weight, and
not obtainable from any external template source), included so the exact
preprocessing grid is reproducible.

## Structure

```
pretraining/        Self-supervised pretraining pipeline (multi-teacher
                     distillation onto a 3D ResEncUNet student, ver5 checkpoint)
common/asparagus/    Shared finetuning framework (Hydra configs, Lightning
                     modules, network architectures, transform presets) used
                     across all downstream tasks
task1/               Infarct classification
task2/               Meningioma binary segmentation
task3/               Brain age regression
task4/               Multiclass tissue segmentation
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

Note: Task2 and Task4 were both developed on separate machines/forks in our
team from Task1/3/5/6-7. `common/asparagus/asparagus/scripts/FOMO26/
Task2_predict.py` and `.../Task4_predict.py` are both present in the shared
framework snapshot but are unfilled boilerplate (`MODEL_DIR = None`) —
**not** our actual solutions for those tasks. The real submission code for
both lives in `task2/submission/` and `task4/submission/`, per the same
convention as the other tasks; see each task's own README for why their
`submission/` folders are fully self-contained rather than depending on the
committed `common/asparagus/` snapshot.

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
(or on Lightning/Hydra/wandb) at runtime — each task's `submission/` folder
vendors a standalone, dependency-free copy of the exact architecture code it
imports (`submission/model/net.py` for Task1/3/4/5/6-7; Task2 instead vendors
its own `submission/FOMO26/` + `submission/asparagus_pkg/` subtrees, matching
its container's actual runtime layout — see `task2/README.md`), so the actual
inference container stays minimal and has no path to accidentally reach the
training framework at runtime.
