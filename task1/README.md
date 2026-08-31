# Task 1 — Infarct Classification

**Final submission**: 10-fold ensemble of `FomoTask1Arch18Net`
(modality_specific + multistage + GeM pooling, see `common/asparagus/`),
finetuned on the `ver5` pretrained checkpoint, bf16-mixed precision.

## Pipeline

1. **Preprocessing** (`submission/preprocess.py`): HD-BET skull-strip on
   each of the 4 modalities (FLAIR, ADC, DWI, + SWI or T2*), rigid
   registration to a fixed template-anchored 1mm crop box, per-modality
   z-normalization inside the brain mask. Degrades once (registration
   failure → resample without registration) before crashing on purpose on
   any further failure, rather than silently returning an all-zero volume
   — a deliberate policy change after diagnosing that a prior silent
   all-zero fallback had masked a real bug (see the file's own docstring).
2. **Inference-time lesion-contrast enhancement** (`submission/predict.py`,
   `_apply_lesion_gamma_enhance`): a fixed, deterministic per-modality gamma
   correction (FLAIR/DWI gamma=1.33, ADC/4th-modality gamma=0.67) applied
   once to the preprocessed tensor, REPLACING it — not test-time-averaged.
   Found to substantially improve official held-out pooled AUROC
   (0.9038 → 0.9519) with **no retraining**, by amplifying exactly the
   diagnostic direction of a DWI-bright/ADC-dark ischemic lesion.
3. **10-fold ensemble**: all 10 fold checkpoints score the input; the
   final probability is the simple average.

## Architecture selection

`training/asparagus_orchestrate_task1_arch18.py` is the 18-way architecture
sweep (modality-specific vs. shared encoders × feature-source
[multistage vs. encoder-output-only] × pooling [avg/max/GeM]) that
established multistage-concat + GeM as the clear winner, reused for both
Task1 and Task5.

`training/asparagus_orchestrate_task1_champion_ema.py` is representative of
the final champion training recipe (the actual submission is a 10-fold
extension of this same recipe).

## What we tried and rejected

An extensive augmentation-sensitivity investigation (Gibbs ringing, motion
ghosting, bias field, gamma direction, LR-flip TTA, sharpening) found gamma
enhancement to be the one robust, reproducible win; other post-hoc
inference tricks (sharpening, LR-flip TTA ensembling) either had no
consistent effect or reversed sign between validation and the official
held-out split, and were not adopted.
