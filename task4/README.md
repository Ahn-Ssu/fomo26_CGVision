# Task 4 — Multiclass Tissue Segmentation

**Final submission**: best-of-fold 5-model ensemble of `FomoStudentSegNet`
(see `common/asparagus/`) -- for each of 5 CV folds, whichever pretrained
teacher (brats vs. vesselfm) scored the higher held-out val Dice was kept
(brats won folds 1/2/4, vesselfm won folds 0/3; see
`submission/checkpoints/manifest.json`), encoder fully frozen / decoder
trained from scratch, finetuned on the `ver5` pretrained checkpoint,
bf16-mixed precision, full 8-way mirror TTA at inference.

## Pipeline

1. **Preprocessing** (`submission/preprocess.py`): HD-BET skull-strip →
   ANTs rigid+affine registration to MNI152 → resample onto a fixed
   256³ @ 0.5mm-isotropic ROI grid (`submission/assets/roi_reference_iso05.nii.gz`)
   in a single interpolation hop from the untouched native input (the affine
   transform is computed once, on a 1mm-downsampled copy for speed — a
   measured ~53% speedup for HD-BET+registration with no measurable
   accuracy cost — then applied directly to the original-resolution image).
   A genuinely blank/degenerate input (mean and std both ≈0) short-circuits
   straight to an all-background output before HD-BET is even attempted —
   deliberately the *only* case that gets this treatment; every other
   failure (registration failing on every fallback candidate, an
   unexpected exception) is allowed to crash the process loudly rather
   than being silently converted into a wrong-but-valid-looking output.
2. **5-model ensemble + 8-way mirror TTA** (`submission/predict.py`): each
   fold's checkpoint runs sliding-window inference (patch=128³, overlap=0.5)
   averaged over all 8 flip combinations of the 3 spatial axes (matches
   the training pipeline's own `mirror_tta=True` default, +0.0114 mean
   Dice over no-TTA across all 15 (teacher, fold) checkpoints measured on
   held-out val subjects), then all 5 checkpoints' softmax probabilities
   are averaged (standard k-fold ensembling).
3. **Postprocessing** (`submission/postprocess.py`): each class's
   probability channel is inverse-warped back into the original scan's
   native grid **separately** (Linear interpolation via ANTs'
   `antsApplyTransforms`), and only then argmaxed in native space —
   argmaxing the ROI-space hard label first and nearest-neighbor
   resampling *that* was measured to lose ~0.5–0.7% Dice to double-NN
   quantization vs. this approach.

## Architecture / training recipe

`training/asparagus_orchestrate_task4.py` is the actual 5-fold × 2-teacher
(brats, vesselfm) sweep that produced the submitted checkpoints — see
`common/asparagus/configs/projects/task4/TASK4_ROI05_PRETRAIN_ENCFROZEN_DECSCRATCH.yaml`
for the full recipe: patch_size=128³, batch_size=2, 100 epochs (250
steps/epoch), warmup_epochs=10, encoder fully frozen (`encoder_convpass_frozen=true`),
decoder reinitialized and trained from scratch (`decoder_scheme=scratch`),
bf16-mixed precision, full-volume validation.

## What we tried and rejected

An extensive post-hoc robustness investigation (repeated stress tests
against every training-time augmentation, plus 4 MONAI-based
out-of-distribution intensity perturbations and a sharpening check not in
the training distribution at all) found three genuine vulnerabilities
relative to the trained augmentation distribution — rotation (−17.1%pt
Dice under a ±30° stress test vs. the ~6% combined per-axis exposure rate
actually seen in training), scale-shrink (−7.5%pt), and gamma-darkening
(−6.4%pt) — all sharing the same failure signature: a **false-negative
(missed-lesion) collapse**, not increased false positives, traced in part
to the registration step itself narrowing the model's effective training
distribution of head orientations to a much tighter band (residual
rotation across 40 registered training subjects: 2.8°–20.5°, mean 8.6°)
than the ±30° stress-test range.

- **Targeted augmentation strengthening** (raising rotation/scale/gamma
  exposure probability and widening the gamma-dark range, retrained
  end-to-end): measurably reduced the false-negative collapse on all 3
  targeted perturbations, but showed ~zero net average-Dice change overall
  and *increased* HD95 / false-negative collapse on several **non-targeted**
  perturbations (sharpening in particular, plus bias-field and
  scale-enlarge) — a real trade-off, not a strict improvement. **Not
  adopted.**
- **Gamma-brighten TTA** (2-way and expanded plane/gamma-only/flip-only/both
  matrix, gamma∈{0.5, 0.7}): a real, modest improvement alone, but **zero
  incremental benefit** once combined with the already-deployed 8-way
  mirror TTA. Not adopted (would have added inference cost for nothing).
- **Fixed gamma=0.5 brightening baked into training** (not just a TTA
  trick): no measurable benefit across a real 5-fold retrain (Δ=−0.0012
  average, only 2/5 folds improved). Not adopted.
- **8-way mirror TTA axis breakdown**: every *individual* one of the 8
  views (and every pairwise/triple combination) performs *worse* than the
  plain no-flip prediction alone — the TTA's benefit comes entirely from
  averaging across diverse, individually-weaker views (error cancellation),
  not from any single axis being an intrinsically better transform. This
  confirmed the full 8-way average (not a cheaper subset) is the right
  design.

## Container hardening

A real submission was marked **INVALID for "attempts to access the
internet"** despite no code in this pipeline ever calling `requests`
directly. Root-caused (via a `socket.connect`/`sitecustomize.py`
monkeypatch that also covers the `hd-bet` subprocess) to
`nnUNetPredictor.predict_from_files()` unconditionally creating a
`torch.multiprocessing.Manager()` for worker coordination — a purely local
`AF_UNIX` socket (`/tmp/pymp-*/listener-*`), but one FOMO26's detector
apparently doesn't distinguish from real external access. Fixed by patching
`HD_BET/hd_bet_prediction.py` to call `predict_from_files_sequential()`
instead (nnU-Net's own no-multiprocessing alternative — since HD-BET only
ever predicts one file here, this measured slightly *faster*, not slower).

Two further hardening passes, both verified end-to-end and both reflected
in `submission/Apptainer.def`:
- **No `pip install` of the full `asparagus` package.** `model/net.py`
  (`FomoStudentSegNet`) is vendored standalone — it only needs torch +
  `gardening_tools` + `model/student.py`. Installing all of `asparagus`
  pulled in `lightning`, `hydra-core`, `omegaconf`, and `wandb` as hard
  dependencies (none ever imported by this container's actual code path)
  and silently downgraded the base image's own CUDA-matched torch build
  via asparagus's pinned `torch==2.6.0`, which in turn produced a real
  `monai 1.6.0 requires torch>=2.8.0` pip conflict. `Apptainer.def` now
  asserts at build time that `wandb`/`mlflow`/`lightning`/`hydra`/`omegaconf`
  are *not* importable.
- **Slim (`state_dict`-only) checkpoints.** The original Lightning
  `.ckpt` files embed `hyper_parameters` containing `omegaconf.ListConfig`
  objects — `torch.load()` unpickles the *entire* file regardless of which
  keys are actually read, so removing `omegaconf` (previous point) broke
  checkpoint loading outright (`ModuleNotFoundError: omegaconf`) until each
  checkpoint was re-saved containing only `{"state_dict": {...}}` (also a
  ~52% size reduction, 269MB → 129MB each). Verified by loading with
  `omegaconf` genuinely uninstalled from the environment.

## Repository layout

```
submission/           The container's actual predict.py/preprocess.py/
                       postprocess.py/model/*.py/Apptainer.def/requirements.txt
                       (checkpoints, HD-BET weights, and the MNI template
                       are NOT included -- see the top-level README)
training/              asparagus_orchestrate_task4.py -- the representative
                       5-fold x 2-teacher sweep that produced the submitted checkpoints
```
