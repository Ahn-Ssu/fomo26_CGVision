# Task 2 — Meningioma Binary Segmentation

**Final submission**: 10-fold CV ensemble of `FomoStudentSegNet` (see
`common/asparagus/` for the shared framework lineage this task's own fork
descends from — note below), each fold contributing its **own** best-epoch
checkpoint (not one epoch fixed across all folds — see "Per-fold-best vs
fixed-epoch ensembling"), encoder Convpass frozen / decoder unfrozen +
pretrained-init + fully fine-tuned (`decoder_scheme=finetune`), softmax-
averaged across all 10 folds, plus two inference-time additions on top:
selective per-channel gamma recalibration and largest-connected-component
(LCC) postprocessing (see `submission/predict.py`).

## Pipeline

1. **Preprocessing** (`submission/predict.py::preprocess`): union
   foreground mask across the 3 input channels (flair, dwi, swi-or-t2s) at
   native resolution → crop → resample to 1mm-isotropic (BSpline) → per-
   channel volume-wise z-normalization.
2. **Selective per-channel gamma recalibration** — applied to the
   normalized 3-channel tensor immediately before inference: flair ×1.5
   (darken), dwi_b1000 ×1.0 (untouched), swi_or_t2s ×1.75 (darken).
   Mechanistically: lesion tissue sits ~2.4–2.9× brighter than normal
   tissue in the normalized space across all 3 channels, and a gamma>1
   power-law amplifies that contrast ratio superlinearly, recovering
   recall the raw model misses. Per-channel magnitudes were found by an
   isolated sweep (`analysis/gamma_sweep_per_channel.py` — dwi_b1000 is
   the *one* channel actively **hurt** by darkening in either direction,
   its own local optimum is exactly gamma=1.0) and confirmed near-optimal
   under channel *interactions* by a greedy coordinate-ascent search
   (`analysis/greedy_gamma_search.py`). This is a single deterministic
   recalibration, **not** a multi-view TTA average (see the mirror-TTA
   ablation below).
3. **10-fold ensemble**: each fold's own best-epoch checkpoint runs
   sliding-window inference (patch=128³, overlap=0.5, Gaussian blend, no
   mirror TTA — see `FomoStudentSegNet.sliding_window_predict`), softmax
   probabilities averaged across all 10 folds.
4. **Largest-connected-component (LCC) postprocessing**: the averaged,
   argmaxed hard prediction is filtered to its single largest 26-connected
   foreground component before being un-resampled back onto the input's
   native grid (`asparagus.functional.reverse_preprocessing`).

Full quantitative before/after (DSC, NSD, HD95, each pipeline stage) is in
`analysis/comprehensive_before_after.py`'s own output — summary: gamma
alone improves DSC/NSD but *worsens* HD95 (introduces small spurious
false-positive blobs elsewhere in the volume); LCC alone mostly fixes
HD95; **combined, LCC removes gamma's spurious blobs while keeping
gamma's DSC/NSD gains** — the two are complementary fixes for different
failure modes, not redundant.

## Architecture / training recipe

`training/orchestrate_10foldcv_cutout_moddrop.py` is the actual 10-fold
sweep that produced the submitted checkpoints — see
`training/configs/SEG024_LOO13_SPATIALINTENSITY_LRMIRROR_CUTOUT_MODDROP_LR1E4.yaml`
for the full recipe: `FomoStudentSegNet` (teacher=`brats`,
`stem_mode=learnable`, `encoder_convpass_frozen=true`,
`decoder_scheme=finetune`), patch_size=128³, batch_size=2, 60 epochs (250
steps/epoch), spatial+intensity augmentation with L-R-only mirror
(`CPU_seg_train_transforms_mild_spatial_lrmirror`), plus two additional
regularizers layered on top of the intensity preset
(`GPU_seg_train_transforms_mild_intensity_cutout_moddrop`,
`training/transforms/`):
- **`Torch_Cutout3D`**: zeroes a random cuboid region (same region across
  all channels), p=0.15, size 10–30% of each spatial dim.
- **`Torch_ModalityDropout`**: zeroes the 3rd modality channel
  (swi_or_t2s) entirely, p=0.2 — targeted at that specific channel given
  known per-subject quality issues there.

### Per-fold-best vs fixed-epoch ensembling

Each fold contributes its *own* best-epoch checkpoint
(`training/extract_weights_bestckpt.py`) rather than one epoch fixed
across every fold. Rationale: the epoch that's best *on average* across
folds and the epoch that's best *for that specific fold's own held-out
validation subjects* encode genuinely different information, since each
fold never sees another fold's data. This was validated with real
official leaderboard scores (not just internal CV) — an earlier
fixed-epoch-across-all-folds submission scored NSD 0.171 / DSC 0.176,
while the per-fold-best ensemble of the *same* underlying recipe scored
NSD 0.195 / DSC 0.21.

## What we tried and rejected

- **Mirror TTA** (both LR-only 2-way and the standard full 8-way flip
  average, on top of the already-deployed gamma+LCC recipe): both hurt
  substantially — 8-way dropped mean DSC from 0.344 to 0.130, with 7
  additional subjects collapsing to a complete zero-dice miss (not just
  blurred accuracy — detections vanish entirely). The model never learned
  true left-right invariance despite L-R mirroring being part of training
  augmentation. `mirror=False` at inference throughout.
- **Gamma baked into training** (retraining on gamma-recalibrated tensors,
  both a single uniform gamma across all 3 channels and the same
  selective per-channel values used at inference): both underperformed
  the plain **test-time-only** gamma recalibration on the per-fold-best
  metric — the deterministic recalibration didn't need to be *learned*,
  applying it only at inference already sufficed, at zero retraining
  cost.
- **Multi-source-guided decoder upsampling** ("Upsampling Matters in
  U-Shaped Medical Image Segmentation," a 2026 paper): a full 3D
  reimplementation (`common/asparagus`'s `DecoderStage.upsample` replaced
  by a block fusing decoder semantics + skip-connected encoder detail +
  image-derived structural cues [avgpool low-freq, high-freq residual, 3D
  Sobel edge] by averaging, then residually refined) — underperformed the
  baseline substantially (per-fold-best 0.278 vs 0.300). Likely a
  data-regime mismatch: the paper validated on far larger 2D datasets
  with pretrained 2D encoders; here the new fusion/projection submodules
  had no pretrained counterpart and only ~20 training volumes/fold across
  60 epochs to cold-start from.
- **`decoder_scheme=scratch`** (decoder randomly reinitialized instead of
  pretrained-init, then fully fine-tuned) on this exact recipe: per-fold-
  best dropped from 0.300 to 0.223. This contradicts an *earlier*
  archsearch result on a different protocol/config (LOO13 splits, a
  spatial+mirror-only augmentation recipe) where scratch-init had
  appeared to beat pretrained-init by a wide margin — that finding did
  not replicate once actually tested on the real winning recipe/protocol.
  The gamma+LCC recipe (tuned for the pretrained-init model) also hurt
  rather than helped when applied on top of this scratch-decoder model,
  confirming those TTA parameters are specific to the model they were
  tuned against, not a model-agnostic fix.
- **Registration-based augmentation** (warping training volumes onto MNI
  space via rigid/affine registration as an augmentation): rigid-only
  exactly tied the no-registration baseline; affine-only and the combined
  rigid+affine variant were both worse, implicating the affine component
  specifically.

## Repository layout

```
submission/           The container's actual predict.py + the vendored
                       FOMO26/ and asparagus_pkg/ subtrees it imports
                       (architecture code, pre/post-processing utilities),
                       Apptainer.def, requirements.txt. Model weights
                       (submission/checkpoints/*.ckpt, ~1.3GB each fold +
                       pretrain_arch.pt) are NOT included -- see the
                       top-level README and submission/checkpoints/
                       manifest.json (per-fold best epoch / val Dice,
                       no weights).
training/              orchestrate_10foldcv_cutout_moddrop.py (the actual
                       10-fold sweep), its Hydra config, the two custom
                       augmentation modules (Torch_Cutout3D,
                       Torch_ModalityDropout), and
                       extract_weights_bestckpt.py (how the per-fold-best
                       checkpoints were selected and packaged).
analysis/              The gamma/LCC diagnostic scripts that directly
                       shaped the final inference recipe (per-channel
                       gamma sweep, greedy coordinate-ascent search under
                       channel interactions, comprehensive DSC/NSD/HD95
                       before-after comparison).
```

**Note on `common/asparagus/`**: this task was developed on a separate
machine/fork from Task1/3/4/5/6-7 (see the top-level README), on a version
of the shared `FomoStudentSegNet`/`StudentResEncUNet` framework with
additional `decoder_scheme`/`encoder_convpass_frozen`/`stem_mode` axes not
present in the `common/asparagus/` snapshot currently in this repository.
`submission/` is therefore fully self-contained (vendors its own copy of
every file it imports, matching `submission/`'s own runtime — it does not
depend on `common/asparagus/` at all), and `training/` includes the exact
config plus the two authored transform modules rather than assuming
compatibility with the committed `common/asparagus/` fork.
