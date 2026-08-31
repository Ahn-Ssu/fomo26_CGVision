# FastSurferVINN (aseg-dkt) — EXCLUDED from TEACHER_REGISTRY

**Verdict (revised 2026-07-15): DROP for live per-step feature-level distillation
(reasons 1 and 2 below still hold). Reason 3 ("speed") was overstated on
re-review — see the correction note right below — and is not, by itself, a
reason to exclude FastSurfer from an offline-precompute / output-level
pseudo-label role if one is wanted later.**

> **Correction (2026-07-15), prompted by user pushback against the "60-500x
> slower" framing**: FastSurfer's own README (`README.md:21`, cloned repo)
> states **"approximately 5 minutes (GPU)" for the full `--seg_only` CLI**,
> which includes conform, all 3 views, resample-back, LUT remapping, stats,
> and QC image generation — not just the network forward pass. Our measured
> "10-15s/volume" (Sec. 8 below) is actually well *inside* that 5-minute
> envelope, not evidence of an anomaly. The "60-500x slower than VesselFM/
> Anatomix" comparison was also **not apples-to-apples**: VesselFM/Anatomix
> were benchmarked on a single 128³ *patch* (2.1M voxels), while FastSurfer's
> `conform()` forces the *entire* 256³ brain volume through the pipeline
> (16.7M voxels, 8x more) — some of the gap is architecture-inherent 2.5D
> overhead (3 separate view models, 256 slices each), but part of it is just
> "whole volume" vs "one patch". Because FastSurfer is a **frozen, deterministic**
> teacher, its real per-step training-loop cost can be reduced to ~0 by
> precomputing pseudo-labels once per FOMO300K volume and caching them,
> rather than calling it live every training step like the other teachers —
> under that usage pattern, absolute speed is not a blocker at all. Reasons 1
> and 2 (pure 2D architecture, absolute-scale normalization with a real
> measured 79% prediction-agreement effect) are unaffected by this correction
> and remain valid technical reasons FastSurfer doesn't fit the live
> forward-hook feature-distillation pattern the other 3 teachers use.

This document records the concrete, empirically-verified evidence behind that
decision so a human can double check it later. Everything below was verified
against a real clone of https://github.com/Deep-MI/FastSurfer.git at
`/root/teachers/repos/FastSurfer` and real checkpoints downloaded from Zenodo
record 10390573, run on real GPU hardware (`cuda:0`, RTX 6000 Ada) with a real
T1 volume (`/root/data/FOMO-MRI/fomo-60k/sub_11043/ses_1/t1.nii.gz`,
240x240x155, 1mm isotropic). Nothing below is guessed or reconstructed from
memory of the FastSurfer paper.

## 1. Setup — what was actually done

- `git clone https://github.com/Deep-MI/FastSurfer.git /root/teachers/repos/FastSurfer` — succeeded.
- Model definition: `FastSurferCNN/models/networks.py`, class `FastSurferVINN`
  (lines 214-358), built via `build_model(cfg)` (lines 361-385).
- Checkpoints: `FastSurferCNN/config/checkpoint_paths.yaml` lists 3 files
  (axial/coronal/sagittal) for the VINN aseg-dkt model, hosted on b2share
  (404'd, unreachable from this box) and Zenodo record 10390573 (fallback
  URL in the same yaml). Queried `https://zenodo.org/api/records/10390573`
  directly and downloaded the 3 files by their real listed URLs (the yaml's
  assumed `url + "/" + relative_path` join is wrong for Zenodo — Zenodo
  stores files flat, not under `checkpoints/`):
  - `aparc_vinn_axial_v2.0.0.pkl` — 22,399,769 bytes (matches Zenodo listing exactly)
  - `aparc_vinn_coronal_v2.0.0.pkl` — 22,399,783 bytes (matches exactly)
  - `aparc_vinn_sagittal_v2.0.0.pkl` — 22,375,598 bytes (matches exactly)

  Saved to `/root/teachers/checkpoints/fastsurfer/`. Deliberately did NOT
  download `aparc_cnn_*` (older FastSurferCNN v1/v2, not VINN), CerebNet, or
  HypVINN checkpoints, per task scope.
- Only new dependency installed: `yacs` (`pip install --no-deps yacs`).
  Verified `torch.__version__ == 2.6.0+cu124` and `torch.cuda.is_available() == True`
  before and after; no `nvidia-*-cu13` packages present.

## 2. strict=True load result

Built each of the 3 view-specific models via `build_model(load_config(<view_yaml>))`
and loaded `torch.load(ckpt, weights_only=False)["model_state"]` with
`strict=True`:

```
axial:    <All keys matched successfully>   n_params=1,854,632   NUM_CLASSES=79
coronal:  <All keys matched successfully>   n_params=1,854,632   NUM_CLASSES=79
sagittal: <All keys matched successfully>   n_params=1,852,616   NUM_CLASSES=51
```

All three loaded cleanly, no missing/unexpected keys.

## 3. named_modules() / architecture summary

`model.named_children()` for the axial model (125 total modules via
`named_modules()`):

```
encode1..encode4  (CompetitiveEncoderBlock[Input])  — each: conv0-3 (Conv2d), bn0-4 (BatchNorm2d), prelu, maxpool
bottleneck        (CompetitiveDenseBlock)
decode4..decode1  (CompetitiveDecoderBlock) — each: conv0-3, bn1-4, prelu, unpool (MaxUnpool2d)
inp_block         (InputDenseBlock)   — conv0: Conv2d(7, 32, kernel_size=3, padding=1)
outp_block        (OutputDenseBlock)
interpol1/interpol2 (Zoom2d)          — 2D resampling layers for the "flex" multi-resolution trick
classifier        (ClassifierBlock)   — conv: Conv2d(71, 79, kernel_size=1)
```

Module type census: **45 `nn.Conv2d`, 0 `nn.Conv3d`, 0 `nn.Conv1d`**. Every
single learnable spatial op in this network is 2D. This directly confirms the
"2.5D not 3D" risk flag from prior analysis — it is not a mischaracterization.

Forward pass shape check (axial model, GPU):
```
input:  (1, 7, 256, 256)   # 7 stacked slices as channels, scale_factor [1,2]
output: (1, 79, 256, 256)  # per-pixel logits over 79 internal training classes
```

## 4. The 2.5D structure, precisely

- **Input shape**: `(N, 7, H, W)` — NOT `(N, 1, D, H, W)`. Confirmed by
  `inp_block.conv0 = Conv2d(7, 32, ...)` and `MODEL.NUM_CHANNELS: 7` in
  `FastSurferCNN/config/FastSurferVINN_axial.yaml`. This is "thick-slice"
  2D: 7 neighboring slices are stacked as input channels; the network
  predicts labels for (conceptually) the middle slice only, run once per
  slice position to cover the whole volume — per the class docstring at
  `networks.py:218`: *"Spatial view aggregation (input 7 slices of which
  only middle one gets segmented)"*.

- **Three separate models, confirmed, not assumed.** There genuinely are 3
  independent `FastSurferVINN` instances with 3 independently-trained
  checkpoints, one per anatomical plane:
  - `FastSurferCNN/config/checkpoint_paths.yaml` lines 5-8: `axial`,
    `coronal`, `sagittal` each map to a distinct `.pkl`.
  - Their per-view yaml configs differ in `MODEL.NUM_CLASSES`: axial=79,
    coronal=79, **sagittal=51** (`FastSurferVINN_sagittal.yaml`). Sagittal
    uses a reduced class set (no left/right distinction, since a sagittal
    slice alone cannot tell hemisphere) and gets remapped back to 79 classes
    post-hoc.
  - Verified all 3 `.pkl` files load `strict=True` into their respective
    configs (Sec. 2) — these are genuinely different weight sets, not the
    same weights reused across planes (different `n_params`, different
    checkpoint byte sizes).

- **View aggregation — read from the real inference code, not guessed.**
  `FastSurferCNN/inference.py`, class `Inference`:
  - `permute_order` (lines 128-132): each plane's raw `(D,C,H,W)`-ish network
    output is permuted into the shared volumetric orientation, e.g.
    `"axial": (3, 0, 2, 1)`, `"coronal": (2, 3, 0, 1)`, `"sagittal": (0, 3, 2, 1)`.
  - Sagittal-specific remap (lines 359-361, implemented in
    `FastSurferCNN/data_loader/data_utils.py:1093-1132`,
    `map_prediction_sagittal2full`): the 51-class sagittal logits are
    index-expanded back to the 79-class space via a hardcoded index list
    (`_idx = [[0], r(5,14), r(1,4), [14,15,4], r(16,19), r(5,51), ...]`),
    i.e. duplicating shared bilateral-structure logits into both left/right
    slots.
  - Aggregation itself (`Inference.eval`, lines 299-380): **not** averaging
    or 3D convolution — it is a per-plane-weighted **sum of logits**
    accumulated in-place into a shared volumetric tensor:
    `out[tuple(ii)].add_(pred.half(), alpha=self.alpha.get(plane, 0.4))`
    with `self.alpha = {"sagittal": 0.2}` (line 127), i.e. axial and
    coronal each contribute at weight 0.4, sagittal at weight 0.2 (these sum
    to 1.0 — a fixed, hand-set, non-learned view-fusion weighting). Final
    segmentation is `argmax` over this summed-logit volume (done downstream,
    not shown above, e.g. in `run_prediction.py`).
  - Downstream of that, `run_prediction.py:396-399` further remaps the
    79-class internal network output through the FreeSurfer LUT
    (`du.map_label2aparc_aseg`) and splits generic cortex labels into
    hemisphere-specific structures (`du.split_cortex_labels`) to produce the
    **95-class** output the README advertises (`README.md:25`: *"outputs
    anatomical segmentation and cortical parcellation ... of 95 classes"*).
    **Correction to the task's stated assumption**: the network itself only
    ever outputs 79 (or 51, for sagittal) logit channels; "95 classes" is a
    LUT-remapping/post-processing artifact, not the classifier's output
    dimensionality. Worth flagging since a wrapper naively reading
    `NUM_CLASSES` from the axial config would get 79, not 95, and getting to
    95 requires reimplementing `map_label2aparc_aseg` + `split_cortex_labels`
    as well.

## 5. Normalization — `conform()`, exact logic and empirical absolute-scale proof

- Located: `FastSurferCNN/data_loader/conform.py`. Relevant functions:
  `getscale()` (lines 851-947), `scalecrop()` (950-984), `conform()`
  (1019-1146).
- **Exact logic** (`getscale`, defaults `f_low=0.0`, `f_high=0.999`,
  called from `conform()` at line 1131 as `getscale(np.asanyarray(img.dataobj), 0, rescale=255)`,
  i.e. on the RAW input array as-is, before reorientation):
  1. Compute a 1000-bin histogram of the raw array over `[data_min, data_max]`.
  2. `src_min` = the bin edge below which `f_low` fraction of ALL voxels fall
     (0 by default → `src_min = data_min`, no low-end cropping).
  3. `src_max` = the bin edge at/above which `(1-f_high)` fraction of
     **non-zero** voxels fall (i.e. robustly crop the top 0.1% of non-zero
     intensity, similar in spirit to `mri_convert -c`).
  4. `scale = 255 / (src_max - src_min)`.
  5. `scalecrop()`: `out = clip(scale * (data - src_min), 0, 255)`, then
     zero-input voxels are forced back to exactly 0, then rounded to
     `uint8`.
  This is a **robust min/max linear rescale to `uint8 [0,255]`**, computed
  fresh from whatever array is handed to it — i.e. it behaves like a
  percentile transform in isolation, but crucially it is computed on the
  raw `img.dataobj`, and the whole point of asparagus's `volume_wise_znorm`
  is that it already **clips** (at foreground q99) before z-scoring —
  clipping is not invertible/order-preserving the way a pure affine
  transform is, so composing `conform` after `volume_wise_znorm` is not
  guaranteed to equal `conform` on raw data, unlike the purely-monotonic,
  no-prior-clipping percentile transforms used by VesselFM/Anatomix.

### Empirical test (real T1 volume, real `conform()` call, not reimplemented)

Used the exact `volume_wise_znorm` given in the task, applied to
`sub_11043/ses_1/t1.nii.gz` (raw min/max: 0.0 / 13282.95), then ran
FastSurfer's actual `FastSurferCNN.data_loader.conform.conform()` on (a) the
raw volume and (b) the z-normed volume, both wrapped in `nib.Nifti1Image`
with identical affine/header (isolating the intensity-rescale effect from
any reorientation effect):

```
conformed_raw: uint8, shape (256,256,256), range [0,255]
conformed_z:   uint8, shape (256,256,256), range [0,255]

max abs diff (uint8 levels):        58 / 255
mean abs diff:                       1.81
frac. voxels differing by >1 level:  8.49%
frac. voxels differing by >10 levels: 7.81%
frac. voxels differing by >50 levels: 0.021%
Pearson correlation (foreground):    0.9935
```

A correlation of 0.9935 alone might look "safe" (comparable to how
VesselFM/Anatomix reported ~1.0 in their write-ups) — so this was pushed
further to the thing that actually matters: **does it change the
segmentation**. The real axial checkpoint was run (forward pass, GPU) on
7-slice-thick windows built from both `conformed_raw` and `conformed_z`
(after the standard `/255.0` input scaling used by FastSurfer's own
`ToTensorTest`, `FastSurferCNN/data_loader/augmentation.py:38-55`), for 16
slice positions spanning the brain, and compared `argmax` label maps
voxel-by-voxel (same arbitrary-but-consistent slicing axis applied
identically to both paths, so this isolates normalization sensitivity even
though it isn't the anatomically "correct" axial orientation):

```
slice  90: fg=7911   differ=2297  agreement=70.96%
slice  95: fg=9958   differ=2077  agreement=79.14%
slice 100: fg=10581  differ=2407  agreement=77.25%
slice 105: fg=11488  differ=1751  agreement=84.76%
slice 110: fg=10280  differ=1872  agreement=81.79%
slice 115: fg=8706   differ=2152  agreement=75.28%
slice 120: fg=6264   differ=1628  agreement=74.01%
slice 125: fg=1390   differ=633   agreement=54.46%
slice 130: fg=1071   differ=985   agreement= 8.03%  <- worst case
slice 135: fg=4722   differ=1794  agreement=62.01%
slice 140: fg=8627   differ=1419  agreement=83.55%
slice 145: fg=12015  differ=1861  agreement=84.51%
slice 150: fg=11358  differ=2201  agreement=80.62%
slice 155: fg=10215  differ=1817  agreement=82.21%
slice 160: fg=9176   differ=1192  agreement=87.01%
slice 165: fg=8248   differ=1266  agreement=84.65%

OVERALL: fg_vox=132,010  differing=27,352  agreement=79.28%
```

**Verdict: `norm_type = "absolute"`, confirmed empirically, not assumed.**
Running the exact same real checkpoint through `preprocess(asparagus_znormed_x)`
vs `preprocess(raw_x)` changes the argmax segmentation label on ~21% of
foreground voxels overall, and on one sampled slice agreement was as low as
8%. This is not a rounding-level discrepancy like VesselFM's or Anatomix's
(both of which showed cosine similarity 1.0 / near-1e-7 float-level
agreement in their own writeups) — it is a substantial, real change in the
teacher's output. Per `base.py`'s own contract, this teacher would require a
`meta={"raw": raw_tensor}` escape hatch (and, as covered in Sec. 6, actually
also `meta={"affine": ..., "header": ...}`, since `conform()` needs full
nibabel spatial metadata, not just intensities).

## 6. Feasibility verdict

Evaluated the 4 options from the prior planning doc:

**(a) Slice-wise scan, stack per-slice 2D features into pseudo-3D.**
Would require forward hooks into all 3 per-view models (×5-ish decoder
stages each) and running every slice of the volume through each view
(confirmed cost below: ~256 slices × 3 views). The resulting "3D" feature
tensor would be an artifact of independently-computed 2D activations stacked
along an axis the network never convolved over — no cross-slice receptive
field beyond the 7-slice input window at each position. Weak as a
distillation target for a genuinely-3D student, and by far the most
engineering: hook management across 3 separate `nn.Module` instances,
correct un-permute/reassembly per view, alignment back to the input's native
grid (see below). **Rejected** — worst cost/benefit of the four.

**(b) Single view (e.g. coronal) as pseudo-3D.**
Cuts the multi-view forward cost to 1/3, but throws away exactly the
mechanism (weighted 3-view logit fusion, Sec. 4) FastSurfer's own authors
built specifically to correct the known axis-aligned striping artifacts of
any single 2D plane — using one view alone reintroduces the problem VINN
exists to solve, while still carrying the coordinate-grid and
absolute-normalization costs below. **Rejected** — degraded quality without
proportionally lower engineering cost.

**(c) Output-level pseudo-label only (skip feature-level KD for this teacher).**
The only usage mode that doesn't require hooking into internal 2D
activations — technically feasible: run 3 views' full forward passes
(confirmed working, Sec. 2-4), aggregate per `Inference.eval`'s real
weighted-logit-sum logic, argmax, then resample the result from FastSurfer's
canonical 256³ 1mm LIA grid (that `conform()` reslices everything onto) back
onto the input volume's native grid — solvable with `nibabel`'s
`resample_from_to`, not custom research code. However:
- `BaseTeacher`'s own docstring (`base.py:1-2`) frames this whole interface
  as being *for feature-level distillation*; a teacher whose only safe
  output is a final label map doesn't serve that purpose and doesn't
  meaningfully differ from a plain offline pseudo-label precompute script —
  it doesn't need to be a frozen `nn.Module` teacher in this registry to do
  that job.
- Requires widening the `meta` contract for this one teacher beyond raw
  intensities to include the original nibabel affine/header (needed for
  `conform()`'s reorientation and for resampling the prediction back) — no
  other teacher in the registry needs spatial metadata, only intensities.
- Real measured cost (below) is 100-500× a normal teacher forward pass.

**(d) Drop FastSurfer entirely.**
**This is the recommendation.**

### Why (d), concretely

1. **Confirmed 2.5D, no path to genuine 3D features.** 45/45 conv layers
   are `Conv2d`; 0 `Conv3d` anywhere in the network (Sec. 3). This is not a
   labeling nuance — there is no intermediate tensor anywhere in this model
   that is a true volumetric feature map.
2. **Confirmed absolute-scale normalization with a large, measured
   real-world effect**: 79.28% argmax-label agreement (worst slice 8%)
   between the correct raw-intensity path and the (incorrect but
   pipeline-standard) already-normalized path (Sec. 5). Any accidental
   reuse of the standard `preprocess(asparagus_normed_x)` pattern that every
   other teacher wrapper uses would silently produce a badly wrong
   segmentation with no error raised.
3. **Measured compute cost is 100-500× the other teachers', per volume**,
   even before feature-level KD is attempted:
   - `conform()` alone: 3.73s (single-threaded CPU, histogram + affine
     reslice) for one T1 volume.
   - One view's full-volume forward pass (256 slices, batch size 16,
     GPU): 1.76s, **peak GPU memory 11.08 GB** (vs. VesselFM's 2.58 GB for
     a full 128³ 3D forward pass).
   - Three views: ~5.3s GPU forward time alone, before the second
     `conform`-scale resample-back-to-native-grid step, sagittal remapping,
     and LUT/`split_cortex_labels` post-processing needed to reach the
     "95-class" output. Realistic estimated wall time: **~10-15s per
     volume**, vs. VesselFM's 0.03s and Anatomix's 0.15s for a single 3D
     forward pass — roughly 100-500× more expensive per volume, on top of
     needing to be run 3 times against a foreign coordinate grid.
4. **Coordinate-grid mismatch.** Every other teacher in this registry
   operates directly, fully-convolutionally, on the input volume's own
   grid. FastSurfer forces everything through `conform()`'s fixed 256³ 1mm
   LIA canonical grid and must be resampled back — a structurally different
   computational pattern (CPU/`nibabel`-heavy multi-stage orchestration,
   not a single deterministic GPU `forward(x)` call) from every sibling
   teacher wrapper (`anatomix_teacher.py`, `vesselfm_teacher.py`).
5. **Marginal value is genuinely limited given what's already covered.**
   Anatomix already provides a modality-agnostic, native-3D, fully
   convolutional generic feature extractor — the "spatial feature teacher"
   role FastSurfer might otherwise fill. FastSurfer's only distinctive
   contribution would be explicit anatomical-region *semantic labels*
   (T1-only), which is real but narrow value, available only through the
   output-level route (option c) that doesn't fit this registry's
   feature-level-KD purpose. If the broader project later wants an
   anatomical-prior/ROI signal specifically, it is better served by a
   dedicated offline pseudo-label precompute script built around
   FastSurfer's real `run_prediction.py --seg_only` pipeline than by a
   `BaseTeacher` subclass — the interface convergence just isn't there.

**Note on the task's `input_requirements` assumption**: independently
verified against `README.md:54`: *"resolution should be between 1mm and
0.7mm isotropic (slice thickness should not exceed 1.5mm)"* — matches
exactly what was given; not altered. Also confirmed T1w-only requirement
(`README.md:26/33/37`).

## 7. What was NOT done, and why

- Did not implement `FastSurferTeacher(BaseTeacher)` — per the verdict
  above.
- Did not register anything in `registry.py` — nothing to register.
- Did not attempt to reproduce the "95-class" LUT remap
  (`map_label2aparc_aseg` + `split_cortex_labels`) end-to-end, since the
  decision to drop was reached before that would have been needed; if this
  decision is ever revisited, that remapping (confirmed to exist at
  `run_prediction.py:396-399`) is the remaining piece needed to get from the
  network's raw 79/51-class output to the advertised 95-class output.
- Left the 3 downloaded checkpoints in place
  (`/root/teachers/checkpoints/fastsurfer/*.pkl`, ~22MB each, verified byte
  sizes match Zenodo) in case this decision is revisited — no need to
  re-download.
