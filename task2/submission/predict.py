#!/usr/bin/env python3
"""FOMO26 Task 2 (Meningioma binary segmentation) submission predict.py.

GAMMA TTA + LCC revision (2026-08-28), built on top of the CUTOUT_MODDROP
best.ckpt ensemble that was the highest-scoring official submission so
far. Two inference-time additions, both validated this session on
CUTOUT_MODDROP's own fold-best.ckpt models against their genuinely
held-out subjects (comprehensive_before_after.py):

  1. SELECTIVE PER-CHANNEL GAMMA (applied to the preprocessed/normalized
     input tensor, right before the model): flair *=1.50, dwi *=1.0
     (untouched), swi_or_t2s *=1.75. These values were found by isolating
     each channel's own gamma sensitivity (gamma_sweep_per_channel.py),
     then confirmed near-optimal under channel INTERACTIONS by a greedy
     coordinate-ascent search (greedy_gamma_search.py) -- dwi_b1000 is the
     one channel that is actively hurt by darkening in either direction
     (its own local optimum is gamma=1.0), while flair/swi_or_t2s benefit
     from darkening (their lesion:normal-tissue contrast ratio is
     amplified superlinearly by gamma>1). This is a single deterministic
     recalibration of channel intensity, NOT a multi-view TTA average.
  2. LARGEST-CONNECTED-COMPONENT (LCC) postprocessing on the final hard
     prediction. An EARLIER same-day ablation (2026-08-20, on different
     configs: AUGEF patch128/epoch14, AUGEF patch128/epoch39, AUGEF
     patch160/epoch14) found CC filtering net-negative and shipped
     WITHOUT it. Re-tested here specifically on CUTOUT_MODDROP + gamma TTA
     and found net-POSITIVE on all three tracked metrics (DSC, NSD, HD95)
     -- gamma TTA's contrast amplification was found to introduce spurious
     small false-positive blobs elsewhere in the volume (13/23 held-out
     subjects had their gamma-TTA prediction changed by LCC), which LCC
     removes; LCC also modestly helps the CC structure of the un-gamma'd
     baseline. HD95 in particular dropped ~16mm with LCC (51.3->34.8mm),
     since a distant spurious blob dominates the 95th-percentile surface
     distance far more than it dominates DSC/NSD.

mirror TTA is still OFF (sliding_window_predict(..., mirror=False)) -- an
earlier ablation found 8-way flip TTA hurts unseen Dice substantially, and
every 10-fold-CV number used to pick a submission epoch was itself
computed with mirror=False, so turning TTA on here would deploy something
never validated.

Pipeline (mirrors preprocess_seg.py / SegmentationModule.test_step exactly,
minus the label):
  1. load flair/dwi/(swi|t2s) at native resolution, fixed channel order
     [flair, dwi, swi_or_t2s] (see preprocess_seg.py module docstring for
     why this order, not the spec's original [dwi,flair,...] claim).
  2. Issue-A-safe foreground mask per channel AT NATIVE RESOLUTION, union
     bbox crop (native coords) BEFORE resampling.
  3. Resample the (smaller) native crop to 1mm iso: BSpline for images.
  4. Re-clean background (Issue-A fix b) + per-channel volume-wise znorm.
  5. Pad to a multiple that's safe for sliding-window (monai handles this
     internally via sliding_window_inference's padding, so no explicit pad
     step is needed here -- unlike training's on-the-fly Torch_Pad, which
     only existed to fix inputs to a training patch size).
  5.5. Apply selective per-channel gamma (flair=1.5, dwi=1.0, swi_or_t2s=
     1.75) to the normalized input tensor -- see GAMMA_CHANNEL_GAMMAS below.
  6. Run sliding_window_inference (PATCH_SIZE^3, overlap 0.5, gaussian
     blend, no mirror TTA -- see note above) with EACH fold checkpoint,
     softmax each, average the probability maps across folds.
  7. reverse_preprocessing (SimpleITK BSpline un-resample -> uncrop onto the
     original native canvas) on the averaged probability map.
  8. argmax -> largest-connected-component filter (see LCC note above) ->
     save NIfTI with the reference image's affine/header (so it lines up
     with the original input, not the 1mm-iso working grid).

Environment (set by the container / Apptainer.def, see container/README.md):
  MODEL_DIR: directory containing model_fold0.ckpt..model_fold9.ckpt
             (weights-only state_dicts, see container/extract_weights.py)
             and pretrain_arch.pt (the ver5 checkpoint, needed only to infer
             the architecture -- see FomoStudentSegNet.__init__).
  PATCH_SIZE: sliding-window patch size (int, cubic), must match how the
              fold checkpoints were trained. Defaults to 128.
"""
import argparse
import os
import sys
import time

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy import ndimage

# IMPORT_ROOT default (/app) matches the Apptainer image layout produced by
# stage_build.py (predict.py itself lives at /app/predict.py, so /app is
# already sys.path[0] automatically -- these three extra entries satisfy
# `from FOMO26.data.X import ...` (needs IMPORT_ROOT), `from networks.student
# import ...` inside fomo26_student.py (needs IMPORT_ROOT/FOMO26 directly),
# and `from asparagus.functional... import ...` (needs IMPORT_ROOT/
# asparagus_pkg). Overridable so this same script can be smoke-tested
# directly against container/build/ before the real apptainer build (see
# container/README.md).
IMPORT_ROOT = os.environ.get("PREDICT_IMPORT_ROOT", "/app")
sys.path.insert(0, os.path.join(IMPORT_ROOT, "FOMO26"))
sys.path.insert(0, IMPORT_ROOT)
sys.path.insert(0, os.path.join(IMPORT_ROOT, "asparagus_pkg"))

import glob

from FOMO26.data.normalize import asparagus_volume_wise_znorm  # noqa: E402
from FOMO26.data.preprocess_raw import _sitk_resample  # noqa: E402
from asparagus.functional.reverse_preprocessing import reverse_preprocessing  # noqa: E402
from asparagus.modules.networks.fomo26_student import FomoStudentSegNet  # noqa: E402

CHANNEL_ORDER = ["flair", "dwi", "swi_or_t2s"]
# index into CHANNEL_ORDER -> gamma (see module docstring); dwi=1.0 means
# "leave this channel untouched" (its own local optimum).
GAMMA_CHANNEL_GAMMAS = {0: 1.50, 1: 1.0, 2: 1.75}
EPS = 1e-6
TARGET_SPACING = (1.0, 1.0, 1.0)
PATCH_SIZE = (int(os.environ.get("PATCH_SIZE", 128)),) * 3
MIRROR_TTA = False  # see module docstring -- matches how this checkpoint's epoch was selected
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
MODEL_CKPTS = sorted(glob.glob(os.path.join(MODEL_DIR, "model_fold*.ckpt")))
PRETRAIN_ARCH_CKPT = os.path.join(MODEL_DIR, "pretrain_arch.pt")
TEACHER_NAME = "brats"
STEM_MODE = "learnable"
FREEZE_DECODER = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BENCHMARK = os.environ.get("PREDICT_BENCHMARK", "0") == "1"


def log_time(label, t0):
    if BENCHMARK:
        print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s", flush=True)
    return time.perf_counter()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--flair", type=str, required=True)
    p.add_argument("--dwi", type=str, required=True)
    p.add_argument("--t2s", type=str, default=None)
    p.add_argument("--swi", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def load_xyz(path):
    img = nib.load(path)
    data = np.asarray(img.dataobj).astype(np.float32)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data, spacing, img


def preprocess(paths):
    """paths: dict channel_name -> filepath, for CHANNEL_ORDER. Returns
    (channels: list[np.ndarray] normalized 1mm-iso, properties: dict,
    reference_img: nib.Nifti1Image at native resolution/affine)."""
    channel_data, channel_spacing, channel_nib = {}, {}, {}
    for ch in CHANNEL_ORDER:
        data, spacing, nib_img = load_xyz(paths[ch])
        channel_data[ch] = data
        channel_spacing[ch] = spacing
        channel_nib[ch] = nib_img

    shapes = {ch: channel_data[ch].shape for ch in CHANNEL_ORDER}
    spacings = {ch: channel_spacing[ch] for ch in CHANNEL_ORDER}
    assert len(set(shapes.values())) == 1, f"channel shapes disagree: {shapes}"
    assert len(set(spacings.values())) == 1, f"channel spacings disagree: {spacings}"

    spacing = channel_spacing[CHANNEL_ORDER[0]]
    original_size = channel_data[CHANNEL_ORDER[0]].shape
    reference_img = channel_nib[CHANNEL_ORDER[0]]  # flair -- used for output affine/header

    native_masks = {ch: (channel_data[ch] > EPS) for ch in CHANNEL_ORDER}
    union_mask = np.zeros(original_size, dtype=bool)
    for ch in CHANNEL_ORDER:
        union_mask |= native_masks[ch]
    assert union_mask.any(), "no foreground in union mask at native resolution"

    coords = np.argwhere(union_mask)
    mins = coords.min(axis=0)
    maxs_incl = coords.max(axis=0)
    crop_box = [int(x) for x in (mins[0], maxs_incl[0], mins[1], maxs_incl[1], mins[2], maxs_incl[2])]

    def crop_native(arr):
        return arr[mins[0]:maxs_incl[0] + 1, mins[1]:maxs_incl[1] + 1, mins[2]:maxs_incl[2] + 1]

    cropped_channels_native = {ch: crop_native(channel_data[ch]) for ch in CHANNEL_ORDER}
    cropped_masks_native = {ch: crop_native(native_masks[ch]) for ch in CHANNEL_ORDER}
    size_before_resample = cropped_channels_native[CHANNEL_ORDER[0]].shape

    resampled_channels, resampled_masks = {}, {}
    for ch in CHANNEL_ORDER:
        img_r = _sitk_resample(cropped_channels_native[ch], spacing, target_spacing=TARGET_SPACING,
                                interpolator=sitk.sitkBSpline, default_value=0.0)
        mask_r = _sitk_resample(cropped_masks_native[ch].astype(np.float32), spacing, target_spacing=TARGET_SPACING,
                                 interpolator=sitk.sitkNearestNeighbor, default_value=0.0)
        mask_bin = mask_r > 0.5
        img_r = np.where(mask_bin, img_r, 0.0).astype(np.float32)
        resampled_channels[ch] = img_r
        resampled_masks[ch] = mask_bin

    resampled_shapes = {ch: resampled_channels[ch].shape for ch in CHANNEL_ORDER}
    assert len(set(resampled_shapes.values())) == 1, f"post-resample channel shapes disagree: {resampled_shapes}"

    normed_channels = []
    for ch in CHANNEL_ORDER:
        mask = resampled_masks[ch]
        assert mask.any(), f"channel {ch} has no foreground after resample"
        normed = asparagus_volume_wise_znorm(resampled_channels[ch], mask=mask)
        normed_channels.append(normed.astype(np.float32))

    new_size = normed_channels[0].shape
    properties = {
        "original_size": list(original_size),
        "size_before_resample": list(size_before_resample),
        "crop_box": crop_box,
        "pad_box": [],
        "shape_before_pad": list(new_size),
        "new_size": list(new_size),
        "original_spacing": list(spacing),
        "new_spacing": list(TARGET_SPACING),
    }
    return normed_channels, properties, reference_img


def gamma_transform_channels(x, channel_gammas):
    """x: (1,C,D,H,W) tensor, already preprocessed/normalized. Per-channel
    min/max normalize -> power law -> denormalize back to the channel's own
    range, so this works regardless of the input's absolute scale. Matches
    gamma_selective_tta_vs_retrain.py / greedy_gamma_search.py exactly."""
    out = x.clone()
    for c, gamma in channel_gammas.items():
        if gamma == 1.0:
            continue
        ch = out[:, c]
        lo, hi = ch.min(), ch.max()
        rng = (hi - lo).clamp_min(1e-7)
        norm = (ch - lo) / rng
        out[:, c] = norm.pow(gamma) * rng + lo
    return out


def largest_cc(pred_hard):
    """Keep only the largest connected component (26-connectivity) of the
    foreground mask; empty mask passes through unchanged. Matches
    comprehensive_before_after.py exactly."""
    if pred_hard.sum() == 0:
        return pred_hard
    structure = np.ones((3, 3, 3), dtype=int)
    labeled, n = ndimage.label(pred_hard, structure=structure)
    if n <= 1:
        return pred_hard
    sizes = ndimage.sum(pred_hard, labeled, range(1, n + 1))
    keep = np.argmax(sizes) + 1
    return (labeled == keep).astype(pred_hard.dtype)


def build_model(ckpt_path):
    model = FomoStudentSegNet(
        input_channels=3, output_channels=2,
        checkpoint_path=PRETRAIN_ARCH_CKPT, teacher_name=TEACHER_NAME,
        from_scratch=False, stem_mode=STEM_MODE, freeze_decoder=FREEZE_DECODER,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # FomoStudentSegNet inherits gardening_tools.BaseNet.load_state_dict, which
    # silently filters (never raises on) missing/shape-mismatched keys and
    # returns None -- not safe to trust blindly (see container/README.md
    # benchmark notes on why this was checked explicitly rather than assumed).
    own_keys = model.state_dict()
    missing = [k for k in own_keys if k not in state_dict]
    shape_mismatch = [k for k in state_dict if k in own_keys and state_dict[k].shape != own_keys[k].shape]
    unexpected = [k for k in state_dict if k not in own_keys]
    assert not missing, f"{ckpt_path}: {len(missing)} keys missing from checkpoint, e.g. {missing[:5]}"
    assert not shape_mismatch, f"{ckpt_path}: shape mismatch, e.g. {shape_mismatch[:5]}"
    assert not unexpected, f"{ckpt_path}: {len(unexpected)} unexpected keys in checkpoint, e.g. {unexpected[:5]}"
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(DEVICE)
    return model


def main():
    args = parse_args()
    t_total0 = time.perf_counter()
    t0 = t_total0

    swi_or_t2s_path = args.swi if args.swi else args.t2s
    assert swi_or_t2s_path is not None, "one of --swi/--t2s is required"
    paths = {"flair": args.flair, "dwi": args.dwi, "swi_or_t2s": swi_or_t2s_path}

    channels, properties, reference_img = preprocess(paths)
    t0 = log_time("preprocess", t0)

    x = torch.from_numpy(np.stack(channels, axis=0)).unsqueeze(0).to(DEVICE)  # (1,3,D,H,W)
    t0 = log_time("to_gpu", t0)

    x = gamma_transform_channels(x, GAMMA_CHANNEL_GAMMAS)
    t0 = log_time("gamma_tta", t0)

    assert MODEL_CKPTS, f"no model_fold*.ckpt found in {MODEL_DIR}"
    probs_sum = None
    for ckpt_path in MODEL_CKPTS:
        model = build_model(ckpt_path)
        t0 = log_time(f"model_load[{os.path.basename(ckpt_path)}]", t0)
        with torch.no_grad(), torch.autocast(device_type="cuda" if DEVICE == "cuda" else "cpu", dtype=torch.bfloat16):
            logits = model.sliding_window_predict(data=x, patch_size=PATCH_SIZE, overlap=0.5, mirror=MIRROR_TTA)
            probs = F.softmax(logits.float(), dim=1)
        t0 = log_time(f"infer[{os.path.basename(ckpt_path)}]", t0)
        probs_sum = probs if probs_sum is None else probs_sum + probs
        del model
        torch.cuda.empty_cache()
    probs_avg = probs_sum / len(MODEL_CKPTS)
    t0 = log_time(f"ensemble_avg[{len(MODEL_CKPTS)} folds]", t0)

    src_probs = reverse_preprocessing(probs_avg, properties)
    t0 = log_time("reverse_preprocessing", t0)

    pred_hard = src_probs.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    t0 = log_time("argmax", t0)

    pred_hard = largest_cc(pred_hard)
    t0 = log_time("largest_cc", t0)

    out_img = nib.Nifti1Image(pred_hard, reference_img.affine, reference_img.header)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    nib.save(out_img, args.output)
    log_time("save", t0)
    log_time("TOTAL", t_total0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
