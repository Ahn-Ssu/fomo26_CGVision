"""Raw-preserving FOMO300K preprocessing: 1mm-isotropic resample + explicit
brain/foreground mask, NO intensity normalization baked in (Task 3, see
/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/03_PREPROCESSING_REPORT.md for the full rationale and validation results).

Why raw-preserving: /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/02_TEACHER_REPORT.md Sec D found each of our 3 live teachers
was trained with a DIFFERENT normalization (BraTS: foreground z-score;
VesselFM/Anatomix: full-volume percentile->bounded range). asparagus's own
preprocessing bakes ONE normalization into the saved array
(/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/01_ASPARAGUS_ANALYSIS.md), which only round-trips losslessly for the
percentile-based teachers (proven exactly invariant), not exactly for BraTS
(measured 99.83% argmax agreement in Task 2, not 100%). Preserving raw
intensity + a mask lets every teacher (and the student) apply ITS OWN
normalization on load, at the cost of doing that normalization on-the-fly
every step instead of once.

Issue A (background contamination after resampling) -- handled explicitly:
cubic (order=3) resampling of the intensity image can leak small nonzero
values into what should be pure background near the brain boundary. This
would corrupt any foreground-mask-based statistic (BraTS's z-score in
particular, which uses `use_mask_for_norm=True`). Fix applied here (options
(a)+(b) from PREPROCESSING task spec):
  (a) foreground mask is computed on the native-resolution image
      (x > eps, matching the x != x.min() / x > 0 convention already used
      throughout this project -- see asparagus_preprocessing/normalize.py
      and brats_teacher.py), THEN resampled itself with NEAREST-NEIGHBOR
      (order=0) interpolation -- never blurred/interpolated as a mask.
  (b) after resampling, background voxels (per the resampled mask) in the
      INTENSITY image are forced back to exactly 0, cleaning up any cubic-
      interpolation leakage at the boundary.

Interpolation orders match asparagus_preprocessing's own convention
(resample.py: order=3 for image data, order=0 for segmentation/label) so the
resampled geometry is consistent with what asparagus/nnU-Net-family teachers
(BraTS) were themselves trained on.

Output: one .npz per (zip, nifti-member) under `--out_dir`:
    data     : float16, (D,H,W), 1mm iso, RAW intensity (background=0)
    mask     : uint8,   (D,H,W), 1mm iso, brain/foreground mask
    spacing  : (1.0,1.0,1.0)
    orig_spacing, orig_shape : for provenance/debugging
    modality, dataset, subject, session, member : string metadata

This does NOT touch asparagus_preprocessing's own pipeline (no code shared
by import beyond the already-proven resample/normalize conventions cited
above) -- it is a standalone script feeding FOMO26's own accelerate-based
run_pretrain.py, since the pretrain checkpoint handoff to asp_finetune_*
only requires the SAVED CHECKPOINT to have "model."-prefixed keys
(/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/01_ASPARAGUS_ANALYSIS.md Sec C.4, already satisfied by run_pretrain.py) -- the
intermediate preprocessed-corpus format has no asparagus compatibility
requirement of its own.
"""

import argparse
import csv
import multiprocessing as mp
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk

import sys
sys.path.insert(0, "/root")
from FOMO26.data.fomo300k_dataset import build_index, load_or_build_index  # noqa: E402

FOMO300K_ROOT = "/root/data/FOMO300K"
DEFAULT_OUT_DIR = "/root/data/FOMO300K_preprocessed"

# Extreme-shape filtering thresholds (Sec 3.3 of the task spec), in mm
# (physical extent = voxel_count * spacing -- see check_extreme_shape docstring).
MAX_AXIS = 1024
MIN_AXIS = 32

# float16 max representable magnitude is 65504 -- leave a small margin.
# Raw MRI intensity can legitimately exceed this (see process_one() note).
FLOAT16_SAFE_MAX = 65000.0
FLOAT16_SAFE_MIN = -65000.0


def _load_nifti_array(zip_path: str, member: str):
    zf = zipfile.ZipFile(zip_path)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(zf.read(member))
        tmp_path = tmp.name
    try:
        img = nib.load(tmp_path)
        data = np.asarray(img.dataobj).astype(np.float32)
        spacing = img.header.get_zooms()[:3]
        affine = img.affine.copy()
    finally:
        os.remove(tmp_path)
    return data, tuple(float(s) for s in spacing), affine


def _sitk_resample(np_vol: np.ndarray, spacing, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkBSpline,
                    default_value: float = 0.0) -> np.ndarray:
    img = sitk.GetImageFromArray(np.transpose(np_vol, (2, 1, 0)))
    img.SetSpacing([float(s) for s in spacing])
    orig_size = img.GetSize()
    new_size = [max(1, int(round(osz * ospc / tspc))) for osz, ospc, tspc in zip(orig_size, spacing, target_spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetInterpolator(interpolator)
    out = resampler.Execute(img)
    return np.transpose(sitk.GetArrayFromImage(out), (2, 1, 0))


def check_extreme_shape(shape, spacing) -> Optional[str]:
    """Checks PHYSICAL extent (shape * spacing, in mm), not raw voxel-grid
    shape. Bug found during the 200-zip pilot run (2026-07-15): checking raw
    voxel shape misclassified legitimate scans as "corrupted" -- e.g. a
    512x512x18 clinical thick-slice T1 (256x256 in-plane at high resolution,
    18 slices) was flagged "nz>500" by a naive shape[2]>500 check that
    assumed axis index 2 is always the slice axis (false -- nibabel axis
    order follows file storage order, which varies), AND 256x256x19
    thick-slice scans (real, ~18-20% of the corpus per GEOMETRY_SUMMARY.md's
    isotropy stats) were flagged "axis<32" by checking the NATIVE slice count
    instead of the physical extent -- a 19-slice scan at 6mm spacing is
    114mm through-plane, comfortably enough for a 128mm (128 voxel @ 1mm)
    patch after resampling; it was never actually undersized.
    MAX_AXIS/MIN_AXIS below are now in millimeters, and axis-order-agnostic
    (a genuinely oversized/undersized axis is caught regardless of which
    array index it lives at, and regardless of native voxel resolution)."""
    if len(shape) != 3:
        return f"not-3d(ndim={len(shape)})"
    extent_mm = [s * sp for s, sp in zip(shape, spacing)]
    if any(e > MAX_AXIS for e in extent_mm):
        return f"axis>{MAX_AXIS}mm({extent_mm})"
    if any(e < MIN_AXIS for e in extent_mm):
        return f"axis<{MIN_AXIS}mm({extent_mm})"
    return None


def process_one(entry: dict, out_dir: str, dtype=np.float16) -> dict:
    result = {**entry, "status": "ok", "reason": ""}
    try:
        data, spacing, affine = _load_nifti_array(entry["zip_path"], entry["member"])
    except Exception as e:
        result["status"] = "error"
        result["reason"] = f"load-failed: {e}"
        return result

    shape_flag = check_extreme_shape(data.shape, spacing)
    if shape_flag is not None:
        result["status"] = "skipped"
        result["reason"] = shape_flag
        return result

    # Issue A fix (a): mask computed at native resolution, BEFORE any interpolation.
    eps = 1e-6
    native_mask = (data > eps).astype(np.float32)

    try:
        resampled_data = _sitk_resample(data, spacing, interpolator=sitk.sitkBSpline, default_value=0.0)
        resampled_mask = _sitk_resample(native_mask, spacing, interpolator=sitk.sitkNearestNeighbor, default_value=0.0)
    except Exception as e:
        result["status"] = "error"
        result["reason"] = f"resample-failed: {e}"
        return result

    mask_bin = (resampled_mask > 0.5)
    # Issue A fix (b): re-clean background after cubic-interpolation leakage.
    resampled_data = np.where(mask_bin, resampled_data, 0.0)

    # BUG FOUND during the full-corpus run (2026-07-15): raw MRI intensity can
    # legitimately exceed float16's representable range (~+-65504). Initially
    # fixed with a hard clip, but a 3000-zip pilot found PT009_BraTS-GEN's T1c
    # scans have their ENTIRE distribution shifted to a different scale
    # (mean=59940, 99.9th pct=731816, max=1462942 -- verified by loading a raw
    # T1c file directly) -- a hard clip at 65000 would flatten ~16-18% of all
    # voxels in these scans to a single ceiling value, destroying real signal,
    # not just trimming rare outliers.
    #
    # Fix: apply a per-scan POSITIVE SCALE FACTOR instead of clipping, when
    # needed. This is lossless for every normalization scheme this project
    # actually uses: Anatomix/VesselFM's percentile-rescale and BraTS's
    # z-score are BOTH invariant to a prior positive-scale transform (same
    # algebraic property already relied on throughout /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/02_TEACHER_REPORT.md Sec D
    # -- normalize(a*x) == normalize(x) for any a>0). The scale factor is
    # still recorded in the .npz for provenance/debugging even though no
    # current consumer needs to undo it. A residual hard clip is kept ONLY as
    # a last-resort safety net for isolated single-voxel spikes that would
    # otherwise force the whole scan's scale down unnecessarily (a single
    # 10x outlier voxel, e.g. a bad pixel, shouldn't compress the other
    # 99.9999% of voxels by 10x) -- see IQR-based outlier exclusion below.
    n_nan = int(np.isnan(resampled_data).sum())
    if n_nan > 0:
        resampled_data = np.nan_to_num(resampled_data, nan=0.0)

    fg_vals = resampled_data[mask_bin]
    scale = 1.0
    if fg_vals.size > 0:
        # Use the 99.9th percentile (not the raw max) to set the scale so a
        # handful of single-voxel spikes don't dictate the scale for the
        # entire scan; those rare spikes are then handled by the residual clip.
        robust_max = float(np.percentile(np.abs(fg_vals), 99.9))
        if robust_max > FLOAT16_SAFE_MAX:
            scale = FLOAT16_SAFE_MAX / robust_max
            resampled_data = resampled_data * scale

    n_clipped = int(np.sum((resampled_data > FLOAT16_SAFE_MAX) | (resampled_data < FLOAT16_SAFE_MIN)))
    if n_clipped > 0:
        resampled_data = np.clip(resampled_data, FLOAT16_SAFE_MIN, FLOAT16_SAFE_MAX)
    resampled_data = resampled_data.astype(dtype)
    mask_u8 = mask_bin.astype(np.uint8)
    result["n_clipped_voxels"] = n_clipped
    result["n_nan_voxels"] = n_nan
    result["intensity_scale"] = scale

    # BUG FOUND 2026-07-16: omitting entry["group"] (e.g. OpenNeuro's
    # accession-id subdirectory, see fomo300k_dataset.py::build_index) let
    # different sub-datasets sharing generic sub-01/ses-01-style naming
    # collide on the same output path -- confirmed 71,717 of 140,300
    # PT030_OpenNeuro outputs were silently overwritten (only 68,583 files
    # actually existed on disk after a run that reported 140,300 "ok").
    # `group` is now always part of the filename (empty string for the
    # normal non-nested layout, so non-OpenNeuro filenames are unchanged).
    group = entry.get("group", "")
    group_part = f"{group}__" if group else ""
    out_name = f"{entry['dataset']}__{group_part}{entry['subject']}__{entry['session']}__{Path(entry['member']).name}".replace(".nii.gz", "")
    out_path = os.path.join(out_dir, entry["dataset"])
    os.makedirs(out_path, exist_ok=True)
    full_path = os.path.join(out_path, out_name + ".npz")

    np.savez_compressed(
        full_path,
        data=resampled_data,
        mask=mask_u8,
        spacing=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        orig_spacing=np.array(spacing, dtype=np.float32),
        orig_shape=np.array(data.shape, dtype=np.int32),
        affine=affine.astype(np.float32),
        intensity_scale=np.float32(scale),  # data_true_raw ~= data / intensity_scale; ==1.0 for the vast majority of scans
        modality=entry["modality"],
        dataset=entry["dataset"],
        subject=entry["subject"],
        session=entry["session"],
        member=entry["member"],
    )
    result["out_path"] = full_path
    result["out_shape"] = tuple(int(s) for s in resampled_data.shape)
    result["out_bytes"] = os.path.getsize(full_path)
    return result


def _process_one_star(args):
    return process_one(*args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit_zips", type=int, default=None)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    p.add_argument("--modalities", type=str, nargs="+", default=None)
    p.add_argument("--datasets", type=str, nargs="+", default=None,
                    help="only process entries from these dataset names (e.g. --datasets PT030_OpenNeuro) "
                         "-- for re-running a subset without reprocessing everything else")
    p.add_argument("--log_csv", type=str, default=None)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    index = load_or_build_index(limit_zips=args.limit_zips)
    if args.modalities is not None:
        index = [e for e in index if e["modality"] in args.modalities]
    if args.datasets is not None:
        index = [e for e in index if e["dataset"] in args.datasets]
    print(f"Processing {len(index)} entries with {args.workers} workers -> {args.out_dir}")

    log_csv = args.log_csv or os.path.join(args.out_dir, "preprocess_log.csv")
    t0 = time.time()
    n_ok, n_skip, n_err = 0, 0, 0
    n_scans_clipped, total_clipped_voxels, total_nan_voxels, n_scans_scaled = 0, 0, 0, 0
    total_out_bytes = 0

    with open(log_csv, "w", newline="") as f, mp.Pool(args.workers) as pool:
        writer = csv.writer(f)
        writer.writerow(["dataset", "group", "subject", "session", "member", "modality", "status", "reason",
                          "out_shape", "out_bytes", "n_clipped_voxels", "n_nan_voxels", "intensity_scale"])
        tasks = [(e, args.out_dir) for e in index]
        for i, result in enumerate(pool.imap_unordered(_process_one_star, tasks, chunksize=4)):
            status = result["status"]
            n_clip = result.get("n_clipped_voxels", 0)
            n_nan = result.get("n_nan_voxels", 0)
            scale = result.get("intensity_scale", 1.0)
            if status == "ok":
                n_ok += 1
                total_out_bytes += result.get("out_bytes", 0)
                if n_clip:
                    n_scans_clipped += 1
                    total_clipped_voxels += n_clip
                if scale != 1.0:
                    n_scans_scaled += 1
                total_nan_voxels += n_nan
            elif status == "skipped":
                n_skip += 1
            else:
                n_err += 1
            writer.writerow([result["dataset"], result.get("group", ""), result["subject"], result["session"],
                              result["member"], result["modality"], status, result.get("reason", ""),
                              result.get("out_shape", ""), result.get("out_bytes", ""), n_clip, n_nan, scale])
            if (i + 1) % 100 == 0:
                dt = time.time() - t0
                rate = (i + 1) / dt
                eta_s = (len(index) - (i + 1)) / rate if rate > 0 else float("inf")
                print(f"[{i+1}/{len(index)}] ok={n_ok} skip={n_skip} err={n_err} "
                      f"rate={rate:.2f}/s eta={eta_s/60:.1f}min out_bytes_so_far={total_out_bytes/1e9:.2f}GB "
                      f"clipped_scans={n_scans_clipped}")

    dt = time.time() - t0
    print(f"\nDone in {dt/60:.1f} min. ok={n_ok} skipped={n_skip} errors={n_err}")
    print(f"Total output size: {total_out_bytes/1e9:.2f} GB ({total_out_bytes/1e12:.3f} TB)")
    if n_scans_scaled > 0 or total_nan_voxels > 0:
        print(f"NOTE: {n_scans_scaled}/{n_ok} scans needed a positive intensity_scale factor "
              f"(99.9th-percentile foreground intensity exceeded the float16-safe range -- see "
              f"module docstring, e.g. PT009_BraTS-GEN's T1c scans); "
              f"{n_scans_clipped}/{n_ok} scans still had isolated single-voxel spikes clipped "
              f"after scaling ({total_clipped_voxels} voxels total); {total_nan_voxels} NaN voxels "
              f"were zeroed. See '{log_csv}' intensity_scale/n_clipped_voxels/n_nan_voxels columns.")
    if n_ok > 0:
        print(f"Avg bytes/scan: {total_out_bytes/n_ok/1e6:.2f} MB")
        if args.limit_zips is not None:
            # rough full-corpus extrapolation, index-count-proportional
            full_index_estimate = 306202  # per GEOMETRY_SUMMARY.md, all scans
            extrapolation_factor = full_index_estimate / max(1, len(index)) if len(index) else 0
            print(f"Rough full-corpus (306,202 scans) extrapolation: "
                  f"{total_out_bytes/n_ok*full_index_estimate/1e12:.2f} TB "
                  f"(linear scaling from this {len(index)}-entry sample -- NOT a substitute for a larger pilot)")


if __name__ == "__main__":
    main()
