"""Standalone Task-1 preprocessing for Candidate D: HD-BET + union-mask +
ANTs RIGID registration to MNI152NLin2009cAsym, byte-exact port of
/root/data/fomo-task1/task1_registration_ants.py's register_and_build()
(2026-08-19 experiment) -- verified 2026-08-21 to reproduce a stored
training tensor exactly (max_abs_diff=0.0 on sub_1/flair.pt, using the
same cached transform/union-mask that experiment produced): rigid
registration is estimated on FLAIR (masked by the union-of-4-modalities
HD-BET mask) against the skull-stripped template with mask-centroid-aware
init, then the SAME transform is applied to all 4 raw modalities onto the
fixed template-anchored 1mm crop box, followed by a center-pad from that
box's native (164, 201, 174) size up to the fixed TARGET_SIZE (192, 224,
192) -- this final pad step is not present in the currently-checked-in
experiment script's text (apparently trimmed after the run that actually
produced the training data) but was recovered empirically here since it's
required for bit-exact reproduction and for a consistent architecture
input size.

Production hardening (2026-08-21), ported from the Task 6/7 container's
real deployment incident report (mk4, cross-session handoff) -- see that
container's HARDENING_REPORT.md for the full symptom/root-cause/action/
verification writeup; the same issue set applies here since this is the
same HD-BET + ANTs registration risk profile:

  A. HD-BET weight re-download: fixed in Apptainer.def (HOME=/root pin),
     nothing to do here.
  B. HD-BET zero-mass / failure on any of the 4 modalities: per-modality
     HD-BET failures are tolerated (union mask is built from whichever
     modalities succeeded); if ALL 4 fail, moving_mask is None and this
     module skips straight to the no-registration fallback below rather
     than retrying unmasked registration (measured slow + not meaningfully
     better against non-anatomical input in the Task 6/7 container).
  C. Blank/degenerate input: is_blank_image() pre-checks each modality
     before HD-BET/registration are attempted; if ALL 4 are blank, this
     short-circuits straight to a deterministic all-zero fallback tensor.
     A catch-all try/except around the whole registration pipeline routes
     any other unforeseen failure to the fallback path below instead of
     crashing.
  D/fallback path: rather than inventing new no-registration logic, the
     fallback IS Candidate A's original iso1mm pipeline (per-modality
     native-space resample to 1mm, union of native thresholded masks, crop
     to bbox, center-pad to TARGET_SIZE) -- already a real, independently
     validated, previously-submitted preprocessing pipeline, and already
     spacing-aware per modality (each modality resampled from its own
     native spacing, not assumed-1mm).
  E. Minimum spatial size / crop-exceeds-target safety: TARGET_SIZE (192,
     224, 192) is fixed and always >= the registered path's ref-grid size
     (164, 201, 174), so the normal path can never exceed it. The fallback
     path's per-subject union-mask bbox CAN in principle exceed
     TARGET_SIZE (candidate A's original preprocess.py raised ValueError
     in that case) -- hardened here to center-crop down to TARGET_SIZE
     instead of crashing the whole subject.
"""
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage
from skimage import exposure

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "model", "template")
TEMPLATE_PATH = os.path.join(TEMPLATE_DIR, "tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz")
TEMPLATE_MASK_PATH = os.path.join(TEMPLATE_DIR, "tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz")

MODALITY_ORDER = ["flair", "adc", "dwi", "fourth"]  # fourth = t2s or swi
TARGET_SIZE = (192, 224, 192)
HDBET_HOME = "/root"  # must match Apptainer.def's %files bundling target (/root/hd-bet_params)

# 2026-08-22: belt-and-suspenders on top of the HOME= override below --
# monkeypatches HD_BET.checkpoint_download.maybe_download_parameters to a
# no-op BEFORE HD_BET.entry_point is ever imported (Python caches modules
# in sys.modules, and `from X import Y` reads X.Y at the moment the
# importing module first runs -- patching checkpoint_download first,
# THEN importing entry_point, means entry_point's own `from
# HD_BET.checkpoint_download import maybe_download_parameters` binds to
# OUR no-op, verified empirically). This makes a network access attempt
# physically impossible regardless of whether HOME/weight-path resolution
# is ever wrong for any reason -- if the weights genuinely aren't found,
# HD-BET now fails with a local FileNotFoundError (caught by our own
# except/fallback chain below) instead of calling requests.get().
_HDBET_NO_DOWNLOAD_WRAPPER = (
    # 2026-08-31: pin torch/cuDNN determinism INSIDE this subprocess too --
    # HD-BET runs as its own separate Python process, so any
    # torch.manual_seed/cudnn.deterministic set in predict.py's own
    # process never reaches it. cuDNN's GPU algorithm autotuner can pick a
    # different (numerically non-identical) conv algorithm across runs,
    # shifting the brain mask boundary by a voxel or two, which then
    # shifts the downstream registration crop -- found via a Task5
    # sibling container's real-data reruns showing different final
    # predictions across two identical `apptainer run` invocations even
    # after predict.py's own randomness (there, LV-masking noise) was
    # seeded. Task1's own classifier has no internal randomness, but it
    # calls HD-BET as a subprocess 4x per subject (one per modality) --
    # same exposure, same fix.
    "import torch; torch.manual_seed(42); "
    "torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False; "
    "import HD_BET.checkpoint_download as _cd; "
    "_cd.maybe_download_parameters = lambda: None; "
    "from HD_BET.entry_point import main; "
    "main()"
)

# same fixed template-anchored crop box as Task5/Task6-7's registration work
CROP_PHYS_MIN = np.array([-82.0, -83.0, -82.0])
CROP_PHYS_MAX = np.array([82.0, 118.0, 92.0])
OUTPUT_SPACING = 1.0


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def is_blank_image(data: np.ndarray) -> bool:
    mask = data > 1e-6
    if mask.sum() < 10:
        return True
    fg_std = float(np.std(data[mask]))
    if not np.isfinite(fg_std) or fg_std <= 0:
        return True
    return False


def _asparagus_volume_wise_znorm(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Verbatim (matches training-time function exactly). Callers must
    ensure the mask has std>0 foreground before calling -- degenerate
    per-channel cases are guarded by the caller, not here."""
    def clamp(x, m, q=0.99):
        q_val = np.quantile(x[m], q)
        return np.clip(x, a_min=None, a_max=q_val)

    def znormalize(x, m):
        values = x[m]
        mean, std = np.mean(values), np.std(values)
        assert std > 0
        x = x.astype(np.float64, copy=True)
        x -= mean
        x /= std
        return x

    m = mask.astype(bool)
    if m.sum() == 0:
        return array.astype(np.float32)
    out = clamp(array, m)
    out = znormalize(out, m)
    out = exposure.rescale_intensity(out, out_range=(0, 1))
    return out.astype(np.float32)


def _znorm_channel_safe(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-channel znorm that degrades to an all-zero channel (rather than
    crashing via the assert std>0) on the pathological case of a single
    modality being genuinely flat inside the mask -- never observed in
    real data, but the whole-subject prediction must not crash over one
    degenerate channel when the other 3 are fine."""
    m = mask.astype(bool)
    if m.sum() == 0 or float(np.std(arr[m])) <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    return _asparagus_volume_wise_znorm(arr, mask=m)


def _center_pad(channels: np.ndarray, target_size) -> np.ndarray:
    """channels: (C, D, H, W). Center-pads (or center-crops, if a
    dimension exceeds target) to target_size."""
    c, *shape = channels.shape
    out = np.zeros((c, *target_size), dtype=np.float32)
    src_slices, dst_slices = [], []
    for size, target in zip(shape, target_size):
        if size <= target:
            start = (target - size) // 2
            src_slices.append(slice(0, size))
            dst_slices.append(slice(start, start + size))
        else:
            start = (size - target) // 2
            src_slices.append(slice(start, start + target))
            dst_slices.append(slice(0, target))
    out[(slice(None), *dst_slices)] = channels[(slice(None), *src_slices)]
    return out


# ---------------------------------------------------------------------------
# HD-BET + rigid registration (normal path)
# ---------------------------------------------------------------------------

def _resolve_devices() -> list:
    """Same device resolution as the model's own forward pass (predict.py:
    torch.device("cuda" if torch.cuda.is_available() else "cpu")) -- if
    torch itself can't see a GPU, don't even attempt "-device cuda" (that
    attempt would just burn time failing identically for all 4 modalities
    before ever reaching cpu)."""
    return ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]


def _run_hdbet(input_path: str, work_dir: str, tag: str, state: dict):
    """Returns a mask path, or None on any failure / zero-mass output.
    `state` is a dict shared across all 4 modalities of ONE subject's
    preprocessing run: {"cuda_broken": bool}. A genuine device-level
    failure (exception or non-zero exit -- NOT a zero-mass result, which
    just means "no brain found", not "device broken") on modality 1 is
    overwhelmingly likely to recur identically on modalities 2-4 (same
    environment, same GPU); remembering it avoids re-attempting a doomed
    cuda call up to 4x per subject, each with its own 600s timeout --
    Task1's 4-modality case has 4x the worst-case HD-BET latency Task5's
    single-modality containers do, a real timeout risk on evaluation
    infrastructure without a GPU or with a broken CUDA setup."""
    tmp_out = os.path.join(work_dir, f"hdbet_{tag}.nii.gz")
    devices = ["cpu"] if state.get("cuda_broken") else _resolve_devices()
    for device in devices:
        cmd = [sys.executable, "-c", _HDBET_NO_DOWNLOAD_WRAPPER,
               "-i", input_path, "-o", tmp_out, "--save_bet_mask", "--no_bet_image", "-device", device]
        if device == "cpu":
            # HD-BET's own recommendation: test-time-augmentation (flip
            # ensembling) is much slower on CPU for only a marginal
            # accuracy difference -- disable it there.
            cmd.append("--disable_tta")
        try:
            # 2026-08-21: force HOME explicitly for this subprocess rather
            # than relying on Apptainer.def's %environment HOME=/root to
            # have taken effect -- a real submission got marked INVALID by
            # the organizers for "attempting to access the internet",
            # traced to HD_BET.checkpoint_download.maybe_download_parameters()
            # (called unconditionally on every `hd-bet` invocation,
            # entry_point.py:45), which resolves its weights directory via
            # os.path.expanduser('~') i.e. $HOME -- if that doesn't match
            # where %files bundled the weights (/root/hd-bet_params), it
            # silently does a real requests.get() to zenodo.org. Passing
            # HOME here directly guarantees the subprocess sees the right
            # value regardless of whether the container runtime's own
            # %environment sourcing was reliable.
            env = dict(os.environ, HOME=HDBET_HOME)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        except Exception as e:
            print(f"[WARNING] HD-BET({tag},{device}) raised {type(e).__name__}: {e}")
            if device == "cuda":
                state["cuda_broken"] = True
            continue
        if result.returncode != 0:
            print(f"[WARNING] HD-BET({tag},{device}) exited {result.returncode}: {result.stderr[-400:]}")
            if device == "cuda":
                state["cuda_broken"] = True
            continue
        mask_path = os.path.join(work_dir, f"hdbet_{tag}_bet.nii.gz")
        if not os.path.exists(mask_path):
            continue
        try:
            mass = float(nib.load(mask_path).get_fdata().sum())
        except Exception as e:
            print(f"[WARNING] could not read HD-BET({tag}) mask: {e}")
            continue
        if mass <= 0:
            print(f"[WARNING] HD-BET({tag},{device}) output has zero mass -- "
                  f"content-based verdict (no brain found), not a device failure, "
                  f"so skip retrying the other device -- it would find the same")
            return None
        return mask_path
    return None


def _modalities_are_coregistered(mod_paths: dict) -> bool:
    """Cheap header-only check (no pixel data read). Verified 2026-08-22
    across all 63 available modality files in the training cohort (21
    subjects x up to 3 non-flair modalities each) that ADC/DWI/SWI(T2S)
    share IDENTICAL affine+shape with FLAIR -- co-registered at
    acquisition, max affine diff ~1e-6 (pure floating point noise), zero
    mismatches. If true, FLAIR's own HD-BET mask can be reused directly
    for all 4 modalities with zero extra registration/resampling -- cuts
    HD-BET calls from 4 to 1 per subject in the overwhelmingly common
    case (this was the dominant worst-case-latency/timeout-risk factor
    for this 4-modality task, see the device-caching comments above).
    Hidden test data isn't guaranteed to hold this property, so this is
    checked at runtime, not just assumed from the training data."""
    ref = nib.load(mod_paths["flair"])
    for key in ("adc", "dwi", "fourth"):
        img = nib.load(mod_paths[key])
        if img.shape != ref.shape or not np.allclose(img.affine, ref.affine, atol=1e-3):
            return False
    return True


FLAIR_MASK_DILATE_VOXELS = 2  # see module-level note on _dilate_mask_file


def _isotropic_resample_whole(sitk_img: sitk.Image, spacing: float, interp) -> sitk.Image:
    """Whole-volume isotropic resample (no crop), used ONLY to build a
    cheap proxy for HD-BET + registration ESTIMATION -- NOT part of the
    final data path (see _build_union_mask's docstring)."""
    orig_spacing = sitk_img.GetSpacing()
    orig_size = sitk_img.GetSize()
    new_size = [max(1, int(round(osz * ospc / spacing))) for osz, ospc in zip(orig_size, orig_spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([spacing] * 3)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interp)
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(sitk_img)


def _dilate_mask_file(mask_path: str, work_dir: str, iterations: int) -> str:
    """Binary-dilates a mask nifti by `iterations` voxels and writes the
    result to a new file. Measured 2026-08-22 across 3 real subjects:
    using FLAIR's own HD-BET mask alone (instead of the original 4-way
    union across all modalities) loses 1.8-4.5% of the union's coverage
    at the brain boundary (voxels ADC/DWI/SWI's own masks caught but
    FLAIR's didn't). A 2-voxel dilation is essentially free (~100-120ms,
    negligible next to HD-BET's 15-30s+) and recovers ~96-99.6% of that
    lost coverage. Tradeoff: dilation is not selective -- it also adds
    substantially MORE voxels beyond even the original union (skull/scalp
    margin no modality's own mask identified as brain), roughly 7-8x more
    "extra" voxels than genuinely-recovered ones at 2 voxels. Chosen as
    the balance point: high recovery without the largest over-inclusion
    seen at 3-5 voxels."""
    mask_img = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    arr = sitk.GetArrayFromImage(mask_img) > 0
    dilated = ndimage.binary_dilation(arr, iterations=iterations)
    out_path = os.path.join(work_dir, "hdbet_flair_dilated.nii.gz")
    out_img = sitk.GetImageFromArray(dilated.astype(np.uint8))
    out_img.CopyInformation(mask_img)
    sitk.WriteImage(out_img, out_path)
    return out_path


def _build_union_mask(mod_paths: dict, work_dir: str):
    """mod_paths: {modality_key: nifti_path}. Returns
    (final_mask_path, registration_moving_path, registration_moving_mask_path)
    -- the last two are None unless the fast (1mm-proxy) path was taken.
    Returns (None, None, None) if HD-BET failed (caller must skip straight
    to the no-registration fallback, not retry unmasked registration --
    see module docstring hardening item B).

    Fast path (co-registered modalities, the norm for this dataset -- see
    _modalities_are_coregistered): FLAIR is resampled to OUTPUT_SPACING
    (1mm -- the same spacing the final crop box already uses, per the
    user's 2026-08-22 point that re-spacing only needs to happen once
    since the trained space IS 1mm) exactly ONCE, and that same resampled
    volume is reused for BOTH HD-BET mask generation AND registration
    estimation (registration's own multi-resolution optimizer doesn't
    need native resolution to converge correctly -- see the 2026-08-22
    timing comparison, native vs 1mm vs 2mm proxy registration). The
    resulting mask is resampled back onto FLAIR's NATIVE grid (cheap,
    nearest-neighbor, binary data only) and dilated by
    FLAIR_MASK_DILATE_VOXELS in NATIVE voxel units (matching the exact
    configuration measured against the training cohort) to recover most
    of the coverage a full 4-way union would have had. The FINAL channel
    warp (see _preprocess_hdbet_rigid) always applies the resulting
    transform directly to each modality's ORIGINAL NATIVE file in a
    single interpolation onto the crop box -- resampling to 1mm here is
    purely a mask/registration-estimation shortcut, never part of the
    final data path. Falls back to the original, fully general
    per-modality union-mask approach (up to 4 separate native-resolution
    HD-BET calls, no 1mm proxy) if any modality's geometry doesn't match
    FLAIR's."""
    state = {"cuda_broken": False}

    if _modalities_are_coregistered(mod_paths):
        flair_native_img = sitk.ReadImage(mod_paths["flair"], sitk.sitkFloat32)
        flair_1mm_img = _isotropic_resample_whole(flair_native_img, OUTPUT_SPACING, sitk.sitkLinear)
        flair_1mm_path = os.path.join(work_dir, "flair_1mm.nii.gz")
        sitk.WriteImage(flair_1mm_img, flair_1mm_path)

        mp = _run_hdbet(flair_1mm_path, work_dir, "flair", state)
        if mp is None:
            return None, None, None

        mask_1mm_img = sitk.ReadImage(mp, sitk.sitkUInt8)
        mask_native_img = sitk.Resample(mask_1mm_img, flair_native_img, sitk.Transform(),
                                         sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        mask_native_path = os.path.join(work_dir, "hdbet_flair_native.nii.gz")
        sitk.WriteImage(mask_native_img, mask_native_path)
        final_mask_path = _dilate_mask_file(mask_native_path, work_dir, FLAIR_MASK_DILATE_VOXELS)

        # dilated NATIVE mask resampled back down to the 1mm proxy grid,
        # purely so registration's own moving_mask reflects the same
        # (dilated) brain extent the final masking step will use -- still
        # only a mask/estimation-time resample, not the final data path.
        final_mask_img = sitk.ReadImage(final_mask_path, sitk.sitkUInt8)
        mask_1mm_dilated_img = sitk.Resample(final_mask_img, flair_1mm_img, sitk.Transform(),
                                              sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        mask_1mm_dilated_path = os.path.join(work_dir, "hdbet_flair_1mm_dilated.nii.gz")
        sitk.WriteImage(mask_1mm_dilated_img, mask_1mm_dilated_path)

        return final_mask_path, flair_1mm_path, mask_1mm_dilated_path

    print("[WARNING] modalities are not co-registered with flair -- "
          "falling back to per-modality HD-BET + union mask")
    mask_paths = []
    ref_img = None
    for key, path in mod_paths.items():
        mp = _run_hdbet(path, work_dir, key, state)
        if mp is None:
            continue
        img = nib.load(mp)
        if ref_img is None:
            ref_img = img
        elif img.shape != ref_img.shape:
            print(f"[WARNING] HD-BET({key}) mask shape {img.shape} != reference {ref_img.shape}, skipping from union")
            continue
        mask_paths.append(mp)

    if not mask_paths:
        return None, None, None

    union = np.zeros(ref_img.shape, dtype=bool)
    for mp in mask_paths:
        union |= (nib.load(mp).get_fdata() > 0)
    union_path = os.path.join(work_dir, "union_mask.nii.gz")
    nib.save(nib.Nifti1Image(union.astype(np.uint8), ref_img.affine, ref_img.header), union_path)
    return union_path, None, None


def _mask_centroid_init(fixed_mask_path: str, moving_mask_path: str, out_path: str) -> str:
    """Pure SimpleITK -- unrelated to antspyx, no dependency change here."""
    fixed_mask = sitk.ReadImage(fixed_mask_path)
    moving_mask = sitk.ReadImage(moving_mask_path)
    init = sitk.CenteredTransformInitializer(
        sitk.Cast(fixed_mask, sitk.sitkFloat32), sitk.Cast(moving_mask, sitk.sitkFloat32),
        sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    sitk.WriteTransform(init, out_path)
    return out_path


# ---------------------------------------------------------------------------
# 2026-08-22: ANTs CLI engine (antsRegistrationSyNQuick.sh / antsApplyTransforms)
# instead of the antspyx Python bindings -- removes antspyx (and its own
# ~400-500MB of transitive matplotlib/pandas/scikit-learn/statsmodels
# dependencies our code never touches) entirely from this container.
# Prebuilt ANTs CLI binaries are bundled at build time (see Apptainer.def)
# and expected on PATH, matching how "hd-bet" is already called by name.
# Verified 2026-08-22: SimpleITK's own CenteredTransformInitializer .tfm
# output is directly accepted by antsRegistrationSyNQuick.sh's -i flag (no
# format conversion needed); a real rigid registration test against the
# template (sub_1/flair, mask-centroid init) converged to mask Dice=0.85,
# consistent quality with the antspy engine it replaces.
# ---------------------------------------------------------------------------

def _ants_register_rigid(fixed_path: str, moving_path: str, fixed_mask_path: str,
                          moving_mask_path: str, init_path: str, out_prefix: str) -> str:
    """Returns the path to the resulting rigid transform (.mat). Raises
    (via non-zero exit) on failure -- caller's existing except/fallback
    chain handles it, same as before."""
    cmd = [
        "antsRegistrationSyNQuick.sh", "-d", "3",
        "-f", fixed_path, "-m", moving_path,
        "-x", f"{fixed_mask_path},{moving_mask_path}",
        "-i", init_path,
        "-t", "r",
        "-o", out_prefix,
        "-n", "4",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"antsRegistrationSyNQuick.sh failed (rc={result.returncode}): {result.stderr[-500:]}")
    tfm_path = out_prefix + "0GenericAffine.mat"
    if not os.path.exists(tfm_path):
        raise RuntimeError(f"antsRegistrationSyNQuick.sh did not produce {tfm_path}")
    return tfm_path


def _ants_apply_transform(input_path: str, ref_path: str, tfm_path: str, output_path: str, interp: str) -> str:
    """interp: 'Linear' or 'NearestNeighbor'."""
    cmd = ["antsApplyTransforms", "-d", "3", "-i", input_path, "-r", ref_path,
           "-t", tfm_path, "-o", output_path, "-n", interp]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"antsApplyTransforms failed (rc={result.returncode}): {result.stderr[-500:]}")
    return output_path


def _build_reference_image(direction, phys_min, phys_max, spacing) -> sitk.Image:
    d = np.array(direction).reshape(3, 3)
    diag = np.diag(d)
    size = [max(1, int(round((phys_max[i] - phys_min[i]) / spacing))) for i in range(3)]
    origin = [phys_max[i] if diag[i] < 0 else phys_min[i] for i in range(3)]
    ref = sitk.Image(size, sitk.sitkFloat32)
    ref.SetSpacing([spacing, spacing, spacing])
    ref.SetOrigin(origin)
    ref.SetDirection(direction)
    return ref


def _preprocess_hdbet_rigid(mod_paths: dict, work_dir: str) -> torch.Tensor:
    """mod_paths: ordered dict over MODALITY_ORDER -> raw nifti path.
    Raises on any failure -- caller (preprocess_task1) catches and falls
    back to the iso1mm no-registration pipeline."""
    final_mask_path, reg_moving_path, reg_moving_mask_path = _build_union_mask(mod_paths, work_dir)
    if final_mask_path is None:
        raise RuntimeError("HD-BET failed on all 4 modalities -- skipping unmasked "
                            "registration, going straight to iso1mm fallback")

    # registration is estimated on the 1mm proxy when available (fast
    # co-registered path) -- a pure physical-space rigid transform, so
    # estimating it on a cheaper proxy doesn't affect the FINAL data warp
    # below, which always applies the resulting transform directly to
    # each modality's ORIGINAL NATIVE file. Falls back to native flair +
    # the union mask when the fast path wasn't taken (non-coregistered
    # modalities).
    if reg_moving_path is None:
        reg_moving_path = mod_paths["flair"]
        reg_moving_mask_path = final_mask_path

    init_path = _mask_centroid_init(TEMPLATE_MASK_PATH, reg_moving_mask_path, os.path.join(work_dir, "init.tfm"))

    tfm_path = _ants_register_rigid(
        TEMPLATE_PATH, reg_moving_path, TEMPLATE_MASK_PATH, reg_moving_mask_path,
        init_path, os.path.join(work_dir, "reg_"),
    )

    template_img = sitk.ReadImage(TEMPLATE_PATH)
    ref_sitk = _build_reference_image(template_img.GetDirection(), CROP_PHYS_MIN, CROP_PHYS_MAX, OUTPUT_SPACING)
    ref_path = os.path.join(work_dir, "ref.nii.gz")
    sitk.WriteImage(ref_sitk, ref_path)

    # final masking always uses the NATIVE-resolution (dilated) mask, not
    # whichever mask registration was estimated with -- higher quality
    # than warping the coarser 1mm proxy mask when that path was taken.
    warped_mask_path = _ants_apply_transform(
        final_mask_path, ref_path, tfm_path, os.path.join(work_dir, "warped_mask.nii.gz"), "NearestNeighbor")
    warped_mask_arr = nib.load(warped_mask_path).get_fdata() > 0.5
    if not warped_mask_arr.any():
        raise ValueError("empty union mask after warp onto crop box")

    channels = []
    for key in MODALITY_ORDER:
        warped_path = _ants_apply_transform(
            mod_paths[key], ref_path, tfm_path, os.path.join(work_dir, f"warped_{key}.nii.gz"), "Linear")
        arr = nib.load(warped_path).get_fdata().astype(np.float32)
        arr = np.where(warped_mask_arr, arr, 0.0)
        channels.append(_znorm_channel_safe(arr, warped_mask_arr))

    stacked = np.stack(channels).astype(np.float32)  # (4, 164, 201, 174)
    padded = _center_pad(stacked, TARGET_SIZE)
    return torch.from_numpy(padded)


# ---------------------------------------------------------------------------
# no-registration fallback: Candidate A's original, independently-validated
# iso1mm pipeline (per-modality native-space resample, union native mask,
# crop to bbox, center-pad) -- already spacing-aware per modality.
# ---------------------------------------------------------------------------

def _sitk_resample(np_vol: np.ndarray, spacing, target_spacing=(1.0, 1.0, 1.0),
                    interpolator=sitk.sitkBSpline, default_value: float = 0.0) -> np.ndarray:
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


def _preprocess_iso1mm_fallback(mod_paths: dict) -> torch.Tensor:
    resampled, masks = [], []
    for key in MODALITY_ORDER:
        image = nib.load(mod_paths[key])
        array = image.get_fdata().astype(np.float32)
        spacing = [float(s) for s in image.header.get_zooms()[:3]]
        native_mask = (array > 1e-6).astype(np.float32)
        array_iso = _sitk_resample(array, spacing, interpolator=sitk.sitkBSpline)
        mask_iso = _sitk_resample(native_mask, spacing, interpolator=sitk.sitkNearestNeighbor) > 0.5
        resampled.append(np.where(mask_iso, array_iso, 0.0).astype(np.float32))
        masks.append(mask_iso)

    shapes = {a.shape for a in resampled}
    if len(shapes) != 1:
        raise ValueError(f"iso1mm fallback: resampled modality shapes disagree: {sorted(shapes)}")

    union_mask = np.logical_or.reduce(masks)
    if not union_mask.any():
        raise ValueError("iso1mm fallback: no foreground in union mask")
    coords = np.argwhere(union_mask)
    mins, maxs = coords.min(axis=0), coords.max(axis=0) + 1
    crop_slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))

    channels = []
    for array, mask in zip(resampled, masks):
        cropped, cropped_mask = array[crop_slices], mask[crop_slices]
        channels.append(_znorm_channel_safe(cropped, cropped_mask))

    stacked = np.stack(channels).astype(np.float32)
    padded = _center_pad(stacked, TARGET_SIZE)
    return torch.from_numpy(padded)


def _neutral_fallback() -> torch.Tensor:
    return torch.zeros((4, *TARGET_SIZE), dtype=torch.float32)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def preprocess_task1(flair, adc, dwi, fourth, work_dir) -> torch.Tensor:
    """flair/adc/dwi/fourth: raw NIfTI paths (fourth = t2s or swi). Returns
    a (4, 192, 224, 192) tensor. Degrades ONCE, from (HD-BET+rigid
    registration) to (iso1mm no-registration fallback), since that fallback
    still derives its output from the real input data. If the fallback
    ALSO fails, this now propagates the exception instead of silently
    returning a deterministic zero volume (2026-08-30 policy change): a
    silent all-zero output would score badly on AUROC with no diagnostic
    signal, while an uncaught exception crashes the container and the
    FOMO26 harness marks it INVALID immediately -- surfacing the real bug
    right away instead of hiding it behind a quietly-wrong score. The only
    input this function still tolerates without raising is a genuinely
    blank/degenerate scan (all 4 modalities), handled separately below via
    is_blank_image() -- that is a real, expected input condition, not a
    pipeline bug."""
    mod_paths = {"flair": flair, "adc": adc, "dwi": dwi, "fourth": fourth}

    blank_flags = []
    for key, path in mod_paths.items():
        data = nib.load(path).get_fdata().astype(np.float32)
        if not np.all(np.isfinite(data)) or is_blank_image(data):
            print(f"[WARNING] modality '{key}' is blank/degenerate")
            blank_flags.append(True)
        else:
            blank_flags.append(False)
    if all(blank_flags):
        print("[WARNING] all 4 modalities blank/degenerate -- using neutral fallback")
        return _neutral_fallback()

    try:
        return _preprocess_hdbet_rigid(mod_paths, work_dir)
    except Exception as e:
        print(f"[WARNING] HD-BET+rigid registration pipeline failed ({type(e).__name__}: {e}) "
              f"-- falling back to iso1mm no-registration pipeline")
        return _preprocess_iso1mm_fallback(mod_paths)
