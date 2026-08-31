"""Standalone Task5 preprocessing for the affine_iso10 submission
container. Port of /root/fomo26task5/scripts/task5_registration_ants.py
(rigid init -> chained affine registration, mask-centroid-aware init,
Dice-gated fallback to rigid if affine regresses) +
task5_build_skullstripped_datasets.py's affine branch (skull-stripped
image warped onto the fixed MNI-space crop box) + task5_pad_to_32.py
(pads the fixed ref-grid warp output up to the next multiple of 32 per
axis) + task5_preprocess_pt.py's znorm step, condensed into a
single-subject pipeline. Verified 2026-08-21 to reproduce a stored
training tensor exactly (max_abs_diff=0.0 on sub_13, using that
experiment's cached HD-BET mask + registration transform; a small amount
of additional variance from a fully-fresh HD-BET+ANTs run is expected
registration-algorithm nondeterminism, not a preprocessing bug -- same
caveat as Task 1 Candidate D's preprocess.py).

Unlike hdbet_only_iso07 (crop to the subject's OWN bbox, cohort-dependent
canvas), rigid/affine warp onto a FIXED template-anchored physical box
(CROP_PHYS_MIN/MAX, same box used throughout this project's Task5/Task1
registration work) at 1.0mm isotropic -- every subject lands on the exact
same (164, 201, 174) grid before the pad_to_32 pass, so there is no
cohort-statistic dependency and no two-stage-pad pitfall (see
hdbet_only_iso07's preprocess.py docstring for that pitfall, which does
NOT apply here).

Production hardening (2026-08-21, same incident-driven principles as
every other container in this project): HD-BET failure -> spacing-aware
no-registration fallback (resample to 1mm, crop to a plain
intensity-threshold bbox, pad the same way); blank/degenerate input ->
deterministic zero volume. If the no-registration fallback ALSO fails,
this now crashes on purpose (2026-08-30 policy change) instead of
silently returning a zero volume -- see preprocess_task5()'s docstring
for why.
"""
import os
import subprocess
import sys

import ants
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from skimage import exposure

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "model", "template")
TEMPLATE_PATH = os.path.join(TEMPLATE_DIR, "tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz")
TEMPLATE_MASK_PATH = os.path.join(TEMPLATE_DIR, "tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz")

CROP_PHYS_MIN = np.array([-82.0, -83.0, -82.0])
CROP_PHYS_MAX = np.array([82.0, 118.0, 92.0])
OUTPUT_SPACING = 1.0
REF_GRID_SHAPE = (164, 201, 174)      # x,y,z -- fixed template-anchored crop box
FINAL_SHAPE = (192, 224, 192)         # x,y,z -- next multiple of 32 per axis
USE_AFFINE = True                     # this container: rigid init -> chained affine, Dice-gated
HDBET_HOME = "/root"  # must match Apptainer.def's %files bundling target (/root/hd-bet_params)

# 2026-08-22: belt-and-suspenders on top of the HOME= override below --
# monkeypatches HD_BET.checkpoint_download.maybe_download_parameters to a
# no-op BEFORE HD_BET.entry_point is ever imported (Python caches modules
# in sys.modules, and `from X import Y` reads X.Y at the moment the
# importing module first runs -- patching checkpoint_download first, THEN
# importing entry_point, means entry_point's own `from
# HD_BET.checkpoint_download import maybe_download_parameters` binds to
# OUR no-op, verified empirically). This makes a network access attempt
# physically impossible regardless of whether HOME/weight-path resolution
# is ever wrong for any reason -- if the weights genuinely aren't found,
# HD-BET now fails with a local FileNotFoundError (caught by our own
# except/fallback chain below) instead of calling requests.get().
_HDBET_NO_DOWNLOAD_WRAPPER = (
    # 2026-08-31: HD-BET runs as its OWN subprocess/Python process, so
    # predict.py's torch.manual_seed/cudnn.deterministic settings never
    # reach it -- confirmed as a real bug via real-data reruns showing
    # DIFFERENT final classification probabilities across two identical
    # `apptainer run` invocations (0.337 vs 0.300) even after those
    # settings were added to predict.py itself. cuDNN's GPU algorithm
    # autotuner picking a different (numerically non-identical) conv
    # algorithm inside HD-BET's own nnU-Net inference shifts the brain
    # mask boundary by a voxel or two, which then shifts the downstream
    # affine registration crop, amplifying into a much bigger difference
    # by the time it reaches the classifier. Pinned here too, inside the
    # subprocess itself, for the same reason as predict.py's own copy.
    "import torch; torch.manual_seed(42); "
    "torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False; "
    "import HD_BET.checkpoint_download as _cd; "
    "_cd.maybe_download_parameters = lambda: None; "
    "from HD_BET.entry_point import main; "
    "main()"
)


def is_blank_image(data: np.ndarray) -> bool:
    mask = data > 1e-6
    if mask.sum() < 10:
        return True
    fg_std = float(np.std(data[mask]))
    if not np.isfinite(fg_std) or fg_std <= 0:
        return True
    return False


def _asparagus_volume_wise_znorm(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
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


def _znorm_safe(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if m.sum() == 0 or float(np.std(arr[m])) <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    return _asparagus_volume_wise_znorm(arr, mask=m)


def _center_pad(arr: np.ndarray, target_shape) -> np.ndarray:
    out = np.zeros(target_shape, dtype=np.float32)
    src_slices, dst_slices = [], []
    for size, target in zip(arr.shape, target_shape):
        if size <= target:
            start = (target - size) // 2
            src_slices.append(slice(0, size))
            dst_slices.append(slice(start, start + size))
        else:
            start = (size - target) // 2
            src_slices.append(slice(start, start + target))
            dst_slices.append(slice(0, target))
    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out


def _resolve_devices() -> list:
    """Same device resolution as the model's own forward pass -- if torch
    itself can't see a GPU, don't even attempt "-device cuda" (that
    attempt would just burn time failing before ever reaching cpu)."""
    return ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]


def _run_hdbet(input_path: str, work_dir: str):
    tmp_out = os.path.join(work_dir, "hdbet_tmp.nii.gz")
    for device in _resolve_devices():
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
            # have taken effect -- a sibling Task6/7 submission got marked
            # INVALID by the organizers for "attempting to access the
            # internet", traced to
            # HD_BET.checkpoint_download.maybe_download_parameters()
            # (called unconditionally on every `hd-bet` invocation), which
            # resolves its weights directory via os.path.expanduser('~')
            # i.e. $HOME -- if that doesn't match where %files bundled the
            # weights, it silently does a real requests.get() to
            # zenodo.org. Passing HOME here directly guarantees the
            # subprocess sees the right value regardless of whether the
            # container runtime's own %environment sourcing was reliable.
            env = dict(os.environ, HOME=HDBET_HOME)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        except Exception as e:
            print(f"[WARNING] HD-BET({device}) raised {type(e).__name__}: {e}")
            continue
        if result.returncode != 0:
            print(f"[WARNING] HD-BET({device}) exited {result.returncode}: {result.stderr[-400:]}")
            continue
        mask_path = os.path.join(work_dir, "hdbet_tmp_bet.nii.gz")
        if not os.path.exists(mask_path):
            continue
        try:
            mass = float(nib.load(mask_path).get_fdata().sum())
        except Exception as e:
            print(f"[WARNING] could not read HD-BET mask: {e}")
            continue
        if mass <= 0:
            print(f"[WARNING] HD-BET({device}) output has zero mass -- "
                  f"content-based verdict (no brain found), not a device failure, "
                  f"so skip retrying the other device -- it would find the same")
            return None
        return mask_path
    return None


def _isotropic_resample_whole(sitk_img: sitk.Image, spacing: float, interp) -> sitk.Image:
    """Whole-volume isotropic resample (no crop), used ONLY to build a
    cheap proxy for HD-BET + registration ESTIMATION -- NOT part of the
    final data path (see preprocess_task5's docstring note)."""
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


def _mask_centroid_init(fixed_mask_path: str, moving_mask_path: str, out_path: str) -> str:
    fixed_mask = sitk.ReadImage(fixed_mask_path)
    moving_mask = sitk.ReadImage(moving_mask_path)
    init = sitk.CenteredTransformInitializer(
        sitk.Cast(fixed_mask, sitk.sitkFloat32), sitk.Cast(moving_mask, sitk.sitkFloat32),
        sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    sitk.WriteTransform(init, out_path)
    return out_path


def _mask_dice(warped_mask_arr: np.ndarray, fixed_mask_arr: np.ndarray) -> float:
    a = warped_mask_arr > 0
    b = fixed_mask_arr > 0
    denom = a.sum() + b.sum()
    return 2 * (a & b).sum() / denom if denom > 0 else 0.0


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


def _register(t1_path: str, mask_path: str, work_dir: str):
    """Returns final ANTs fwdtransforms for warping onto the template box.
    Rigid-only for this container (USE_AFFINE=False); rigid->affine with a
    Dice-gated fallback to rigid (matching task5_registration_ants.py
    exactly) when USE_AFFINE=True (the affine container subclasses this
    module's behavior via that flag)."""
    fixed = ants.image_read(TEMPLATE_PATH)
    fixed_mask = ants.image_read(TEMPLATE_MASK_PATH)
    moving = ants.image_read(t1_path)
    moving_mask = ants.image_read(mask_path)

    init_path = _mask_centroid_init(TEMPLATE_MASK_PATH, mask_path, os.path.join(work_dir, "init.tfm"))
    rigid_reg = ants.registration(
        fixed, moving, type_of_transform="Rigid",
        mask=fixed_mask, moving_mask=moving_mask, mask_all_stages=True,
        initial_transform=init_path,
    )
    if not USE_AFFINE:
        return rigid_reg["fwdtransforms"]

    affine_reg = ants.registration(
        fixed, moving, type_of_transform="Affine",
        mask=fixed_mask, moving_mask=moving_mask, mask_all_stages=True,
        initial_transform=rigid_reg["fwdtransforms"][0],
    )
    fixed_mask_arr = fixed_mask.numpy()
    rigid_warped_mask = ants.apply_transforms(fixed, moving_mask, rigid_reg["fwdtransforms"], interpolator="nearestNeighbor")
    affine_warped_mask = ants.apply_transforms(fixed, moving_mask, affine_reg["fwdtransforms"], interpolator="nearestNeighbor")
    rigid_dice = _mask_dice(rigid_warped_mask.numpy(), fixed_mask_arr)
    affine_dice = _mask_dice(affine_warped_mask.numpy(), fixed_mask_arr)
    if affine_dice < rigid_dice - 0.01:
        return rigid_reg["fwdtransforms"]
    return affine_reg["fwdtransforms"]


def _build_registered(t1_native_path: str, mask_native_path: str,
                       t1_reg_path: str, mask_reg_path: str, work_dir: str) -> torch.Tensor:
    """t1_native_path/mask_native_path: ORIGINAL native-resolution T1 +
    its mask (resampled back from the proxy, see preprocess_task5) --
    used for the actual data warp, exactly once, native -> crop box.
    t1_reg_path/mask_reg_path: whatever resolution registration was
    ESTIMATED on (the 1mm proxy in the fast path, or the same native
    pair in the fallback) -- the resulting transform is a pure
    physical-space rigid/affine matrix, so estimating it on a cheaper
    proxy doesn't affect the final warp's quality."""
    tfm = _register(t1_reg_path, mask_reg_path, work_dir)

    template_img = sitk.ReadImage(TEMPLATE_PATH)
    ref_sitk = _build_reference_image(template_img.GetDirection(), CROP_PHYS_MIN, CROP_PHYS_MAX, OUTPUT_SPACING)
    ref_path = os.path.join(work_dir, "ref.nii.gz")
    sitk.WriteImage(ref_sitk, ref_path)
    ref_ants = ants.image_read(ref_path)

    moving = ants.image_read(t1_native_path)
    moving_mask = ants.image_read(mask_native_path)
    stripped = moving * moving_mask  # skull-strip BEFORE warp, matches training (mask * raw T1)

    warped_img = ants.apply_transforms(ref_ants, stripped, tfm, interpolator="linear")
    warped_mask = ants.apply_transforms(ref_ants, moving_mask, tfm, interpolator="nearestNeighbor")
    img_arr = warped_img.numpy()
    mask_arr = warped_mask.numpy() > 0.5
    if not mask_arr.any():
        raise ValueError("empty mask after warp onto crop box")

    img_padded = _center_pad(img_arr, FINAL_SHAPE)
    mask_padded = _center_pad(mask_arr.astype(np.float32), FINAL_SHAPE) > 0.5
    normed = _znorm_safe(img_padded, mask_padded)
    return torch.from_numpy(normed[None])  # (1, x, y, z) -- ants arrays are already x,y,z order


def _resample_isotropic_1mm(nifti_path: str, work_dir: str) -> str:
    img = ants.image_read(nifti_path)
    resampled = ants.resample_image(img, (OUTPUT_SPACING,) * 3, use_voxels=False, interp_type=1)
    out_path = os.path.join(work_dir, "resampled_1mm.nii.gz")
    ants.image_write(resampled, out_path)
    return out_path


def _build_no_registration_fallback(t1_path: str, work_dir: str) -> torch.Tensor:
    """Spacing-aware resample (no registration), plain intensity-threshold
    mask, crop to bbox, pad to FINAL_SHAPE -- last resort before the
    neutral fallback."""
    resampled_path = _resample_isotropic_1mm(t1_path, work_dir)
    arr = ants.image_read(resampled_path).numpy()
    mask_arr = arr > 1e-6
    if not mask_arr.any():
        raise ValueError("no foreground after no-registration resample")
    coords = np.argwhere(mask_arr)
    mins, maxs = coords.min(axis=0), coords.max(axis=0) + 1
    crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))
    cropped_arr, cropped_mask = arr[crop], mask_arr[crop]

    img_padded = _center_pad(np.where(cropped_mask, cropped_arr, 0.0).astype(np.float32), FINAL_SHAPE)
    mask_padded = _center_pad(cropped_mask.astype(np.float32), FINAL_SHAPE) > 0.5
    normed = _znorm_safe(img_padded, mask_padded)
    return torch.from_numpy(normed[None])


def _neutral_fallback() -> torch.Tensor:
    return torch.zeros((1, *FINAL_SHAPE), dtype=torch.float32)


def preprocess_task5(t1_path: str, work_dir: str) -> torch.Tensor:
    """Degrades ONCE, from (HD-BET + rigid[/affine] registration onto the
    template box) to (spacing-aware resample, no registration), since that
    fallback still derives its output from the real input data. If it ALSO
    fails, this now propagates the exception instead of silently returning
    a deterministic zero volume (2026-08-30 policy change): a silent
    all-zero output scores badly with no diagnostic signal, while an
    uncaught exception crashes the container and the FOMO26 harness marks
    it INVALID immediately, surfacing the real bug instead of hiding it.
    The only input still tolerated without raising is a genuinely blank/
    degenerate scan, handled separately below via is_blank_image().

    HD-BET and registration are both estimated on a T1 resampled to
    OUTPUT_SPACING (1mm) exactly ONCE -- the same target spacing the
    final crop box already uses, per the 2026-08-22 finding that
    re-spacing only needs to happen once since the trained space IS 1mm.
    Measured 2026-08-22 on a real 6500-万-voxel subject: HD-BET
    30.7s->15.3s, registration 10.6s->3.6s at 1mm vs native. The
    resulting mask is resampled back onto T1's NATIVE grid (cheap,
    nearest-neighbor) so the FINAL data warp (_build_registered) always
    applies the resulting transform directly to the ORIGINAL native T1 in
    a single interpolation onto the crop box -- resampling to 1mm here is
    purely a mask/registration-estimation shortcut, never part of the
    final data path."""
    data = nib.load(t1_path).get_fdata().astype(np.float32)
    if not np.all(np.isfinite(data)) or is_blank_image(data):
        print("[WARNING] blank/degenerate input -- using neutral fallback")
        return _neutral_fallback()

    try:
        t1_native_img = sitk.ReadImage(t1_path, sitk.sitkFloat32)
        t1_proxy_img = _isotropic_resample_whole(t1_native_img, OUTPUT_SPACING, sitk.sitkLinear)
        t1_proxy_path = os.path.join(work_dir, "t1_1mm.nii.gz")
        sitk.WriteImage(t1_proxy_img, t1_proxy_path)

        mask_proxy_path = _run_hdbet(t1_proxy_path, work_dir)
        if mask_proxy_path is None:
            raise RuntimeError("HD-BET failed or produced a zero-mass mask -- "
                                "skipping unmasked registration, going straight to fallback")

        mask_proxy_img = sitk.ReadImage(mask_proxy_path, sitk.sitkUInt8)
        mask_native_img = sitk.Resample(mask_proxy_img, t1_native_img, sitk.Transform(),
                                         sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        mask_native_path = os.path.join(work_dir, "hdbet_mask_native.nii.gz")
        sitk.WriteImage(mask_native_img, mask_native_path)

        return _build_registered(t1_path, mask_native_path, t1_proxy_path, mask_proxy_path, work_dir)
    except Exception as e:
        print(f"[WARNING] HD-BET+registration pipeline failed ({type(e).__name__}: {e}) "
              f"-- falling back to spacing-aware resample without registration")
        return _build_no_registration_fallback(t1_path, work_dir)
