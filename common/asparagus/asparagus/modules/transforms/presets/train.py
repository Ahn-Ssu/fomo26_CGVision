from asparagus.modules.transforms.crop import Torch_Crop
from asparagus.modules.transforms.pad import Torch_Pad
from gardening_tools.functional.transforms.spatial import get_max_rotated_size
from gardening_tools.modules.transforms.bias_field import Torch_BiasField
from gardening_tools.modules.transforms.blur import Torch_Blur
from gardening_tools.modules.transforms.cropping_and_padding import Torch_CenterCrop
from gardening_tools.modules.transforms.deep_supervision import Torch_DownsampleSegForDS
from gardening_tools.modules.transforms.gamma import Torch_Gamma
from gardening_tools.modules.transforms.mirror import Torch_Mirror
from gardening_tools.modules.transforms.motion_ghosting import Torch_MotionGhosting
from gardening_tools.modules.transforms.noise import Torch_AdditiveNoise, Torch_MultiplicativeNoise
from gardening_tools.modules.transforms.normalize import Torch_CT_NormalizeC0, Torch_Normalize
from gardening_tools.modules.transforms.ringing import Torch_GibbsRinging
from gardening_tools.modules.transforms.sampling import Torch_Resize, Torch_SimulateLowres
from gardening_tools.modules.transforms.spatial import Torch_Spatial
from gardening_tools.functional.transforms.gamma import torch_gamma
from monai.transforms import (
    RandHistogramShift,
    RandKSpaceSpikeNoise,
    RandRicianNoise,
    RandScaleIntensity,
)
from torchvision import transforms
import numpy as np
import torch


def none(*args, **kwargs):
    return None


def CPU_seg_train_transforms(patch_size, normalize=True):
    if len(patch_size) == 2:
        axes = (0, 1)
    else:
        axes = (0, 1, 2)
    p_rot_all_channel = 0.2
    p_scale_all_channel = 0.2

    if p_rot_all_channel > 0 or p_scale_all_channel > 0:
        pre_aug_patch_size = get_max_rotated_size(patch_size)
    else:
        pre_aug_patch_size = patch_size

    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_Pad(patch_size=pre_aug_patch_size),
            Torch_Crop(patch_size=pre_aug_patch_size, p_oversample_foreground=0.33),
            Torch_Spatial(
                patch_size=patch_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=p_rot_all_channel,
                p_rot_per_axis=0.3,
                x_rot_in_degrees=(-30.0, 30.0),
                y_rot_in_degrees=(-30.0, 30.0),
                z_rot_in_degrees=(-30.0, 30.0),
                p_scale_all_channel=p_scale_all_channel,
                scale_factor=(0.7, 1.4),
            ),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
        ]
    )


def CPU_clsreg_train_transforms_crop(target_size, normalize=True):
    if len(target_size) == 2:
        axes = (0, 1)
    else:
        axes = (0, 1, 2)
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_Pad(patch_size=target_size),
            Torch_CenterCrop(target_size=target_size),
            Torch_Spatial(
                patch_size=target_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=0.2,
                p_rot_per_axis=0.3,
                p_scale_all_channel=0.2,
                x_rot_in_degrees=(-30.0, 30.0),
                y_rot_in_degrees=(-30.0, 30.0),
                z_rot_in_degrees=(-30.0, 30.0),
                scale_factor=(0.7, 1.4),
                crop=False,
                clip_to_input_range=False,
                skip_label=True,
            ),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
        ]
    )


def CPU_clsreg_train_transforms_mirror_only(target_size, normalize=True):
    """Classification/regression control arm: geometry-preserving mirror only."""
    axes = (0, 1) if len(target_size) == 2 else (0, 1, 2)
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_Pad(patch_size=target_size),
            Torch_CenterCrop(target_size=target_size),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
        ]
    )


class Torch_FixedModalityGamma:
    """Deterministic (non-random) per-channel gamma correction, applied to
    EVERY sample -- train AND val/test alike, unlike every other transform
    in this file. 2026-08-25, per user request: bakes each Task1 modality's
    own known lesion-vs-normal intensity-contrast direction into the data
    as a fixed preprocessing step rather than a stochastic augmentation.

    Direction/magnitude source: a 13-subject pooled lesion-vs-normal-tissue
    intensity probe (raw-space, HD-BET brain mask, Cohen's d) found
    FLAIR +0.34 (lesion brighter), ADC -0.845 (lesion darker, by far the
    largest effect -- matches ADC being the most specific clinical sign of
    acute infarct), DWI +0.44 (lesion brighter), 4th/SWI-T2S -0.376 (lesion
    darker). gamma expands contrast on whichever side of the normalized
    [0,1] range is FAR from 1 -- concretely: gamma>1 expands bright-region
    contrast (and crushes dark-region contrast together); gamma<1 expands
    dark-region contrast (and crushes bright-region contrast). So the
    direction that AMPLIFIES each modality's own lesion signal is gamma>1
    for a bright lesion (FLAIR, DWI) and gamma<1 for a dark lesion (ADC,
    4th) -- verified via gamma_lesion_targeted.py: applying this exact
    "enhance_mild" combination (1.33/0.67) at TEST time only, to an
    already-trained hdbet_rigid classifier that never saw it in training,
    raised pooled val AUROC 0.8352->0.8901 (+0.055); the opposite
    ("washout", 0.67/1.33) dropped it to 0.7582 (-0.077). This preset
    bakes the same enhance-direction transform into the data pipeline
    itself (after Torch_Normalize, matching the normalized-tensor space
    that probe was run in) so the model sees it during training too, not
    just as a one-off test-time perturbation.

    Channel order must be MODALITY_ORDER = [flair, adc, dwi, fourth]."""

    GAMMAS = (1.33, 0.67, 1.33, 0.67)  # flair, adc, dwi, fourth
    EPS = 1e-7

    def __init__(self, data_key="image"):
        self.data_key = data_key

    def __call__(self, data_dict):
        image = data_dict[self.data_key]
        for c in range(min(image.shape[0], len(self.GAMMAS))):
            ch = image[c]
            img_min = ch.min()
            img_max = ch.max()
            img_range = img_max - img_min
            image[c] = (
                torch.pow((ch - img_min) / (img_range + self.EPS), self.GAMMAS[c])
                * (img_range + self.EPS)
                + img_min
            )
        data_dict[self.data_key] = image
        return data_dict


def CPU_clsreg_train_transforms_mild_spatial_lesion_gamma(target_size, normalize=True):
    """CPU_clsreg_train_transforms_mild_spatial with Torch_FixedModalityGamma
    inserted right after normalization -- see that class's docstring."""
    axes = (0, 1) if len(target_size) == 2 else (0, 1, 2)
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_FixedModalityGamma(),
            Torch_Pad(patch_size=target_size),
            Torch_CenterCrop(target_size=target_size),
            Torch_Spatial(
                patch_size=target_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=0.2,
                p_rot_per_axis=0.3,
                p_scale_all_channel=0.2,
                x_rot_in_degrees=(-15.0, 15.0),
                y_rot_in_degrees=(-15.0, 15.0),
                z_rot_in_degrees=(-15.0, 15.0),
                scale_factor=(0.9, 1.1),
                crop=False,
                clip_to_input_range=False,
                skip_label=True,
            ),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
        ]
    )


def CPU_clsreg_val_test_transforms_crop_lesion_gamma(target_size, normalize=True):
    """CPU_clsreg_val_test_transforms_crop with the same fixed
    Torch_FixedModalityGamma step -- must match the train preset's
    preprocessing exactly, so this pairs with
    CPU_clsreg_train_transforms_mild_spatial_lesion_gamma via
    transforms.cpu_val_transforms=."""
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_FixedModalityGamma(),
            Torch_Pad(patch_size=target_size),
            Torch_CenterCrop(target_size=target_size),
        ]
    )


# Short aliases -- the Hydra clargs-embedded output directory naming scheme
# is only safe from ext4's 255-byte single-component NAME_MAX because the
# embedded absolute checkpoint_path/test_split_path/train_split_path values
# contain real '/' characters that split the string into short segments;
# the segment AFTER the last such '/' (transforms.* onward) has no more
# slashes to help it, so a long preset name here can push that segment over
# 255 bytes and make mkdir fail with "File name too long" (hit 2026-08-25
# using the full descriptive names below -- 262 bytes, just over the
# limit). Use these short names in any clargs string instead.
CPU_lesion_gamma_train = CPU_clsreg_train_transforms_mild_spatial_lesion_gamma
CPU_lesion_gamma_val = CPU_clsreg_val_test_transforms_crop_lesion_gamma


def CPU_clsreg_train_transforms_mild_spatial(target_size, normalize=True):
    """Classification/regression spatial arm: mirror + mild rotate/scale only."""
    axes = (0, 1) if len(target_size) == 2 else (0, 1, 2)
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            # Keep the classifier's model-input geometry fixed even on the
            # 80% of samples where neither spatial operation is drawn.
            Torch_Pad(patch_size=target_size),
            Torch_CenterCrop(target_size=target_size),
            Torch_Spatial(
                patch_size=target_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=0.2,
                p_rot_per_axis=0.3,
                p_scale_all_channel=0.2,
                x_rot_in_degrees=(-15.0, 15.0),
                y_rot_in_degrees=(-15.0, 15.0),
                z_rot_in_degrees=(-15.0, 15.0),
                scale_factor=(0.9, 1.1),
                crop=False,
                clip_to_input_range=False,
                skip_label=True,
            ),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
        ]
    )


class Torch_MonaiIntensity:
    """Wraps a MONAI intensity transform (per-sample, no batch dim, same
    (C,X,Y,Z) convention as this project's own transforms) into the
    dict-based __call__(data_dict) convention every other transform here
    uses. Casts the result back to a plain torch.Tensor (.as_tensor()) so a
    MONAI MetaTensor never leaks into downstream gardening_tools transforms
    or collation, which don't know about MONAI's tensor subclass."""

    def __init__(self, monai_transform, data_key="image"):
        self.monai_transform = monai_transform
        self.data_key = data_key

    def __call__(self, data_dict):
        image = data_dict[self.data_key]
        out = self.monai_transform(image)
        data_dict[self.data_key] = out.as_tensor() if hasattr(out, "as_tensor") else out
        return data_dict


class Torch_LVRandomCutout:
    """Randomly occludes 1-4 random-sized boxes within a FIXED lateral-
    ventricle bounding box (in canonical, pre-Torch_Spatial array-index
    space) with locally-toned noise (2026-08-29, Task5 PMG generalization
    probe). Motivation: LR-asymmetry + saliency analysis on Task5 showed
    the classifier's attention concentrates on ventricle-boundary voxels
    rather than the cortex, where the actual PMG pathology (gray-matter
    folding) lives -- real leaderboard AUROC for the affected variants was
    near/below random despite locally-decent held-out AUROC, consistent
    with the model latching onto ventricle shape/size as a shortcut that
    doesn't generalize. This transform denies that shortcut by randomly
    corrupting sub-regions of the ventricle bbox every epoch, without
    touching the surrounding cortex. Applied BEFORE Torch_Spatial so the
    fixed bbox coordinates -- valid in this dataset variant's shared
    registration template space -- still line up with the actual
    ventricles; the (now-noised) region is then carried along by whatever
    rotation/mirror follows, same as everything else in the volume.

    bbox is (x0,x1,y0,y1,z0,z1), inclusive, derived per-variant from a
    population CSF-consistency ventricle mask (eroded to drop the thin
    3rd-ventricle/interhemispheric-fissure sliver, then re-expanded).

    Box size is sampled as a FRACTION of each axis's bbox extent
    (size_frac_range), not an absolute voxel count: volume scales with the
    cube of linear size, so a literal "min size = 1/3 of bbox volume"
    target would force every single box to already span ~70% of each axis
    -- at that point 2+ overlapping boxes just reproduce Torch_LVFixedMask
    every draw, with no real between-epoch variability left. Sampling the
    LINEAR fraction per axis instead (default 0.4-0.9) keeps individual
    boxes substantial (min ~6% of bbox volume, max ~73%) while 1-3 of them
    landing at random offsets still yields a different, non-trivial (very
    roughly 30-70%) total-coverage pattern on every call."""

    def __init__(self, bbox, n_cutouts_range=(1, 3), size_frac_range=(0.4, 0.9), p=0.8, data_key="image"):
        self.bbox = bbox
        self.n_range = n_cutouts_range
        self.size_frac_range = size_frac_range
        self.p = p
        self.data_key = data_key

    def __call__(self, data_dict):
        if np.random.random() > self.p:
            return data_dict
        img = data_dict[self.data_key].clone()
        x0, x1, y0, y1, z0, z1 = self.bbox
        ext_x, ext_y, ext_z = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
        vlo, vhi = img.min(), img.max()
        n = np.random.randint(self.n_range[0], self.n_range[1] + 1)
        for _ in range(n):
            fx, fy, fz = np.random.uniform(*self.size_frac_range, size=3)
            sx = max(1, min(round(fx * ext_x), ext_x))
            sy = max(1, min(round(fy * ext_y), ext_y))
            sz = max(1, min(round(fz * ext_z), ext_z))
            cx = np.random.randint(x0, x1 - sx + 2)
            cy = np.random.randint(y0, y1 - sy + 2)
            cz = np.random.randint(z0, z1 - sz + 2)
            patch = img[..., cx:cx + sx, cy:cy + sy, cz:cz + sz]
            local_mean = patch.mean()
            local_std = patch.std().clamp_min(1e-3)
            noise = (torch.randn_like(patch) * local_std + local_mean).clamp(vlo, vhi)
            img[..., cx:cx + sx, cy:cy + sy, cz:cz + sz] = noise
        data_dict[self.data_key] = img
        return data_dict


class Torch_LVFixedMask:
    """Always corrupts the ENTIRE fixed LV+periventricular bbox with fresh
    locally-toned noise, every sample, every epoch (2026-08-29, companion
    to Torch_LVRandomCutout above -- same motivation, but deterministic
    coverage instead of a random sub-region each draw, to test whether
    fully and permanently denying the ventricle region forces the
    classifier onto the cortex). Same bbox convention and pre-Torch_Spatial
    placement as Torch_LVRandomCutout."""

    def __init__(self, bbox, data_key="image"):
        self.bbox = bbox
        self.data_key = data_key

    def __call__(self, data_dict):
        img = data_dict[self.data_key].clone()
        x0, x1, y0, y1, z0, z1 = self.bbox
        vlo, vhi = img.min(), img.max()
        patch = img[..., x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
        local_mean = patch.mean()
        local_std = patch.std().clamp_min(1e-3)
        noise = (torch.randn_like(patch) * local_std + local_mean).clamp(vlo, vhi)
        img[..., x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = noise
        data_dict[self.data_key] = img
        return data_dict


class Torch_LVFixedMaskGPU:
    """GPU-stage (batched, post-on_after_batch_transfer) counterpart to
    Torch_LVFixedMask (2026-08-30, per user request). The CPU-stage version
    sits before Torch_Spatial/Mirror/MONAI-intensity in the CPU pipeline,
    but Gamma/BiasField/MotionGhosting/GibbsRinging/etc. all run AFTER it
    (GPU_all_train_transforms style presets execute in on_after_batch_
    transfer, a separate later stage -- see base_module.py's
    on_after_batch_transfer, which applies cpu_tr_transforms per-sample in
    the Dataset and gpu_tr_transforms batched afterward). That means the
    CPU version's carefully local-mean/std-matched noise block gets
    reshuffled by every later intensity augmentation, especially under
    boosted Gamma (measured: masked-test AUROC dropped from 0.8681 to
    0.7917 when Gamma was boosted p=0.15->0.4, with no epoch-count
    explanation -- the gap WIDENED, not narrowed, from 5 to 10 epochs).
    Putting the mask here instead -- appended as the LAST step of a
    GPU_*_train_transforms preset -- means it always matches this batch's
    fully-augmented final state, matching what "denying the ventricle"
    should actually mean: hiding it in the image the model actually sees,
    not in some intermediate pre-augmentation version of it.

    Operates on a BATCHED (N, C, X, Y, Z) tensor -- loops over the batch
    dim since a batch of per-sample torch.randn_like noise draws still
    needs each sample's own local mean/std, and batch sizes here are small
    (2) so the loop is cheap."""

    def __init__(self, bbox, data_key="image"):
        self.bbox = bbox
        self.data_key = data_key

    def __call__(self, data_dict):
        img = data_dict[self.data_key].clone()
        x0, x1, y0, y1, z0, z1 = self.bbox
        for i in range(img.shape[0]):
            sample = img[i]
            vlo, vhi = sample.min(), sample.max()
            patch = sample[..., x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
            local_mean = patch.mean()
            local_std = patch.std().clamp_min(1e-3)
            noise = (torch.randn_like(patch) * local_std + local_mean).clamp(vlo, vhi)
            sample[..., x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = noise
        data_dict[self.data_key] = img
        return data_dict


class Torch_LVRandomCutoutGPU:
    """GPU-stage (batched) counterpart to Torch_LVRandomCutout -- same
    rationale as Torch_LVFixedMaskGPU above, and same per-sample-loop
    approach for independent random count/size/position draws per batch
    element."""

    def __init__(self, bbox, n_cutouts_range=(1, 3), size_frac_range=(0.4, 0.9), p=0.8, data_key="image"):
        self.bbox = bbox
        self.n_range = n_cutouts_range
        self.size_frac_range = size_frac_range
        self.p = p
        self.data_key = data_key

    def __call__(self, data_dict):
        img = data_dict[self.data_key].clone()
        x0, x1, y0, y1, z0, z1 = self.bbox
        ext_x, ext_y, ext_z = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
        for i in range(img.shape[0]):
            if np.random.random() > self.p:
                continue
            sample = img[i]
            vlo, vhi = sample.min(), sample.max()
            n = np.random.randint(self.n_range[0], self.n_range[1] + 1)
            for _ in range(n):
                fx, fy, fz = np.random.uniform(*self.size_frac_range, size=3)
                sx = max(1, min(round(fx * ext_x), ext_x))
                sy = max(1, min(round(fy * ext_y), ext_y))
                sz = max(1, min(round(fz * ext_z), ext_z))
                cx = np.random.randint(x0, x1 - sx + 2)
                cy = np.random.randint(y0, y1 - sy + 2)
                cz = np.random.randint(z0, z1 - sz + 2)
                patch = sample[..., cx:cx + sx, cy:cy + sy, cz:cz + sz]
                local_mean = patch.mean()
                local_std = patch.std().clamp_min(1e-3)
                noise = (torch.randn_like(patch) * local_std + local_mean).clamp(vlo, vhi)
                sample[..., cx:cx + sx, cy:cy + sy, cz:cz + sz] = noise
        data_dict[self.data_key] = img
        return data_dict


# Fixed LV+periventricular bboxes (x0,x1,y0,y1,z0,z1), derived once from a
# 48-subject population CSF-consistency ventricle mask per dataset variant
# (eroded to isolate the ventricle body from the thin 3rd-ventricle/
# interhemispheric-fissure sliver, then re-expanded with a 15-voxel margin
# to also cover the immediately surrounding periventricular tissue --
# see /root/fomo26task5/lr_asymmetry_analysis/{variant}_lv_augment_bbox.npy).
# Only rigid/affine have a shared registration template these fixed
# coordinates are valid in; hdbet_only has no common space.
_LV_BBOX_RIGID = (52, 141, 61, 165, 60, 134)
_LV_BBOX_AFFINE = (49, 144, 55, 172, 57, 134)

# Mask-size ablation for affine (2026-08-30, per user request after
# affine_lv_fixedmask + matched-masked-test evaluation surprisingly beat the
# unmasked baseline, 0.8681 vs 0.8299 -- want to know how much of the bbox's
# generous 15-voxel margin is actually necessary). Same core (eroded,
# largest-CC ventricle body) as _LV_BBOX_AFFINE, just re-expanded by a
# smaller margin: 0 voxels (tight, ventricle body only, 3.4% of volume) and
# 8 voxels (the original visualization-cube margin, 6.6%), vs the existing
# 15-voxel-margin box's 10.7%.
_LV_BBOX_AFFINE_SMALL = (64, 129, 70, 157, 72, 119)   # margin=0, 3.38% of volume
_LV_BBOX_AFFINE_MEDIUM = (56, 137, 62, 165, 64, 127)  # margin=8, 6.61% of volume


def CPU_clsreg_train_transforms_mild_spatial_intensity(target_size, normalize=True):
    """mild_spatial + boosted scale augmentation + 4 MONAI intensity augs
    (2026-08-19, Task5 invariance-probe follow-up): a synthetic +-10% resize
    on an already-trained mild_spatial classifier shifted its predicted
    probability by several points in a consistent direction on EVERY
    dataset variant tested, including the fully affine-registered one --
    p_scale_all_channel 0.2->0.5 and scale_factor (0.9,1.1)->(0.85,1.15) so
    the model sees scale variation on more samples and over a wider range at
    train time. Intensity augmentations added per user request (contrast/
    noise-robustness, discourage any other single easy shortcut cue) --
    applied after the spatial arm, matching this file's other conventions
    (GPU intensity presets are similarly a separate trailing group).

    No existing gardening_tools equivalent for these 4 (checked: it has
    gamma/blur/bias-field/ghosting/ringing/lowres/gaussian-noise, not
    Rician/k-space-spike/histogram-shift/scale-intensity) -- MONAI mixed
    into this torchvision Compose is an established pattern elsewhere in
    this repo (transforms/DinoV2.py already does the same thing).
    """
    axes = (0, 1) if len(target_size) == 2 else (0, 1, 2)
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_Pad(patch_size=target_size),
            Torch_CenterCrop(target_size=target_size),
            Torch_Spatial(
                patch_size=target_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=0.2,
                p_rot_per_axis=0.3,
                p_scale_all_channel=0.5,
                x_rot_in_degrees=(-15.0, 15.0),
                y_rot_in_degrees=(-15.0, 15.0),
                z_rot_in_degrees=(-15.0, 15.0),
                scale_factor=(0.85, 1.15),
                crop=False,
                clip_to_input_range=False,
                skip_label=True,
            ),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
            Torch_MonaiIntensity(RandHistogramShift(num_control_points=(5, 15), prob=0.15)),
            Torch_MonaiIntensity(RandScaleIntensity(factors=0.1, prob=0.15)),
            Torch_MonaiIntensity(RandRicianNoise(prob=0.15, std=0.05, relative=True, sample_std=True)),
            Torch_MonaiIntensity(RandKSpaceSpikeNoise(prob=0.1)),
        ]
    )


def _clsreg_train_transforms_mild_spatial_intensity_plus(target_size, normalize, lv_transform):
    """Shared body for the 4 LV-augmented presets below: identical to
    CPU_clsreg_train_transforms_mild_spatial_intensity, with lv_transform
    spliced in right after Torch_CenterCrop and before Torch_Spatial so its
    fixed bbox coordinates -- valid in the pre-rotation/pre-mirror grid --
    still line up with the actual ventricles."""
    axes = (0, 1) if len(target_size) == 2 else (0, 1, 2)
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_Pad(patch_size=target_size),
            Torch_CenterCrop(target_size=target_size),
            lv_transform,
            Torch_Spatial(
                patch_size=target_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=0.2,
                p_rot_per_axis=0.3,
                p_scale_all_channel=0.5,
                x_rot_in_degrees=(-15.0, 15.0),
                y_rot_in_degrees=(-15.0, 15.0),
                z_rot_in_degrees=(-15.0, 15.0),
                scale_factor=(0.85, 1.15),
                crop=False,
                clip_to_input_range=False,
                skip_label=True,
            ),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
            Torch_MonaiIntensity(RandHistogramShift(num_control_points=(5, 15), prob=0.15)),
            Torch_MonaiIntensity(RandScaleIntensity(factors=0.1, prob=0.15)),
            Torch_MonaiIntensity(RandRicianNoise(prob=0.15, std=0.05, relative=True, sample_std=True)),
            Torch_MonaiIntensity(RandKSpaceSpikeNoise(prob=0.1)),
        ]
    )


def CPU_clsreg_train_transforms_lv_cutout_rigid(target_size, normalize=True):
    """mild_spatial_intensity + random-count/random-size cutout boxes
    confined to the rigid variant's fixed LV+periventricular bbox (2026-08-29,
    see Torch_LVRandomCutout docstring)."""
    return _clsreg_train_transforms_mild_spatial_intensity_plus(
        target_size, normalize, Torch_LVRandomCutout(_LV_BBOX_RIGID))


def CPU_clsreg_train_transforms_lv_cutout_affine(target_size, normalize=True):
    """Same as CPU_clsreg_train_transforms_lv_cutout_rigid but for the affine
    variant's (larger) fixed LV+periventricular bbox."""
    return _clsreg_train_transforms_mild_spatial_intensity_plus(
        target_size, normalize, Torch_LVRandomCutout(_LV_BBOX_AFFINE))


def CPU_clsreg_train_transforms_lv_fixedmask_rigid(target_size, normalize=True):
    """mild_spatial_intensity + the ENTIRE rigid variant's fixed
    LV+periventricular bbox always corrupted with fresh noise, every sample
    (2026-08-29, see Torch_LVFixedMask docstring)."""
    return _clsreg_train_transforms_mild_spatial_intensity_plus(
        target_size, normalize, Torch_LVFixedMask(_LV_BBOX_RIGID))


def CPU_clsreg_train_transforms_lv_fixedmask_affine(target_size, normalize=True):
    """Same as CPU_clsreg_train_transforms_lv_fixedmask_rigid but for the
    affine variant's (larger) fixed LV+periventricular bbox."""
    return _clsreg_train_transforms_mild_spatial_intensity_plus(
        target_size, normalize, Torch_LVFixedMask(_LV_BBOX_AFFINE))


def CPU_clsreg_train_transforms_lv_fixedmask_affine_medium(target_size, normalize=True):
    """Mask-size ablation: same as CPU_clsreg_train_transforms_lv_fixedmask_affine
    but with the smaller _LV_BBOX_AFFINE_MEDIUM bbox (6.6% of volume vs 10.7%)."""
    return _clsreg_train_transforms_mild_spatial_intensity_plus(
        target_size, normalize, Torch_LVFixedMask(_LV_BBOX_AFFINE_MEDIUM))


def CPU_clsreg_train_transforms_lv_fixedmask_affine_small(target_size, normalize=True):
    """Mask-size ablation: same as CPU_clsreg_train_transforms_lv_fixedmask_affine
    but with the tightest _LV_BBOX_AFFINE_SMALL bbox (3.4% of volume, no
    margin beyond the eroded ventricle body itself)."""
    return _clsreg_train_transforms_mild_spatial_intensity_plus(
        target_size, normalize, Torch_LVFixedMask(_LV_BBOX_AFFINE_SMALL))


def CPU_CT_C0_clsreg_train_transforms_crop(target_size, normalize=True):
    if len(target_size) == 2:
        axes = (0, 1)
    else:
        axes = (0, 1, 2)
    return transforms.Compose(
        [
            Torch_CT_NormalizeC0(normalize=normalize),
            Torch_Pad(patch_size=target_size),
            Torch_CenterCrop(target_size=target_size),
            Torch_Spatial(
                patch_size=target_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=0.2,
                p_rot_per_axis=0.3,
                p_scale_all_channel=0.2,
                x_rot_in_degrees=(-30.0, 30.0),
                y_rot_in_degrees=(-30.0, 30.0),
                z_rot_in_degrees=(-30.0, 30.0),
                scale_factor=(0.7, 1.4),
                crop=False,
                clip_to_input_range=False,
                skip_label=True,
            ),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
        ]
    )


def CPU_clsreg_train_transforms_resize(target_size, normalize=True):
    if len(target_size) == 2:
        axes = (0, 1)
    else:
        axes = (0, 1, 2)
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_Resize(target_size=target_size),
            Torch_Spatial(
                patch_size=target_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=0.2,
                p_rot_per_axis=0.3,
                p_scale_all_channel=0.2,
                x_rot_in_degrees=(-30.0, 30.0),
                y_rot_in_degrees=(-30.0, 30.0),
                z_rot_in_degrees=(-30.0, 30.0),
                scale_factor=(0.7, 1.4),
                crop=False,
                clip_to_input_range=False,
                skip_label=True,
            ),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
        ]
    )


def CPU_clsreg_val_test_transforms_crop(target_size, normalize=True):
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_Pad(patch_size=target_size),
            Torch_CenterCrop(target_size=target_size),
        ]
    )


def CPU_CT_C0_clsreg_val_test_transforms_crop(target_size, normalize=True):
    return transforms.Compose(
        [
            Torch_CT_NormalizeC0(normalize=normalize),
            Torch_Pad(patch_size=target_size),
            Torch_CenterCrop(target_size=target_size),
        ]
    )


def CPU_seg_val_transforms(patch_size, normalize=True):
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_Pad(patch_size=patch_size),
            Torch_Crop(patch_size=patch_size, p_oversample_foreground=0.0),
        ]
    )


def CPU_seg_test_transforms(patch_size, normalize=True):
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_Pad(patch_size=patch_size),
        ]
    )


def GPU_all_train_transforms(ndim=3, deep_supervision=False):
    axes = (0, ndim)
    tforms = transforms.Compose(
        [
            Torch_Blur(p_per_channel=0.15),
            Torch_BiasField(p_per_channel=0.2),
            Torch_Gamma(p_all_channel=0.15),
            Torch_MotionGhosting(p_per_channel=0.1, axes=axes),
            Torch_GibbsRinging(p_per_channel=0.1, axes=axes),
            Torch_SimulateLowres(p_per_channel=0.5, p_per_axis=0.25),
            Torch_MultiplicativeNoise(p_per_channel=0.1),
            Torch_AdditiveNoise(p_per_channel=0.1),
        ]
    )

    if deep_supervision:
        tforms.transforms.append(Torch_DownsampleSegForDS(deep_supervision=True))

    return tforms


def GPU_boosted_gamma_biasfield_train_transforms(ndim=3, deep_supervision=False):
    """GPU_all_train_transforms with ONLY Gamma and BiasField boosted --
    2026-08-24, Task1 D robustness-probe follow-up: forcing Gamma/BiasField
    to p=1 at inference time on an already-trained hdbet_rigid classifier
    dropped pooled AUROC by up to -0.088/-0.077 (biggest drop of all 8
    GPU_all_train_transforms ingredients, and much larger than the same
    perturbation on the un-registered iso1mm candidate A, which showed
    ~0.000/-0.066) -- suggests the registered-input model leans on an
    intensity/contrast cue the un-registered one doesn't, and is a
    candidate explanation for D/F's local-CV-vs-real-leaderboard gap.
    Every other ingredient (Blur/MotionGhosting/GibbsRinging/SimulateLowres/
    MultiplicativeNoise/AdditiveNoise) is left at GPU_all_train_transforms'
    original probability -- only Gamma (0.15->0.4) and BiasField (0.2->0.5)
    are boosted, same ~2.5x jump this project already used once before for
    CPU_clsreg_train_transforms_mild_spatial_intensity's own scale boost
    (p_scale_all_channel 0.2->0.5)."""
    axes = (0, ndim)
    tforms = transforms.Compose(
        [
            Torch_Blur(p_per_channel=0.15),
            Torch_BiasField(p_per_channel=0.5),
            Torch_Gamma(p_all_channel=0.4),
            Torch_MotionGhosting(p_per_channel=0.1, axes=axes),
            Torch_GibbsRinging(p_per_channel=0.1, axes=axes),
            Torch_SimulateLowres(p_per_channel=0.5, p_per_axis=0.25),
            Torch_MultiplicativeNoise(p_per_channel=0.1),
            Torch_AdditiveNoise(p_per_channel=0.1),
        ]
    )

    if deep_supervision:
        tforms.transforms.append(Torch_DownsampleSegForDS(deep_supervision=True))

    return tforms


def GPU_boosted_ringing_ghosting_train_transforms(ndim=3, deep_supervision=False):
    """GPU_all_train_transforms with ONLY GibbsRinging and MotionGhosting
    boosted -- 2026-08-28, Task5 robustness-probe follow-up: a seeded,
    K=5-repeat-averaged 21-condition sensitivity probe on hdbet_only_iso07
    (real leaderboard AUROC 0.515, near-random) found GibbsRinging
    (delta -0.042) and MotionGhosting (delta -0.037) as the largest
    non-invert vulnerabilities, consistent across hdbet_only/rigid/affine
    (rigid's GibbsRinging delta reached -0.057, the single largest
    non-invert effect found in either Task1 or Task5's probes). PMG is a
    gray-matter cortical-folding malformation -- a fine-structural-detail
    diagnosis, not primarily an intensity-based one (unlike Task1's
    infarct) -- and GibbsRinging/MotionGhosting specifically degrade fine
    spatial detail/edges rather than bulk intensity, making them a
    plausible link to the real-vs-local gap. Every other ingredient is
    left at GPU_all_train_transforms' original probability -- only
    GibbsRinging and MotionGhosting (0.1->0.4 each, same ~4x jump as this
    project's other boosted-augmentation presets) are increased."""
    axes = (0, ndim)
    tforms = transforms.Compose(
        [
            Torch_Blur(p_per_channel=0.15),
            Torch_BiasField(p_per_channel=0.2),
            Torch_Gamma(p_all_channel=0.15),
            Torch_MotionGhosting(p_per_channel=0.4, axes=axes),
            Torch_GibbsRinging(p_per_channel=0.4, axes=axes),
            Torch_SimulateLowres(p_per_channel=0.5, p_per_axis=0.25),
            Torch_MultiplicativeNoise(p_per_channel=0.1),
            Torch_AdditiveNoise(p_per_channel=0.1),
        ]
    )

    if deep_supervision:
        tforms.transforms.append(Torch_DownsampleSegForDS(deep_supervision=True))

    return tforms


def _gpu_all_train_transforms_plus_lv(ndim, deep_supervision, lv_transform):
    """Shared body for the 4 GPU-stage LV-masking presets below:
    GPU_all_train_transforms with lv_transform appended LAST, so the
    ventricle-denial always matches the fully-augmented final image
    instead of getting reshuffled by Gamma/BiasField/etc. that would
    otherwise run after it (see Torch_LVFixedMaskGPU's docstring)."""
    axes = (0, ndim)
    tforms = transforms.Compose(
        [
            Torch_Blur(p_per_channel=0.15),
            Torch_BiasField(p_per_channel=0.2),
            Torch_Gamma(p_all_channel=0.15),
            Torch_MotionGhosting(p_per_channel=0.1, axes=axes),
            Torch_GibbsRinging(p_per_channel=0.1, axes=axes),
            Torch_SimulateLowres(p_per_channel=0.5, p_per_axis=0.25),
            Torch_MultiplicativeNoise(p_per_channel=0.1),
            Torch_AdditiveNoise(p_per_channel=0.1),
            lv_transform,
        ]
    )
    if deep_supervision:
        tforms.transforms.append(Torch_DownsampleSegForDS(deep_supervision=True))
    return tforms


def GPU_lv_fixedmask_affine_train_transforms(ndim=3, deep_supervision=False):
    """GPU_all_train_transforms + Torch_LVFixedMaskGPU(_LV_BBOX_AFFINE)
    appended last (2026-08-30). Pair with
    CPU_clsreg_train_transforms_mild_spatial_intensity (NOT the CPU-stage
    lv_fixedmask preset -- masking now lives here instead)."""
    return _gpu_all_train_transforms_plus_lv(ndim, deep_supervision, Torch_LVFixedMaskGPU(_LV_BBOX_AFFINE))


def GPU_lv_fixedmask_rigid_train_transforms(ndim=3, deep_supervision=False):
    """Same as GPU_lv_fixedmask_affine_train_transforms but for rigid's bbox."""
    return _gpu_all_train_transforms_plus_lv(ndim, deep_supervision, Torch_LVFixedMaskGPU(_LV_BBOX_RIGID))


def GPU_lv_cutout_affine_train_transforms(ndim=3, deep_supervision=False):
    """Same as GPU_lv_fixedmask_affine_train_transforms but with the random
    (not always-full) Torch_LVRandomCutoutGPU, matching the CPU-stage
    lv_cutout preset's default parameters."""
    return _gpu_all_train_transforms_plus_lv(ndim, deep_supervision, Torch_LVRandomCutoutGPU(_LV_BBOX_AFFINE))


def GPU_lv_cutout_rigid_train_transforms(ndim=3, deep_supervision=False):
    """Same as GPU_lv_cutout_affine_train_transforms but for rigid's bbox."""
    return _gpu_all_train_transforms_plus_lv(ndim, deep_supervision, Torch_LVRandomCutoutGPU(_LV_BBOX_RIGID))


class Torch_GammaPerChannel:
    """Gamma with an independent per-channel application probability."""

    def __init__(self, p_per_channel=0.05, gamma_range=(0.9, 1.1), data_key="image"):
        self.p_per_channel = p_per_channel
        self.gamma_range = gamma_range
        self.data_key = data_key

    def __call__(self, data_dict):
        image = data_dict[self.data_key]
        for batch_idx in range(image.shape[0]):
            for channel_idx in range(image.shape[1]):
                if np.random.uniform() < self.p_per_channel:
                    image[batch_idx, channel_idx] = torch_gamma(
                        image[batch_idx, channel_idx],
                        gamma_range=self.gamma_range,
                        per_channel=False,
                        clip_to_input_range=False,
                    )
        return data_dict


def GPU_mild_intensity_train_transforms(ndim=3, deep_supervision=False):
    """Mild intensity policy for Task-1: no transform exceeds its specified rate."""
    axes = (0, ndim)
    tforms = transforms.Compose(
        [
            Torch_Blur(p_per_channel=0.05),
            Torch_BiasField(p_per_channel=0.05),
            Torch_GammaPerChannel(p_per_channel=0.05, gamma_range=(0.9, 1.1)),
            Torch_MotionGhosting(p_per_channel=0.05, axes=axes),
            Torch_GibbsRinging(p_per_channel=0.05, axes=axes),
            Torch_SimulateLowres(
                p_per_channel=0.2,
                p_per_axis=0.25,
                zoom_range=(0.75, 1.0),
            ),
            Torch_MultiplicativeNoise(p_per_channel=0.05),
            Torch_AdditiveNoise(p_per_channel=0.05),
        ]
    )

    if deep_supervision:
        tforms.transforms.append(Torch_DownsampleSegForDS(deep_supervision=True))
    return tforms
