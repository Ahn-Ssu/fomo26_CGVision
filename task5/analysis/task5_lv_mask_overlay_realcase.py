"""Overlays the fixed LV+periventricular masking bbox (_LV_BBOX_AFFINE, see
common/asparagus/asparagus/modules/transforms/presets/train.py) on a single
real PMG case, preprocessed with the SAME steps as the actual affine
submission container (submission/preprocess.py): HD-BET skull-strip ->
rigid+affine registration to MNI152NLin2009cAsym -> warp onto the fixed
physical crop box -> center-pad to the training grid (192, 224, 192). Unlike
task5_lv_mask_illustration.png (an MNI-template stand-in with the bbox only
proportionally approximated), this reproduces the EXACT training-time grid,
so the bbox coordinates are used verbatim, no rescaling.

Why this is a separate, standalone script rather than reusing
submission/preprocess.py directly: preprocess.py uses antspyx (`import
ants`) for registration; this analysis was run in an environment that only
had the ANTs CLI binaries (antsRegistrationSyNQuick.sh / antsApplyTransforms,
same tool, different interface) plus SimpleITK, HD-BET, and templateflow (to
fetch the exact MNI152NLin2009cAsym template/mask this project's registration
work has always targeted) installed. The registration math and physical
crop-box definition are otherwise identical to preprocess.py's
`_build_registered()`.

Note this pipeline here skips preprocess.py's mask-centroid-aware initial
transform and Dice-gated rigid/affine fallback (a single combined
rigid->affine `antsRegistrationSyNQuick.sh -t a` call is used instead) --
fine for a one-off illustrative overlay, but NOT a byte-for-byte
reproduction of what a training/submission run would do for this subject.

Requires: HD-BET, ANTs CLI (antsRegistrationSyNQuick.sh, antsApplyTransforms)
on PATH, `pip install templateflow`, SimpleITK, nibabel, matplotlib.

Usage: python task5_lv_mask_overlay_realcase.py <raw_t1.nii.gz> <work_dir> <out_png>
"""
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk

# Exact bbox + physical crop box + final grid this project's affine Task5
# recipe actually trains/infers on -- see common/asparagus/asparagus/
# modules/transforms/presets/train.py (_LV_BBOX_AFFINE) and
# submission/preprocess.py (CROP_PHYS_MIN/MAX, REF_GRID_SHAPE, FINAL_SHAPE).
LV_BBOX_AFFINE = (49, 144, 55, 172, 57, 134)  # x0,x1,y0,y1,z0,z1
CROP_PHYS_MIN = np.array([-82.0, -83.0, -82.0])
CROP_PHYS_MAX = np.array([82.0, 118.0, 92.0])
FINAL_SHAPE = (192, 224, 192)
SPACING = 1.0


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def center_pad(arr, target_shape):
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


def fetch_template():
    import templateflow.api as tflow
    t1 = tflow.get("MNI152NLin2009cAsym", resolution=1, desc="brain", suffix="T1w")
    return str(t1)


def build_reference_grid(template_path, out_path):
    template_img = sitk.ReadImage(template_path)
    direction = template_img.GetDirection()
    d = np.array(direction).reshape(3, 3)
    diag = np.diag(d)
    size = [max(1, int(round((CROP_PHYS_MAX[i] - CROP_PHYS_MIN[i]) / SPACING))) for i in range(3)]
    origin = [CROP_PHYS_MAX[i] if diag[i] < 0 else CROP_PHYS_MIN[i] for i in range(3)]
    ref = sitk.Image(size, sitk.sitkFloat32)
    ref.SetSpacing([SPACING] * 3)
    ref.SetOrigin(origin)
    ref.SetDirection(direction)
    sitk.WriteImage(ref, out_path)


def resample_1mm(t1_path, out_path):
    img = sitk.ReadImage(t1_path, sitk.sitkFloat32)
    orig_spacing, orig_size = img.GetSpacing(), img.GetSize()
    new_size = [max(1, int(round(osz * ospc / SPACING))) for osz, ospc in zip(orig_size, orig_spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([SPACING] * 3)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    sitk.WriteImage(resampler.Execute(img), out_path)


def main():
    t1_path, work_dir, out_png = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(work_dir, exist_ok=True)

    t1_1mm = os.path.join(work_dir, "t1_1mm.nii.gz")
    resample_1mm(t1_path, t1_1mm)

    brain = os.path.join(work_dir, "brain.nii.gz")
    run(["hd-bet", "-i", t1_1mm, "-o", brain, "-device", "cuda", "--save_bet_mask"])
    mask_path = brain.replace(".nii.gz", "_bet.nii.gz")

    t1_img = nib.load(t1_1mm)
    stripped_arr = np.asanyarray(t1_img.dataobj).astype(np.float32) * \
        (np.asanyarray(nib.load(mask_path).dataobj).astype(np.float32) > 0.5)
    stripped_path = os.path.join(work_dir, "stripped.nii.gz")
    nib.save(nib.Nifti1Image(stripped_arr, t1_img.affine, t1_img.header), stripped_path)

    template_path = fetch_template()
    reg_prefix = os.path.join(work_dir, "reg_")
    run(["antsRegistrationSyNQuick.sh", "-d", "3", "-f", template_path, "-m", stripped_path,
         "-t", "a", "-o", reg_prefix, "-n", "8"])
    transform = reg_prefix + "0GenericAffine.mat"

    ref_crop = os.path.join(work_dir, "ref_crop.nii.gz")
    build_reference_grid(template_path, ref_crop)

    warped = os.path.join(work_dir, "warped_cropbox.nii.gz")
    run(["antsApplyTransforms", "-d", "3", "-i", stripped_path, "-r", ref_crop,
         "-o", warped, "-t", transform, "-n", "Linear"])

    arr = np.asanyarray(nib.load(warped).dataobj).astype(np.float32)
    img = center_pad(arr, FINAL_SHAPE)
    assert img.shape == FINAL_SHAPE

    x0, x1, y0, y1, z0, z1 = LV_BBOX_AFFINE
    mask = np.zeros(FINAL_SHAPE, dtype=bool)
    mask[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = True
    cx, cy, cz = (x0 + x1) // 2, (y0 + y1) // 2, (z0 + z1) // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    views = [
        ("Sagittal", img[cx, :, :], mask[cx, :, :]),
        ("Coronal", img[:, cy, :], mask[:, cy, :]),
        ("Axial", img[:, :, cz], mask[:, :, cz]),
    ]
    for col, (name, sl, msl) in enumerate(views):
        ax = axes[0, col]
        ax.imshow(sl.T, cmap="gray", origin="lower")
        ax.set_title(f"{name} (real PMG case)")
        ax.axis("off")

        ax2 = axes[1, col]
        ax2.imshow(sl.T, cmap="gray", origin="lower")
        overlay = np.zeros((*msl.T.shape, 4))
        overlay[msl.T, 0] = 1.0
        overlay[msl.T, 3] = 0.45
        ax2.imshow(overlay, origin="lower")
        ax2.set_title(f"{name} + masked LV/periventricular bbox")
        ax2.axis("off")

    fig.suptitle(
        "Task 5: fixed lateral-ventricle + periventricular masking region\n"
        "(HD-BET + rigid+affine registration to MNI152NLin2009cAsym,\n"
        "warped onto the exact fixed crop box used in training, no coordinate rescaling)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print("saved:", out_png)


if __name__ == "__main__":
    main()
