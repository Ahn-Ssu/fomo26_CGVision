import json, os
import torch
import numpy as np
from scipy.ndimage import binary_dilation, binary_closing, binary_erosion, label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/root/fomo26task5/lr_asymmetry_analysis"

VARIANTS = {
    "rigid": "/root/asparagus_data/Task5_PMG_rigid_iso10",
    "affine": "/root/asparagus_data/Task5_PMG_affine_iso10",
}

LOW_INTENSITY_PCTL = 15   # per-subject: bottom-N% of in-brain intensity counts as "CSF-like"
CONSISTENCY_THRESH = 0.30  # a voxel must be CSF-like in >=30% of subjects to count as ventricle
CLOSE_ITERS = 3
DILATE_ITERS = 2
SURFACE_ERODE_ITERS = 18  # exclude peripheral/pial CSF: only consider voxels this far inside the group-mean brain


def load_all_subjects(data_path):
    paths = []
    for fold in range(5):
        paths += json.load(open(f"{data_path}/TEST_stratified5_fold{fold}.json"))
    imgs = []
    for p in paths:
        img, label_ = torch.load(p, map_location="cpu", weights_only=False)
        imgs.append(img[0].numpy())
    return np.stack(imgs)


def make_montage(volume, title, out_path, cmap="gray", vmin=None, vmax=None, n_slices=24):
    X, Y, Z = volume.shape
    margin = int(Z * 0.15)
    idxs = np.linspace(margin, Z - margin, n_slices).astype(int)
    ncols = 6
    nrows = int(np.ceil(n_slices / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    for i, idx in enumerate(idxs):
        ax = axes.flat[i]
        im = ax.imshow(volume[:, :, idx].T, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"z={idx}", fontsize=8)
        ax.axis("off")
    for j in range(len(idxs), nrows * ncols):
        axes.flat[j].axis("off")
    fig.suptitle(title, fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.6)
    fig.savefig(out_path, dpi=95)
    plt.close(fig)


for variant_name, data_path in VARIANTS.items():
    print(f"\n=== {variant_name} ===")
    imgs = load_all_subjects(data_path)
    n = imgs.shape[0]
    print(f"loaded {n} subjects")

    csf_like = np.zeros_like(imgs, dtype=bool)
    for i in range(n):
        brain_fg = imgs[i] > (imgs[i].max() * 0.05)
        thresh = np.percentile(imgs[i][brain_fg], LOW_INTENSITY_PCTL)
        csf_like[i] = brain_fg & (imgs[i] <= thresh)

    consistency = csf_like.mean(axis=0)  # fraction of subjects CSF-like at each voxel

    ref_anatomy_fg = imgs.mean(axis=0) > (imgs.mean(axis=0).max() * 0.05)
    deep_brain = binary_erosion(ref_anatomy_fg, iterations=SURFACE_ERODE_ITERS)
    print(f"deep-brain interior region (excludes peripheral/pial CSF): {deep_brain.sum()} voxels ({100*deep_brain.mean():.2f}%)")

    vent_mask = (consistency >= CONSISTENCY_THRESH) & deep_brain

    # keep only the largest connected component(s) near the center -- ventricles
    # are the big central blob; drop small scattered false positives
    labeled, n_comp = label(vent_mask)
    if n_comp > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # background
        # keep components with at least 5% of the largest component's size
        biggest = sizes.max()
        keep_labels = np.where(sizes >= 0.05 * biggest)[0]
        vent_mask = np.isin(labeled, keep_labels)

    vent_mask = binary_closing(vent_mask, iterations=CLOSE_ITERS)
    vent_mask = binary_dilation(vent_mask, iterations=DILATE_ITERS)

    print(f"consistency map: max={consistency.max():.2f}")
    print(f"final ventricle mask: {vent_mask.sum()} voxels ({100*vent_mask.mean():.2f}% of volume)")

    ref_anatomy = imgs.mean(axis=0)
    make_montage(consistency, f"{variant_name}: CSF-consistency map (fraction of subjects CSF-like per voxel)",
                 f"{OUT_DIR}/{variant_name}_ventmask_A_consistency.png", cmap="viridis", vmin=0, vmax=1)
    make_montage(ref_anatomy * (1 - vent_mask) + vent_mask * ref_anatomy.max() * 1.3,
                 f"{variant_name}: group-mean anatomy with ventricle mask highlighted (bright)",
                 f"{OUT_DIR}/{variant_name}_ventmask_B_overlay_on_anatomy.png", cmap="gray")

    np.save(f"{OUT_DIR}/{variant_name}_ventricle_mask.npy", vent_mask)
    print(f"saved mask to {OUT_DIR}/{variant_name}_ventricle_mask.npy")

print(f"\nAll outputs in: {OUT_DIR}")
