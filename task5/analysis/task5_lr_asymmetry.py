import json, os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/root/fomo26task5/lr_asymmetry_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

LR_AXIS = 0  # spatial axis 0 (of the (X,Y,Z) dims after the leading channel dim) -- confirmed empirically: flip-diff is smallest here for both rigid and affine

VARIANTS = {
    "rigid": "/root/asparagus_data/Task5_PMG_rigid_iso10",
    "affine": "/root/asparagus_data/Task5_PMG_affine_iso10",
}


def load_all_subjects(data_path):
    paths = []
    for fold in range(5):
        paths += json.load(open(f"{data_path}/TEST_stratified5_fold{fold}.json"))
    imgs, labels, subs = [], [], []
    for p in paths:
        img, label = torch.load(p, map_location="cpu", weights_only=False)
        imgs.append(img[0].numpy())  # (X,Y,Z)
        labels.append(int(label.item()))
        subs.append(p.split("/")[-3])
    return np.stack(imgs), np.array(labels), subs


def make_montage(volume, title, out_path, cmap="gray", vmin=None, vmax=None, n_slices=24):
    X, Y, Z = volume.shape
    # slice along axis2 so each slice shows the full axis0(LR) x axis1 plane
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
    fig.suptitle(title, fontsize=14)
    fig.colorbar(im, ax=axes, shrink=0.6)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"saved: {out_path}")


for variant_name, data_path in VARIANTS.items():
    print(f"\n=== {variant_name} ===")
    imgs, labels, subs = load_all_subjects(data_path)
    print(f"loaded {len(labels)} subjects, {labels.sum()} PMG positive, {(labels==0).sum()} control")

    flipped = np.flip(imgs, axis=1 + LR_AXIS)  # +1 because imgs has a leading subject dim
    asym = np.abs(imgs - flipped)  # (N, X, Y, Z)

    pmg_mean_asym = asym[labels == 1].mean(axis=0)
    ctrl_mean_asym = asym[labels == 0].mean(axis=0)
    diff = pmg_mean_asym - ctrl_mean_asym

    ref_anatomy = imgs.mean(axis=0)  # group-mean anatomical reference for context

    vmax_asym = np.percentile(np.stack([pmg_mean_asym, ctrl_mean_asym]), 99.5)
    diff_absmax = np.percentile(np.abs(diff), 99.5)

    make_montage(ref_anatomy, f"{variant_name}: group-mean anatomy (reference)",
                 f"{OUT_DIR}/{variant_name}_00_reference_anatomy.png", cmap="gray")
    make_montage(pmg_mean_asym, f"{variant_name}: mean |LR-flip asymmetry| -- PMG positive (n={int(labels.sum())})",
                 f"{OUT_DIR}/{variant_name}_01_pmg_mean_asymmetry.png", cmap="hot", vmin=0, vmax=vmax_asym)
    make_montage(ctrl_mean_asym, f"{variant_name}: mean |LR-flip asymmetry| -- control (n={int((labels==0).sum())})",
                 f"{OUT_DIR}/{variant_name}_02_control_mean_asymmetry.png", cmap="hot", vmin=0, vmax=vmax_asym)
    make_montage(diff, f"{variant_name}: DIFFERENCE (PMG asymmetry - control asymmetry)\nred=PMG more asymmetric here, blue=control more asymmetric here",
                 f"{OUT_DIR}/{variant_name}_03_diff_pmg_minus_control.png", cmap="RdBu_r", vmin=-diff_absmax, vmax=diff_absmax)

    np.savez(f"{OUT_DIR}/{variant_name}_asymmetry_maps.npz",
             pmg_mean_asym=pmg_mean_asym, ctrl_mean_asym=ctrl_mean_asym, diff=diff, ref_anatomy=ref_anatomy)
    print(f"saved arrays to {OUT_DIR}/{variant_name}_asymmetry_maps.npz")

print(f"\nAll outputs in: {OUT_DIR}")
