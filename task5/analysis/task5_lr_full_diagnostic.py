import json, os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/root/fomo26task5/lr_asymmetry_analysis"
LR_AXIS = 0
N_SLICES = 24

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
        imgs.append(img[0].numpy())
        labels.append(int(label.item()))
        subs.append(p.split("/")[-3])
    return np.stack(imgs), np.array(labels), subs


def make_triplet_montage(img, asym, midline, title, out_path, vmax_asym, n_slices=N_SLICES):
    X, Y, Z = img.shape
    margin = int(Z * 0.15)
    idxs = np.linspace(margin, Z - margin, n_slices).astype(int)
    fig, axes = plt.subplots(n_slices, 3, figsize=(9, 3 * n_slices))
    for row, z in enumerate(idxs):
        raw_slice = img[:, :, z]
        asym_slice = asym[:, :, z]

        ax0 = axes[row, 0]
        ax0.imshow(raw_slice.T, cmap="gray", origin="lower")
        ax0.set_title(f"original  z={z}", fontsize=9)
        ax0.axis("off")

        ax1 = axes[row, 1]
        ax1.imshow(asym_slice.T, cmap="hot", origin="lower", vmin=0, vmax=vmax_asym)
        ax1.set_title(f"|diff|  z={z}", fontsize=9)
        ax1.axis("off")

        ax2 = axes[row, 2]
        ax2.imshow(raw_slice.T, cmap="gray", origin="lower")
        ax2.imshow(asym_slice.T, cmap="hot", origin="lower", alpha=0.55, vmin=0, vmax=vmax_asym)
        ax2.axvline(x=midline, color="cyan", linestyle="--", linewidth=1.0)
        ax2.set_title(f"overlay+midline  z={z}", fontsize=9)
        ax2.axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=85)
    plt.close(fig)


for variant_name, data_path in VARIANTS.items():
    print(f"\n=== {variant_name} ===")
    imgs, labels, subs = load_all_subjects(data_path)
    flipped = np.flip(imgs, axis=1 + LR_AXIS)
    asym = np.abs(imgs - flipped)
    vmax_asym = np.percentile(asym, 99.5)
    X = imgs.shape[1 + LR_AXIS]
    midline = (X - 1) / 2.0

    sub_out_dir = f"{OUT_DIR}/{variant_name}/per_subject_triplet"
    os.makedirs(sub_out_dir, exist_ok=True)
    for i, (sid, label) in enumerate(zip(subs, labels)):
        label_str = "PMG" if label == 1 else "control"
        title = f"{variant_name} / {sid} (true={label_str}) -- original | |LR-diff| | overlay+midline"
        out_path = f"{sub_out_dir}/{label_str}_{sid}.png"
        make_triplet_montage(imgs[i], asym[i], midline, title, out_path, vmax_asym)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(subs)} done", flush=True)
    print(f"saved {len(subs)} triplet montages to {sub_out_dir}")

print(f"\nAll outputs in: {OUT_DIR}")
