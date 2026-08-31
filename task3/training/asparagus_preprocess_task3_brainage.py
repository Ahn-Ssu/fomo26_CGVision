"""Preprocesses FOMO26 Task 3 (Brain Age regression, single T1w modality) to
asparagus_data-compatible (image, label) .pt pairs, matching pretraining's
normalization regime (foreground-masked znorm, verbatim `asparagus_volume_wise_znorm`
-- see /root/FOMO26/data/normalize.py) rather than the un-normalized "no_norm"
preset REGR002_FOMO26_BrainAge.py (asparagus_preprocessing) defaults to: feeding
the frozen pretrained backbone raw un-normalized intensities would reproduce the
exact train/pretrain distribution mismatch that made Task 1's ORIGINAL
preprocessing attempt show "no signal" (see project memory
project_fomo26_task1_pretrain_comparison.md) -- fixed there by matching
pretraining's preprocessing exactly, same lesson applied here up front.

Unlike Task 1 (CLS002, severely anisotropic native spacing, per-subject-variable
shape), Task 3 needs NO resample and NO crop/pad step at all: measured
(2026-08-13, see chat) that all 494 subjects already sit on an IDENTICAL
(176,256,256) grid at (1,1,1)mm spacing with zero variance across the whole
cohort -- i.e. this is already a registered/standardized template space, not
raw acquisition. So this script is just: per-subject foreground-masked znorm
+ label attach + torch.save (one .pt file per subject), matching
asparagus_preprocess_cls002_iso1mm.py's saved-tensor format (`[image, label]`
list, `ClsRegDataset.__getitem__` reads `data[0]`/`data[1]`) but skipping that
script's resample/union-crop/pad machinery entirely since none of it is needed.

Also fixes REGR002_FOMO26_BrainAge.py's `subdir="Task_3/Task_3"` default bug
(that nested subdir does not exist -- actual data is flat at `Task_3/`) by not
depending on that script at all; this one reads directly from the known-correct
paths.
"""
import json
import os
import sys

sys.path.insert(0, "/root/FOMO26")

import nibabel as nib
import numpy as np
import torch
from data.normalize import asparagus_volume_wise_znorm  # noqa: E402

RAW_ROOT = "/root/data/FOMO2026_downstream/Task_3/preprocessed"
LABELS_ROOT = "/root/data/FOMO2026_downstream/Task_3/labels"
OUT_ROOT = "/root/asparagus_data/REGR002_FOMO26_BrainAge_iso1mm/preprocessed"
EXPECTED_SHAPE = (176, 256, 256)


def load_labels() -> dict:
    labels = {}
    for sub in sorted(os.listdir(RAW_ROOT)):
        label_path = os.path.join(LABELS_ROOT, sub, "ses-01", "labels.txt")
        if not os.path.exists(label_path):
            print(f"[SKIP] {sub}: no label file at {label_path}")
            continue
        with open(label_path) as f:
            labels[sub] = float(f.read().strip())
    return labels


def process_subject(sub: str, age: float) -> bool:
    out_path = os.path.join(OUT_ROOT, sub, "ses-01", "t1w.pt")
    if os.path.exists(out_path):
        return True

    nii_path = os.path.join(RAW_ROOT, sub, "ses-01", "t1w.nii.gz")
    if not os.path.exists(nii_path):
        print(f"[SKIP] {sub}: missing {nii_path}")
        return False

    data = nib.load(nii_path).get_fdata().astype(np.float32)
    if data.shape != EXPECTED_SHAPE:
        print(f"[WARN] {sub}: shape {data.shape} != expected {EXPECTED_SHAPE} "
              f"-- this subject deviates from the rest of the cohort, double check before trusting it")

    mask = data > 1e-6
    if not mask.any():
        print(f"[SKIP] {sub}: no foreground (all-zero volume?)")
        return False

    normed = asparagus_volume_wise_znorm(data, mask=mask).astype(np.float32)

    image_tensor = torch.from_numpy(normed[None])  # [1, D, H, W] -- single modality
    label_tensor = torch.tensor([age], dtype=torch.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save([image_tensor, label_tensor], out_path)
    print(f"[OK] {sub}: shape={data.shape}, age={age}")
    return True


def main():
    labels = load_labels()
    os.makedirs(OUT_ROOT, exist_ok=True)
    ok, failed = [], []
    for sub in sorted(labels):
        if process_subject(sub, labels[sub]):
            ok.append(sub)
        else:
            failed.append(sub)

    print(f"\n{len(ok)}/{len(labels)} subjects processed OK.")
    if failed:
        print(f"FAILED ({len(failed)}): {failed}")

    with open(os.path.join(os.path.dirname(OUT_ROOT), "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)


if __name__ == "__main__":
    main()
