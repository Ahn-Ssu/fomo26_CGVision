"""Quick sanity check: index a small subset of FOMO300K, load a few real
samples, print shapes/stats/modalities. Defaults to the preprocessed .npz
source (fast); pass --data_source zip to test the original zip-reading path
instead (does NOT build the full 81k-zip index by default -- pass --limit to
control how much of the corpus to scan).

Usage: python3 /root/FOMO26/examples/sanity_check_dataset.py [--data_source preprocessed|zip] [--limit N] [--n_samples N]
"""

import argparse
import sys
import time

sys.path.insert(0, "/root")
from FOMO26.data.fomo300k_dataset import FOMO300KDataset, FOMO300KPreprocessedDataset  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_source", type=str, default="preprocessed", choices=["preprocessed", "zip"])
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--patch_size", type=int, default=128)
    args = p.parse_args()

    t0 = time.time()
    if args.data_source == "preprocessed":
        ds = FOMO300KPreprocessedDataset(patch_size=args.patch_size, limit=args.limit, seed=0)
    else:
        ds = FOMO300KDataset(patch_size=args.patch_size, limit_zips=args.limit, seed=0)
    print(f"Indexed {len(ds)} entries ({args.data_source}, limit={args.limit}) in {time.time()-t0:.1f}s")

    modalities_seen = {}
    for i in range(min(args.n_samples, len(ds))):
        t0 = time.time()
        sample = ds[i]
        dt = time.time() - t0
        img, mask = sample["image"], sample["mask"]
        modalities_seen[sample["modality"]] = modalities_seen.get(sample["modality"], 0) + 1
        print(f"[{i}] {dt:.2f}s modality={sample['modality']:10s} dataset={sample['dataset']:20s} "
              f"shape={tuple(img.shape)} mean={img.mean():.4f} std={img.std():.4f} "
              f"min={img.min():.4f} max={img.max():.4f} mask_frac={mask.float().mean():.3f}")

    print(f"\nModality distribution in this sample: {modalities_seen}")
