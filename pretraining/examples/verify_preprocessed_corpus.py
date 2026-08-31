"""Exhaustive verification of the raw-preserving FOMO300K .npz cache
(/root/data/FOMO300K_preprocessed/): checks EVERY file's `spacing` field
(cheap partial zip-member read, no full decompression) is exactly
[1.0, 1.0, 1.0], plus does a deeper full-load sanity check (finite values,
shape consistency, required keys) on a random sample.

Usage: python3 /root/FOMO26/examples/verify_preprocessed_corpus.py [--sample N]
"""

import argparse
import glob
import io
import multiprocessing as mp
import random
import time
import zipfile

import numpy as np

ROOT = "/root/data/FOMO300K_preprocessed"
REQUIRED_KEYS = {"data", "mask", "spacing", "orig_spacing", "orig_shape", "affine",
                  "intensity_scale", "modality", "dataset", "subject", "session", "member"}


def check_spacing(path: str):
    try:
        zf = zipfile.ZipFile(path)
        with zf.open("spacing.npy") as fh:
            spacing = np.load(io.BytesIO(fh.read()))
        ok = np.allclose(spacing, [1.0, 1.0, 1.0])
        return (path, ok, tuple(spacing.tolist()), None)
    except Exception as e:
        return (path, False, None, str(e))


def check_full(path: str):
    try:
        d = np.load(path)
        keys = set(d.keys())
        missing = REQUIRED_KEYS - keys
        data = d["data"]
        mask = d["mask"]
        spacing = d["spacing"]
        shape_match = data.shape == mask.shape
        finite = bool(np.isfinite(data).all())
        spacing_ok = np.allclose(spacing, [1.0, 1.0, 1.0])
        bg_clean = bool(np.all(data[mask == 0] == 0)) if mask.sum() < mask.size else True
        return {
            "path": path, "ok": True, "missing_keys": missing, "shape_match": shape_match,
            "finite": finite, "spacing_ok": spacing_ok, "bg_clean": bg_clean,
            "data_shape": data.shape, "data_dtype": str(data.dtype), "mask_dtype": str(mask.dtype),
        }
    except Exception as e:
        return {"path": path, "ok": False, "error": str(e)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=1000, help="number of files for deep full-load check")
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()

    print("Globbing all .npz files...")
    t0 = time.time()
    all_files = glob.glob(f"{ROOT}/*/*.npz")
    print(f"Found {len(all_files)} files in {time.time()-t0:.1f}s")

    print("\n=== Exhaustive spacing check (all files, partial read) ===")
    t0 = time.time()
    n_bad_spacing = 0
    bad_examples = []
    with mp.Pool(args.workers) as pool:
        for i, (path, ok, spacing, err) in enumerate(pool.imap_unordered(check_spacing, all_files, chunksize=200)):
            if not ok:
                n_bad_spacing += 1
                if len(bad_examples) < 10:
                    bad_examples.append((path, spacing, err))
            if (i + 1) % 50000 == 0:
                print(f"  checked {i+1}/{len(all_files)}...")
    dt = time.time() - t0
    print(f"Done in {dt:.1f}s ({dt/len(all_files)*1000:.2f}ms/file).")
    print(f"Files with spacing != [1,1,1] (or read error): {n_bad_spacing}/{len(all_files)}")
    for path, spacing, err in bad_examples:
        print(f"  BAD: {path} spacing={spacing} err={err}")

    print(f"\n=== Deep full-load check (random sample of {args.sample}) ===")
    random.seed(0)
    sample = random.sample(all_files, min(args.sample, len(all_files)))
    t0 = time.time()
    n_issues = 0
    with mp.Pool(args.workers) as pool:
        for r in pool.imap_unordered(check_full, sample, chunksize=10):
            if not r["ok"]:
                n_issues += 1
                print(f"  LOAD FAILED: {r['path']} -- {r['error']}")
                continue
            problems = []
            if r["missing_keys"]:
                problems.append(f"missing_keys={r['missing_keys']}")
            if not r["shape_match"]:
                problems.append("data/mask shape mismatch")
            if not r["finite"]:
                problems.append("non-finite values in data")
            if not r["spacing_ok"]:
                problems.append("spacing != [1,1,1]")
            if not r["bg_clean"]:
                problems.append("background not clean (mask==0 but data!=0)")
            if problems:
                n_issues += 1
                print(f"  ISSUE: {r['path']} -- {', '.join(problems)}")
    dt = time.time() - t0
    print(f"Done in {dt:.1f}s. Issues found: {n_issues}/{len(sample)}")

    print(f"\n=== SUMMARY ===")
    print(f"Total files: {len(all_files)}")
    print(f"Spacing check (exhaustive): {len(all_files)-n_bad_spacing}/{len(all_files)} OK")
    print(f"Deep check (sample of {len(sample)}): {len(sample)-n_issues}/{len(sample)} OK")
