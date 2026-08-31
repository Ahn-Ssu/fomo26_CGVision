"""Extract each fold's OWN best-epoch checkpoint (per-fold peak, from that
fold's own held-out subjects only -- no cross-fold leakage since each
fold's epoch choice never looks at another fold's data or the official
test set) into container/models_staging_<out>/model_fold{i}.ckpt.

Contrasts with extract_weights_ensemble.py (single epoch fixed across all
10 folds): here every fold can use a DIFFERENT epoch, matching its own
best pooled-dice checkpoint on its own 2-3 held-out subjects. User's
rationale (2026-08-20): a fixed epoch captures "the on-average best
stopping point," while per-fold best.ckpt captures "each fold's own best
validation weights" -- genuinely different information, worth testing as
a separate submission rather than assuming one dominates.

Usage: python3 extract_weights_bestckpt.py --variant AUGEF --out augef_bestckpt"""
import argparse
import csv
import glob
import os
import shutil
import sys
from collections import defaultdict

import torch

sys.path.insert(0, "/root/fomo26_segm_mk5")
sys.path.insert(0, "/root/fomo26_segm_mk5/task2_archsearch")
from evaluate_10foldcv_trajectory import VARIANTS, run_dir  # noqa: E402

PRETRAIN_CKPT = "/root/FOMO26/expr/pretraining/ver5/checkpoints/step_280000.pt"
CONTAINER_DIR = "/root/fomo26_segm_mk5/segmentation/container"
TRAJECTORY_DIR = "/root/fomo26_segm_mk5/task2_archsearch/output/10foldcv_trajectory"


def per_fold_best_epochs(variant):
    rows = []
    for f in sorted(glob.glob(f"{TRAJECTORY_DIR}/{variant}_fold*.csv")):
        with open(f) as fh:
            rows.extend(csv.DictReader(fh))
    by_fold = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_fold[int(r["fold"])][int(r["epoch"])].append(float(r["dice"]))
    best = {}
    for fold, eps in by_fold.items():
        epoch_means = {ep: sum(v) / len(v) for ep, v in eps.items()}
        best[fold] = max(epoch_means, key=epoch_means.get)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = VARIANTS[args.variant]
    best_epochs = per_fold_best_epochs(args.variant)
    assert len(best_epochs) == 10, f"expected 10 folds, found {len(best_epochs)}: {sorted(best_epochs)}"

    out_dir = os.path.join(CONTAINER_DIR, f"models_staging_{args.out}")
    os.makedirs(out_dir, exist_ok=True)

    for fold in range(10):
        epoch = best_epochs[fold]
        step = (epoch + 1) * 250
        seed = cfg["seed_base"] + fold
        d = run_dir(args.variant, fold, seed)
        ckpt_path = d / f"milestone_step={step:06d}.ckpt"
        assert ckpt_path.exists(), f"fold {fold}: missing {ckpt_path}"

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        raw_sd = ckpt["state_dict"]
        stripped = {k[len("model."):]: v for k, v in raw_sd.items() if k.startswith("model.")}
        out_path = os.path.join(out_dir, f"model_fold{fold}.ckpt")
        torch.save(stripped, out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"fold{fold}: own-best epoch={epoch} (step={step}) -> {out_path} ({size_mb:.0f}MB, {len(stripped)} tensors)")

    pretrain_out = os.path.join(out_dir, "pretrain_arch.pt")
    if not os.path.exists(pretrain_out):
        shutil.copy(PRETRAIN_CKPT, pretrain_out)
    print(f"pretrain_arch.pt -> {pretrain_out} ({os.path.getsize(pretrain_out)/1e6:.0f}MB)")

    total_mb = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)) / 1e6
    print(f"\n{out_dir} ready: {total_mb:.0f}MB total ({args.variant}, per-fold best epochs: {best_epochs})")


if __name__ == "__main__":
    main()
