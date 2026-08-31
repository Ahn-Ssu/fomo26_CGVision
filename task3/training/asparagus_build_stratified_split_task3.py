"""Stratified 10-fold split for Task 3 (Brain Age, continuous target) --
adapts asparagus_build_stratified_split_iso1mm.py's manual construction (that
one is hand-rolled for a 21-subject BINARY label, not reusable as-is for a
continuous target across 494 subjects -- see chat 2026-08-13; no
quantile-stratification helper exists anywhere in asparagus/asparagus_preprocessing
to reuse either, confirmed by search).

Method: bin ages into N_BINS quantile bins (equal-COUNT bins, not equal-width --
robust to the right-skewed age distribution any clinical cohort typically has),
then round-robin-distribute each bin's (seed-shuffled) members across the 10
folds. This keeps each fold's age DISTRIBUTION close to the full cohort's,
without pulling in sklearn's StratifiedKFold as a new dependency for something
this simple.

Output format matches asparagus_build_stratified_split_iso1mm.py exactly
(split_stratified10.json / TEST_stratified10_fold{i}.json /
stratified10_labels.json) so the same orchestrator/aggregator patterns apply
unchanged -- see asparagus_orchestrate_task3_head_arch.py.
"""
import json
import os
import random

import numpy as np

DATA_ROOT = "/root/asparagus_data/REGR002_FOMO26_BrainAge_iso1mm"
LABELS_JSON = os.path.join(DATA_ROOT, "preprocessed", "..", "labels.json")  # written by the preprocess script
N_FOLDS = 10
N_BINS = 10  # deciles
VAL_FRAC = 0.1  # fraction of each fold's held-in pool used for validation
SEED = 0


def load_labels() -> dict:
    with open(os.path.normpath(LABELS_JSON)) as f:
        return json.load(f)


def build_folds(labels: dict) -> list:
    subs = sorted(labels)
    ages = np.array([labels[s] for s in subs])
    # N_BINS equal-COUNT bins via quantile edges (not equal-width -- age
    # distributions are rarely uniform, and equal-width bins would leave some
    # bins nearly empty at the tails).
    edges = np.quantile(ages, np.linspace(0, 1, N_BINS + 1))
    edges[-1] += 1e-6  # include the max value in the last bin (digitize is right-exclusive)
    bin_idx = np.digitize(ages, edges[1:-1])  # values in [0, N_BINS)

    folds = [[] for _ in range(N_FOLDS)]
    rng = random.Random(SEED)
    for b in range(N_BINS):
        bin_subs = [subs[i] for i in range(len(subs)) if bin_idx[i] == b]
        rng.shuffle(bin_subs)
        for i, s in enumerate(bin_subs):
            folds[i % N_FOLDS].append(s)
    return folds


def build_split_entries(labels: dict, folds: list) -> list:
    entries = []
    for i, held_out in enumerate(folds):
        held_in = [s for s in labels if s not in held_out]
        fold_rng = random.Random(SEED + 100 + i)
        fold_rng.shuffle(held_in)
        n_val = max(1, round(VAL_FRAC * len(held_in)))
        entries.append({"train": held_in[n_val:], "val": held_in[:n_val]})
    return entries


def validate_partition(labels: dict, folds: list, split_entries: list):
    n = len(labels)
    all_test = set().union(*[set(f) for f in folds])
    assert all_test == set(labels), (
        f"partition doesn't cover exactly the label set "
        f"(missing={set(labels) - all_test}, extra={all_test - set(labels)})"
    )
    assert sum(len(f) for f in folds) == n, "overlap between test folds (duplicate subject count != n)"
    for i, held_out in enumerate(folds):
        train_of_fold = set(split_entries[i]["train"]) | set(split_entries[i]["val"])
        leakage = set(held_out) & train_of_fold
        assert not leakage, f"fold {i}: held-out subjects leaked into its own train/val: {leakage}"
    seen = {}
    for i, f in enumerate(folds):
        key = tuple(sorted(f))
        assert key not in seen, f"fold {i} is identical to fold {seen[key]}: {key}"
        seen[key] = i
    print(f"Partition integrity checks passed: full coverage ({n} subjects), "
          f"no overlap, no leakage, no duplicate folds.")


def report_age_balance(labels: dict, folds: list):
    all_ages = np.array(list(labels.values()))
    print(f"\nCohort age: mean={all_ages.mean():.1f} std={all_ages.std():.1f} "
          f"min={all_ages.min():.0f} max={all_ages.max():.0f} (n={len(labels)})")
    for i, f in enumerate(folds):
        ages = np.array([labels[s] for s in f])
        print(f"  fold {i}: n={len(f)}, mean age={ages.mean():.1f}, std={ages.std():.1f}")


def to_path(sub: str) -> str:
    return os.path.join(DATA_ROOT, "preprocessed", sub, "ses-01", "t1w.pt")


def main():
    labels = load_labels()
    folds = build_folds(labels)
    split_entries = build_split_entries(labels, folds)
    validate_partition(labels, folds, split_entries)
    report_age_balance(labels, folds)

    for fold in folds:
        for s in fold:
            assert os.path.exists(to_path(s)), f"missing preprocessed file for {s}: {to_path(s)}"

    split_out = [
        {"train": [to_path(s) for s in e["train"]], "val": [to_path(s) for s in e["val"]]}
        for e in split_entries
    ]
    with open(os.path.join(DATA_ROOT, "split_stratified10.json"), "w") as f:
        json.dump(split_out, f, indent=2)

    for i, held_out in enumerate(folds):
        with open(os.path.join(DATA_ROOT, f"TEST_stratified10_fold{i}.json"), "w") as f:
            json.dump([to_path(s) for s in held_out], f, indent=2)

    with open(os.path.join(DATA_ROOT, "stratified10_labels.json"), "w") as f:
        json.dump(labels, f, indent=2)

    print("\nWrote split_stratified10.json, TEST_stratified10_fold{0..9}.json, stratified10_labels.json")


if __name__ == "__main__":
    main()
