"""Pools per-subject age predictions across all 10 folds for each of the 2
head_mode arms (base, multistage_gem), computes ONE overall MAE/RMSE/Pearson-r
per arm (not per-fold -- a fold's ~50 held-out subjects CAN support per-fold
metrics unlike Task 1's 2-3-subject folds, but pooling still gives the
tightest, most representative estimate and keeps this directly comparable in
methodology to asparagus_aggregate_stratified10.py), a 1000-resample bootstrap
95% CI per arm, and a PAIRED bootstrap (same resampled subject indices for
both arms) for Delta-MAE = MAE_base - MAE_multistage_gem, so fold-split noise
common to both arms cancels out.

Run only after all 20 asparagus_orchestrate_task3_head_arch.py jobs have
finished. Writes a flat CSV: subject_id, fold, arm, true_age, pred_age,
best_epoch, best_val_loss (one row per (subject, arm) pair, 494*2 rows).
"""
import csv
import glob
import json
import os

import numpy as np

MODELS_ROOT = "/root/FOMO26/expr/downstream_finetune/REGR002_FOMO26_BrainAge/fomo26_student_clsreg__3D/script=finetune_reg"
LABELS_PATH = "/root/asparagus_data/REGR002_FOMO26_BrainAge_iso1mm/stratified10_labels.json"
OUT_CSV = "/root/FOMO26/expr/downstream_finetune/task3_stratified10_predictions.csv"
N_BOOTSTRAP = 1000
SEED = 0

HEAD_MODES = ["base", "multistage_gem"]


def find_prediction_json(fold, head_mode):
    # Exact clargs= key order confirmed empirically (2026-08-13 smoke test run
    # on the authoring machine, root=task3_smoketest): only keys NOT already
    # in configs/core/base.yaml's override_dirname.exclude_keys show up here,
    # in plain alphabetical order -- data.fold/data.train_split/
    # lightning.lightning_module/logger.wandb_logging/training.* are all
    # excluded already, so they must NOT appear in this pattern.
    ckpt = "/root/FOMO26/expr/pretraining/ver5/checkpoints/step_280000.pt"
    data_root = "/root/asparagus_data/REGR002_FOMO26_BrainAge_iso1mm"
    pattern = os.path.join(
        MODELS_ROOT,
        "root=task3_head_arch__stem=None_last.ckpt",
        f"leaf=default_finetune_reg__clargs=hardware.compile_mode=null,"
        f"model.checkpoint_path={ckpt},model.from_scratch=false,"
        f"model.fusion_mode=early,model.head_mode={head_mode},model.stem_mode=frozen,"
        f"model.teacher_name=brats,"
        f"test_split_path={data_root}/TEST_stratified10_fold{fold}.json,"
        f"train_split_path={data_root}/split_stratified10.json",
        "split_stratified10__fold=" + str(fold),
        "run_id=*",
        "predictions",
        f"REGR002_FOMO26_BrainAge__TEST_stratified10_fold{fold}__best.json",
    )
    matches = glob.glob(pattern)
    if not matches:
        # override_dirname key ORDER is alphabetical by hydra convention but
        # hasn't been verified against a real run yet on this machine (Task 3
        # was never actually launched here, only smoke-tested elsewhere) --
        # fall back to a loose glob if the strict pattern above finds nothing,
        # and PRINT what it found so a human can fix the exact pattern once
        # one real run_id directory exists to inspect.
        loose = glob.glob(os.path.join(
            MODELS_ROOT, "root=task3_head_arch*", f"leaf=*head_mode={head_mode}*",
            f"*fold={fold}", "run_id=*", "predictions",
            f"REGR002_FOMO26_BrainAge__TEST_stratified10_fold{fold}__best.json",
        ))
        if loose:
            print(f"  [pattern fallback] strict pattern found nothing for "
                  f"(fold={fold}, head_mode={head_mode}), but loose glob found "
                  f"{len(loose)} match(es) -- fix find_prediction_json's exact "
                  f"pattern to match, this fallback is not guaranteed correct "
                  f"if more than one run_id exists.")
        return loose
    return matches


def pool_arm(head_mode, true_labels):
    pooled = {}
    missing_folds = []
    for fold in range(10):
        matches = find_prediction_json(fold, head_mode)
        if not matches:
            missing_folds.append(fold)
            continue
        with open(matches[-1]) as f:
            result = json.load(f)
        for file_path, entry in result.items():
            if file_path == "metrics":
                continue
            sub = [p for p in file_path.split("/") if p.startswith("sub-")][0]
            pooled[sub] = {
                "fold": fold,
                "true_age": entry["label"],
                "pred_age": entry["prediction"],
                "best_epoch": entry.get("best_epoch"),
                "best_val_loss": entry.get("best_val_loss"),
            }
    return pooled, missing_folds


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def pearson_r(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi


def paired_delta_mae_ci(y_true, pred_a, pred_b, n_boot=N_BOOTSTRAP, seed=SEED):
    """Same resampled indices applied to both arms per resample -- cancels
    fold-split noise common to both, since both arms were evaluated on the
    identical partition. Delta = MAE_a - MAE_b (negative = a is better)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pred_a, pred_b = np.asarray(pred_a), np.asarray(pred_b)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas.append(mae(y_true[idx], pred_a[idx]) - mae(y_true[idx], pred_b[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return lo, hi


def main():
    with open(LABELS_PATH) as f:
        true_labels = json.load(f)
    n_total = len(true_labels)

    all_pooled = {}
    csv_rows = []
    for head_mode in HEAD_MODES:
        pooled, missing_folds = pool_arm(head_mode, true_labels)
        all_pooled[head_mode] = pooled

        print(f"\n=== {head_mode} ===")
        if missing_folds:
            print(f"  MISSING folds (not yet complete or failed): {missing_folds}")
        print(f"  subjects pooled: {len(pooled)}/{n_total}")

        missing_subjects = set(true_labels) - set(pooled)
        if missing_subjects:
            print(f"  subjects never covered: {sorted(missing_subjects)[:10]}"
                  f"{' ...' if len(missing_subjects) > 10 else ''}")

        for sub, entry in pooled.items():
            csv_rows.append({
                "subject_id": sub, "fold": entry["fold"], "arm": head_mode,
                "true_age": entry["true_age"], "pred_age": entry["pred_age"],
                "best_epoch": entry["best_epoch"], "best_val_loss": entry["best_val_loss"],
            })

        if len(pooled) < n_total:
            print("  -> incomplete, skipping metrics")
            continue

        subs = sorted(pooled)
        y_true = [pooled[s]["true_age"] for s in subs]
        y_pred = [pooled[s]["pred_age"] for s in subs]
        m, r, rho = mae(y_true, y_pred), rmse(y_true, y_pred), pearson_r(y_true, y_pred)
        mae_lo, mae_hi = bootstrap_ci(y_true, y_pred, mae)
        rmse_lo, rmse_hi = bootstrap_ci(y_true, y_pred, rmse)
        print(f"  Pooled MAE (n={n_total}): {m:.2f} years, 95% CI [{mae_lo:.2f}, {mae_hi:.2f}]")
        print(f"  Pooled RMSE: {r:.2f} years, 95% CI [{rmse_lo:.2f}, {rmse_hi:.2f}]")
        print(f"  Pearson r (pred vs true age): {rho:.3f}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "fold", "arm", "true_age", "pred_age",
                                           "best_epoch", "best_val_loss"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nWrote per-subject predictions CSV: {OUT_CSV}")

    print("\n=== Paired Delta-MAE (base - multistage_gem), same bootstrap indices ===")
    a, b = "base", "multistage_gem"
    pooled_a, pooled_b = all_pooled[a], all_pooled[b]
    if len(pooled_a) < n_total or len(pooled_b) < n_total:
        print(f"  {a} vs {b}: incomplete, skipping")
        return
    subs = sorted(set(pooled_a) & set(pooled_b))
    if len(subs) < n_total:
        print(f"  {a} vs {b}: subject sets don't fully match ({len(subs)}/{n_total} common) "
              f"-- check partitions are identical")
    y_true = [true_labels[s] for s in subs]
    pred_a = [pooled_a[s]["pred_age"] for s in subs]
    pred_b = [pooled_b[s]["pred_age"] for s in subs]
    mae_a, mae_b = mae(y_true, pred_a), mae(y_true, pred_b)
    lo, hi = paired_delta_mae_ci(y_true, pred_a, pred_b)
    print(f"  {a} (MAE={mae_a:.2f}) vs {b} (MAE={mae_b:.2f}): "
          f"Delta={mae_a - mae_b:+.2f} years, 95% CI=[{lo:+.2f}, {hi:+.2f}]"
          f"{'  -> excludes 0, signal' if lo > 0 or hi < 0 else '  -> includes 0, no significant difference'}")


if __name__ == "__main__":
    main()
