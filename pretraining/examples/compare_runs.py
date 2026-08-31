"""Compare eval_log.csv (and optionally train_log.csv) across multiple
FOMO26 runs at matching steps -- e.g. ver1 (shared-only) vs ver2 (+LoRA/
per-teacher bias). Relies on every run using the SAME fixed monitoring
holdout (data/fomo300k_dataset.py::split_train_eval, EVAL_HASH_MODULUS is
intentionally not run-configurable -- see run_pretrain.py) so eval numbers
are directly comparable step-for-step across versions.

Usage:
    python3 /root/FOMO26/examples/compare_runs.py ver1 ver2 [ver3 ...]
    python3 /root/FOMO26/examples/compare_runs.py ver1 ver2 --metric distill_loss --phase eval
"""

import argparse
import csv
import os
from collections import defaultdict

EXPR_ROOT = "/root/FOMO26/expr"


def load_log(run_name: str, filename: str):
    path = os.path.join(EXPR_ROOT, run_name, filename)
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_names", nargs="+", help="run_name(s) under FOMO26/expr/ to compare")
    p.add_argument("--phase", type=str, default="eval", choices=["eval", "train"])
    p.add_argument("--metric", type=str, default="distill_loss")
    args = p.parse_args()

    filename = "eval_log.csv" if args.phase == "eval" else "train_log.csv"

    # {(step, teacher): {run_name: value}}
    table = defaultdict(dict)
    teachers_seen = set()
    steps_seen = set()

    for run_name in args.run_names:
        rows = load_log(run_name, filename)
        if not rows:
            print(f"WARNING: no {filename} found for run '{run_name}' (looked under {EXPR_ROOT}/{run_name}/)")
            continue
        for row in rows:
            if row["metric"] != args.metric:
                continue
            step = int(row["step"])
            teacher = row["teacher"]
            table[(step, teacher)][run_name] = float(row["value"])
            teachers_seen.add(teacher)
            steps_seen.add(step)

    if not table:
        print(f"No rows found for phase={args.phase} metric={args.metric} across runs {args.run_names}")
        raise SystemExit(1)

    col_width = max(12, max(len(r) for r in args.run_names) + 2)
    header = f"{'step':>8} {'teacher':>16} " + "".join(f"{r:>{col_width}}" for r in args.run_names)
    print(header)
    print("-" * len(header))
    for step in sorted(steps_seen):
        for teacher in sorted(teachers_seen):
            if (step, teacher) not in table:
                continue
            vals = table[(step, teacher)]
            row_str = f"{step:>8} {teacher:>16} "
            for run_name in args.run_names:
                v = vals.get(run_name)
                row_str += f"{v:>{col_width}.4f}" if v is not None else f"{'--':>{col_width}}"
            print(row_str)
