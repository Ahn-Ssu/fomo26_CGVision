"""Plot train/eval learning curves across FOMO26 runs (ver1, ver2, ...).

Reads the tidy/long-format train_log.csv + eval_log.csv (see run_pretrain.py's
_LOG_FIELDS) for each run under FOMO26/expr/<run_name>/. One subplot per
teacher, ver1/ver2/... overlaid as different colors within each subplot --
same layout for the train-loss figure and the eval-loss figure.

Resume dedup: a resumed run re-logs the step range between its last
checkpoint and the interruption point (see run_pretrain.py's auto-resume --
last.pt only saves every --ckpt_every, so anything logged after that but
before an interruption gets logged AGAIN after resuming from that
checkpoint). Rows are deduplicated by (run_name, step, metric, teacher),
keeping the LAST occurrence in the file (append-only -> later = written by
the resumed process, authoritative).

Usage:
    python3 FOMO26/examples/plot_curves.py ver1 ver2
    python3 FOMO26/examples/plot_curves.py ver1 ver2 --out FOMO26/expr/plots
    python3 FOMO26/examples/plot_curves.py ver1 ver2 --smooth 10
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPR_ROOT = "/root/FOMO26/expr"


def load_log(run_name: str, filename: str):
    path = os.path.join(EXPR_ROOT, run_name, filename)
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def dedup_last(rows):
    """Keep the LAST row per (step, teacher) -- later in file wins (resumed
    run's re-logged rows are authoritative over the pre-interruption ones)."""
    by_key = {}
    for row in rows:
        key = (int(row["step"]), row["teacher"])
        by_key[key] = row  # later overwrites earlier
    return by_key


def rolling_mean(xs, ys, window):
    if window <= 1:
        return xs, ys
    out_x, out_y = [], []
    for i in range(len(ys)):
        lo = max(0, i - window + 1)
        out_x.append(xs[i])
        out_y.append(sum(ys[lo:i + 1]) / (i - lo + 1))
    return out_x, out_y


def plot_metric(run_names, filename, phase, metric, out_path, smooth=1, title=None,
                 clip_percentile=99.0):
    # {teacher: {run_name: [(step, value), ...]}}
    data = {}
    for run_name in run_names:
        rows = load_log(run_name, filename)
        rows = [r for r in rows if r["phase"] == phase and r["metric"] == metric]
        by_key = dedup_last(rows)
        for (step, teacher), row in by_key.items():
            data.setdefault(teacher, {}).setdefault(run_name, []).append((step, float(row["value"])))

    if not data:
        print(f"WARNING: no rows for phase={phase} metric={metric} across runs {run_names} -- skipping")
        return

    teachers = sorted(data.keys())

    # Robust y-limit: rare extreme spikes (bad batch / outlier sample -- see
    # 2026-07-23 chat, confirmed genuine logged values, not a parsing bug,
    # ~1% of rows reach 1e6+ while the rest sit in the 0-2 range) would
    # otherwise crush the whole informative range to a flat line. Clip the
    # VIEW only (data on disk is untouched) to clip_percentile of all values
    # actually plotted, and report how many points fall outside it per subplot.
    all_vals = [v for teacher in data.values() for run_vals in teacher.values() for _, v in run_vals]
    y_cap = None
    if all_vals:
        sorted_vals = sorted(all_vals)
        idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * clip_percentile / 100))
        y_cap = max(sorted_vals[idx], 1e-6)

    fig, axes = plt.subplots(1, len(teachers), figsize=(6 * len(teachers), 4.5), sharey=True)
    if len(teachers) == 1:
        axes = [axes]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, teacher in zip(axes, teachers):
        n_clipped = 0
        for i, run_name in enumerate(run_names):
            series = sorted(data[teacher].get(run_name, []))
            if not series:
                continue
            xs = [s for s, _ in series]
            ys = [v for _, v in series]
            if y_cap is not None:
                n_clipped += sum(1 for v in ys if v > y_cap)
            xs_s, ys_s = rolling_mean(xs, ys, smooth)
            ax.plot(xs_s, ys_s, label=run_name, color=colors[i % len(colors)], linewidth=1.5)
        ax.set_title(teacher + (f"  ({n_clipped} pt clipped)" if n_clipped else ""))
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        if y_cap is not None:
            ax.set_ylim(0, y_cap * 1.05)
    axes[0].set_ylabel(metric)
    axes[0].legend()
    fig.suptitle(title or f"{phase} {metric}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_names", nargs="+", help="run_name(s) under FOMO26/expr/ to compare")
    p.add_argument("--out", type=str, default="/root/FOMO26/expr/plots",
                    help="output directory for the PNGs")
    p.add_argument("--smooth", type=int, default=5,
                    help="rolling-mean window applied to the TRAIN loss curve only "
                         "(episodic sampling makes raw per-teacher train loss noisy/sparse); "
                         "eval loss is plotted raw (already averaged over the full holdout "
                         "at each --eval_every checkpoint)")
    p.add_argument("--clip_percentile", type=float, default=99.0,
                    help="y-axis view is clipped to this percentile of plotted values "
                         "(rare extreme-loss outlier steps, e.g. a bad batch, would otherwise "
                         "crush the informative range to a flat line -- see script docstring)")
    args = p.parse_args()

    tag = "_".join(args.run_names)
    plot_metric(args.run_names, "train_log.csv", "train", "distill_loss",
                os.path.join(args.out, f"train_loss_{tag}.png"),
                smooth=args.smooth, title=f"Train loss ({', '.join(args.run_names)})",
                clip_percentile=args.clip_percentile)
    plot_metric(args.run_names, "eval_log.csv", "eval", "distill_loss",
                os.path.join(args.out, f"eval_loss_{tag}.png"),
                smooth=1, title=f"Eval loss ({', '.join(args.run_names)})",
                clip_percentile=args.clip_percentile)
