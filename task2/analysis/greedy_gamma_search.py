"""Greedy coordinate-ascent search over per-channel gamma TTA combinations
(user request 2026-08-27), following up on gamma_selective_tta_vs_retrain.py
which showed flair=1.5/dwi=1.0/swi_or_t2s=1.75 (each channel's OWN isolated
optimum, found by varying ONE channel at a time in gamma_sweep_per_channel.py)
beats both the no-gamma baseline and the gamma-baked-into-training retrain.

That per-channel sweep never tested channel INTERACTIONS -- e.g. flair's
optimal gamma might shift once swi_or_t2s is also darkened, since both
push the same softmax decision boundary. This script does proper greedy
coordinate ascent: fix two channels, sweep the third over the full grid,
lock in its best value, move to the next channel; repeat for a second
round to check convergence.

Uses CUTOUT_MODDROP's existing fold-best.ckpt models (TTA only, no
retraining) on each fold's own genuinely held-out subjects -- same
infra/model-loading as test_time_augmentation_sensitivity.py, but
restructured so each fold's model is loaded ONCE per sweep and reused
across all grid values (avoids reloading the checkpoint per gamma
value x per fold, which would be ~7x more disk I/O per sweep)."""
import sys
sys.path.insert(0, '.')
from test_time_augmentation_sensitivity import *
import csv

GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
CHANNEL_NAMES = ["flair", "dwi_b1000", "swi_or_t2s"]
SWEEP_ORDER = [0, 2, 1]  # flair, swi_or_t2s, dwi_b1000
N_ROUNDS = 2

cfg = VARIANTS[VARIANT]  # CUTOUT_MODDROP
best_epochs = per_fold_best_epochs(VARIANT)
model = FomoStudentSegNet(input_channels=3, output_channels=2, checkpoint_path=PRETRAIN,
                           teacher_name='brats', from_scratch=False, stem_mode='learnable',
                           freeze_decoder=False).eval().to(DEVICE)


def gamma_transform_channels(image, channel_gammas):
    out = image.clone()
    for c, gamma in channel_gammas.items():
        if gamma == 1.0:
            continue
        ch = out[c]
        lo, hi = ch.min(), ch.max()
        rng = (hi - lo).clamp_min(1e-7)
        norm = (ch - lo) / rng
        out[c] = norm.pow(gamma) * rng + lo
    return out


def sweep_channel(channel_idx, base_gammas, grid):
    """One model-load per fold; every grid value for this channel is
    evaluated on every subject in that same pass."""
    sums = {g: {"dice": 0.0, "nsd": 0.0, "n": 0} for g in grid}
    for fold in range(10):
        epoch = best_epochs[fold]
        step = (epoch + 1) * 250
        seed = cfg["seed_base"] + fold
        d = run_dir(VARIANT, fold, seed)
        ckpt = d / f"milestone_step={step:06d}.ckpt"
        if not ckpt.exists():
            continue
        load_state(model, ckpt)
        model.eval()
        subjects = held_out_subjects(cfg["test_prefix"], fold)
        for subject in subjects:
            batch = make_batch(subject, cfg["pre_dir"])
            image0 = batch["image"][0]
            label0 = batch["label"][0]
            gt = label0.unsqueeze(0).to(DEVICE)
            gt_np = label0.squeeze().cpu().numpy().astype("uint8")
            for g in grid:
                gammas = dict(base_gammas)
                gammas[channel_idx] = g
                img_p = gamma_transform_channels(image0, gammas)
                x = img_p.unsqueeze(0).to(DEVICE)
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model.sliding_window_predict(x, patch_size=PATCH_SIZE, overlap=0.5, mirror=False)
                dice, tp, fp, fn = hard_dice(logits, gt)
                pred = logits.argmax(1)[0].cpu().numpy().astype("uint8")
                nsd = surface_dice(pred, gt_np)
                sums[g]["dice"] += dice
                sums[g]["nsd"] += nsd
                sums[g]["n"] += 1
        print(f"  [fold={fold} done, channel={CHANNEL_NAMES[channel_idx]}]", flush=True)
    return {g: (v["dice"] / v["n"], v["nsd"] / v["n"]) for g, v in sums.items()}


log_rows = []
base = {0: 1.0, 1: 1.0, 2: 1.0}
history = []

for round_idx in range(1, N_ROUNDS + 1):
    print(f"\n=== ROUND {round_idx} === base={base}", flush=True)
    round_changed = False
    for ch in SWEEP_ORDER:
        others = {k: v for k, v in base.items() if k != ch}
        print(f"\n--- sweeping {CHANNEL_NAMES[ch]} (others fixed at {others}) ---", flush=True)
        results = sweep_channel(ch, base, GRID)
        best_g = max(results, key=lambda g: results[g][0])
        for g in GRID:
            md, mn = results[g]
            print(f"  gamma={g:.2f} dice={md:.4f} nsd={mn:.4f}", flush=True)
            log_rows.append({"round": round_idx, "channel": CHANNEL_NAMES[ch], "gamma": g,
                              "dice": md, "nsd": mn, "base_others": str(others)})
        if base[ch] != best_g:
            round_changed = True
        base[ch] = best_g
        print(f"  -> locked {CHANNEL_NAMES[ch]}={best_g} (dice={results[best_g][0]:.4f})", flush=True)
    history.append(dict(base))
    print(f"\nround {round_idx} final combo: {base}", flush=True)
    if not round_changed and round_idx > 1:
        print("CONVERGED -- no change from previous round", flush=True)
        break

with open("output/greedy_gamma_search.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
    w.writeheader()
    w.writerows(log_rows)

print(f"\nFINAL GREEDY COMBO: flair={base[0]}, dwi_b1000={base[1]}, swi_or_t2s={base[2]}")
print("Wrote output/greedy_gamma_search.csv")
