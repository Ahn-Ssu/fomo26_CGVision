"""Comprehensive before/after comparison (user request 2026-08-28) of the
officially highest-scoring submitted container (CUTOUT_MODDROP per-fold
best.ckpt ensemble, no gamma, no CC filtering) against candidate
inference-time improvements identified this session:
  - selective gamma TTA (flair=1.5, dwi_b1000=1.0, swi_or_t2s=1.75,
    confirmed near-optimal by greedy_gamma_search.py's coordinate ascent)
  - largest-connected-component (LCC) postprocessing (NOT used in any
    prior submission per explicit earlier decision -- tested here for
    the first time)
  - the 2x2 combination of both

Metrics: DSC (dice), NSD (1mm tolerance, matches the official FOMO26
metric), HD95 (mm) -- all computed on CUTOUT_MODDROP's own fold-best.ckpt
models, on each fold's genuinely held-out subjects (same methodology used
throughout this session for internal CV proxies of official leaderboard
behavior)."""
import sys
sys.path.insert(0, '.')
from test_time_augmentation_sensitivity import *
import csv
import numpy as np
from scipy import ndimage
from monai.metrics import compute_hausdorff_distance

CHANNEL_GAMMAS = {0: 1.50, 1: 1.0, 2: 1.75}  # flair, dwi_b1000, swi_or_t2s


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


def largest_cc(pred_hard_np):
    """Keep only the largest connected component (26-connectivity) of the
    foreground mask; empty mask passes through unchanged."""
    if pred_hard_np.sum() == 0:
        return pred_hard_np
    structure = np.ones((3, 3, 3), dtype=int)
    labeled, n = ndimage.label(pred_hard_np, structure=structure)
    if n <= 1:
        return pred_hard_np
    sizes = ndimage.sum(pred_hard_np, labeled, range(1, n + 1))
    keep = np.argmax(sizes) + 1
    return (labeled == keep).astype(pred_hard_np.dtype)


def dice_from_hard(pred_np, gt_np):
    tp = float(((pred_np == 1) & (gt_np == 1)).sum())
    fp = float(((pred_np == 1) & (gt_np == 0)).sum())
    fn = float(((pred_np == 0) & (gt_np == 1)).sum())
    if tp + fp + fn == 0:
        return 1.0  # both empty -> perfect match
    return 2 * tp / (2 * tp + fp + fn)


def hd95_from_hard(pred_np, gt_np):
    if pred_np.sum() == 0 and gt_np.sum() == 0:
        return 0.0
    if pred_np.sum() == 0 or gt_np.sum() == 0:
        return float('nan')  # undefined -- one side has no surface
    pred_t = torch.as_tensor(pred_np).long().unsqueeze(0)
    gt_t = torch.as_tensor(gt_np).long().unsqueeze(0)
    pred_onehot = F.one_hot(pred_t, num_classes=2).permute(0, 4, 1, 2, 3).float()
    gt_onehot = F.one_hot(gt_t, num_classes=2).permute(0, 4, 1, 2, 3).float()
    hd = compute_hausdorff_distance(pred_onehot, gt_onehot, include_background=False,
                                     percentile=95, spacing=[1.0, 1.0, 1.0])
    return float(hd[0, 0])


cfg = VARIANTS[VARIANT]  # CUTOUT_MODDROP
best_epochs = per_fold_best_epochs(VARIANT)
model = FomoStudentSegNet(input_channels=3, output_channels=2, checkpoint_path=PRETRAIN,
                           teacher_name='brats', from_scratch=False, stem_mode='learnable',
                           freeze_decoder=False).eval().to(DEVICE)

CONDITIONS = ['baseline', 'gamma', 'lcc', 'gamma+lcc']
results = []

for fold in range(10):
    epoch = best_epochs[fold]
    step = (epoch + 1) * 250
    seed = cfg['seed_base'] + fold
    d = run_dir(VARIANT, fold, seed)
    ckpt = d / f'milestone_step={step:06d}.ckpt'
    if not ckpt.exists():
        continue
    load_state(model, ckpt)
    model.eval()
    subjects = held_out_subjects(cfg['test_prefix'], fold)
    for subject in subjects:
        batch = make_batch(subject, cfg['pre_dir'])
        image0 = batch['image'][0]
        label0 = batch['label'][0]
        gt_np = label0.squeeze().cpu().numpy().astype('uint8')

        def predict_hard(img):
            x = img.unsqueeze(0).to(DEVICE)
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model.sliding_window_predict(x, patch_size=PATCH_SIZE, overlap=0.5, mirror=False)
            return logits.argmax(1)[0].cpu().numpy().astype('uint8')

        pred_base = predict_hard(image0)
        pred_gamma = predict_hard(gamma_transform_channels(image0, CHANNEL_GAMMAS))
        pred_lcc = largest_cc(pred_base)
        pred_gamma_lcc = largest_cc(pred_gamma)

        for tag, pred in [('baseline', pred_base), ('gamma', pred_gamma),
                           ('lcc', pred_lcc), ('gamma+lcc', pred_gamma_lcc)]:
            dice = dice_from_hard(pred, gt_np)
            nsd = surface_dice(pred, gt_np)
            hd95 = hd95_from_hard(pred, gt_np)
            results.append({'fold': fold, 'subject': subject, 'condition': tag,
                             'dice': dice, 'nsd': nsd, 'hd95': hd95})
        print(f'fold={fold} subject={subject} '
              f'base=({dice_from_hard(pred_base,gt_np):.3f}) '
              f'gamma=({dice_from_hard(pred_gamma,gt_np):.3f}) '
              f'lcc=({dice_from_hard(pred_lcc,gt_np):.3f}) '
              f'gamma+lcc=({dice_from_hard(pred_gamma_lcc,gt_np):.3f})', flush=True)

with open('output/comprehensive_before_after.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)

from collections import defaultdict
by_cond = defaultdict(lambda: {'dice': [], 'nsd': [], 'hd95': []})
for r in results:
    b = by_cond[r['condition']]
    b['dice'].append(r['dice'])
    b['nsd'].append(r['nsd'])
    if not (r['hd95'] != r['hd95']):  # not NaN
        b['hd95'].append(r['hd95'])

print()
print(f"{'condition':12s} {'mean_dice':>10s} {'mean_nsd':>10s} {'mean_hd95':>11s} {'n_hd95_defined':>15s}")
for tag in CONDITIONS:
    b = by_cond[tag]
    md = sum(b['dice']) / len(b['dice'])
    mn = sum(b['nsd']) / len(b['nsd'])
    mh = sum(b['hd95']) / len(b['hd95']) if b['hd95'] else float('nan')
    print(f"{tag:12s} {md:10.4f} {mn:10.4f} {mh:11.4f} {len(b['hd95']):15d}")
print('Wrote output/comprehensive_before_after.csv')
