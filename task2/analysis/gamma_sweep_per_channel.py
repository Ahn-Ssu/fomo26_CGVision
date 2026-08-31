"""Per-channel gamma sweep (user request 2026-08-26): darken ONE modality
channel at a time (holding the other 2 at gamma=1.0), across the same
CUTOUT_MODDROP fold-best.ckpt models used for the uniform-gamma sweep --
tests whether flair/dwi_b1000/swi_or_t2s each have a different optimal
darkening magnitude (direction is unified to darkening only, per user's
own observation: lesion is consistently higher-contrast/brighter across
all 3 channels in Task 2 -- confirmed earlier via direct measurement:
flair 2.38x, dwi_b1000 2.89x, swi_or_t2s 2.82x lesion:normal-tissue
intensity ratio -- so only the magnitude, not the direction, needs
per-channel tuning)."""
import sys
sys.path.insert(0, '.')
from test_time_augmentation_sensitivity import *
import csv

GAMMA_VALUES = [1.0, 1.25, 1.5, 1.75, 2.0]
CHANNEL_NAMES = ["flair", "dwi_b1000", "swi_or_t2s"]


def gamma_transform_single_channel(image, gamma, channel_idx):
    out = image.clone()
    ch = out[channel_idx]
    lo, hi = ch.min(), ch.max()
    rng = (hi - lo).clamp_min(1e-7)
    norm = (ch - lo) / rng
    out[channel_idx] = norm.pow(gamma) * rng + lo
    return out


cfg = VARIANTS[VARIANT]
best_epochs = per_fold_best_epochs(VARIANT)
model = FomoStudentSegNet(input_channels=3, output_channels=2, checkpoint_path=PRETRAIN,
                           teacher_name='brats', from_scratch=False, stem_mode='learnable',
                           freeze_decoder=False).eval().to(DEVICE)

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
        for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
            for gamma in GAMMA_VALUES:
                img_p = gamma_transform_single_channel(image0, gamma, ch_idx) if gamma != 1.0 else image0
                x = img_p.unsqueeze(0).to(DEVICE)
                with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model.sliding_window_predict(x, patch_size=PATCH_SIZE, overlap=0.5, mirror=False)
                gt = label0.unsqueeze(0).to(DEVICE)
                dice, tp, fp, fn = hard_dice(logits, gt)
                pred = logits.argmax(1)[0].cpu().numpy().astype('uint8')
                gt_np = label0.squeeze().cpu().numpy().astype('uint8')
                nsd = surface_dice(pred, gt_np)
                results.append({'fold': fold, 'subject': subject, 'channel': ch_name, 'gamma': gamma, 'dice': dice, 'nsd': nsd})
                print(f'fold={fold} subject={subject} channel={ch_name:10s} gamma={gamma:.2f} dice={dice:.4f} nsd={nsd:.4f}', flush=True)

with open('output/gamma_sweep_per_channel.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)

from collections import defaultdict
by_ch_gamma = defaultdict(lambda: {'dice': [], 'nsd': []})
for r in results:
    b = by_ch_gamma[(r['channel'], r['gamma'])]
    b['dice'].append(r['dice']); b['nsd'].append(r['nsd'])

print()
print(f"{'channel':10s} {'gamma':>6s} {'mean_dice':>10s} {'mean_nsd':>10s}")
for ch_name in CHANNEL_NAMES:
    for gamma in GAMMA_VALUES:
        b = by_ch_gamma[(ch_name, gamma)]
        md = sum(b['dice']) / len(b['dice'])
        mn = sum(b['nsd']) / len(b['nsd'])
        print(f"{ch_name:10s} {gamma:6.2f} {md:10.4f} {mn:10.4f}")
    print()
print('Wrote output/gamma_sweep_per_channel.csv')
