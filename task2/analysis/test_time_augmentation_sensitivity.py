"""Test-time perturbation sensitivity analysis (user request 2026-08-24):
for the winning submission recipe (CUTOUT_MODDROP, fold-best.ckpt), measure
how much DSC/NSD drop when EACH training-time augmentation is applied, one
at a time, at test time on the held-out fold subjects -- answers "how
robust is the final model to this specific perturbation", not "did adding
this augmentation to training help" (that would need leave-one-out
retraining, explicitly NOT what the user asked for this round -- see
extract_weights_bestckpt.py for the fold-best selection this reuses).

Bidirectional augmentations are split into their two opposing arms per
user instruction ("gamma는 밝아질때와 어두워질때를 구분", "서로 상반된
방향을 커버하는 경우에도 구분"):
  - Gamma: brighten (gamma=0.9) vs darken (gamma=1.1) -- implemented via
    direct power-law math (bypasses gardening_tools' torch_gamma, which
    has a quirky random branch for gamma_range[0]<1 that prevents an exact
    deterministic value from a degenerate (g,g) range).
  - Spatial scale: zoom-in (factor=1.1) vs zoom-out (factor=0.9) -- via
    Torch_Spatial(p_scale_all_channel=1.0, p_rot_all_channel=0.0), which
    (unlike Gamma) samples degenerate ranges correctly deterministically.
  Rotation and mirror are NOT split -- no directional-asymmetry rationale
  (CW/CCW rotation and flipped/not-flipped don't have an anatomically
  meaningful "opposite effect" the way brighten/darken or zoom in/out do).

All OTHER (single-direction) perturbations use the functional forms
directly from gardening_tools.functional.transforms (fixed scalar params,
no probability gating) at a representative magnitude drawn from the
training config's own range (usually the more aggressive end, to make the
sensitivity signal clearly visible against noise).

Geometric perturbations (rotation, scale x2, mirror) warp BOTH image and
label together (Torch_Spatial/Torch_Mirror's label_key= mechanism) so DSC/
NSD are computed in a consistently-transformed space -- comparing a warped
prediction against the ORIGINAL unwarped label would just measure
misalignment, not perturbation robustness."""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics import compute_surface_dice

sys.path.insert(0, "/root/fomo26_segm_mk5")
sys.path.insert(0, "/root/fomo26_segm_mk5/segmentation")
sys.path.insert(0, "/root/fomo26_segm_mk5/segmentation/container")
sys.path.insert(0, "/root/fomo26_segm_mk5/task2_archsearch")

from evaluate_10foldcv_trajectory import VARIANTS, held_out_subjects, load_state, run_dir, ROOT
from evaluate_archsearch_trajectory import PRETRAIN, hard_dice
from extract_weights_bestckpt import per_fold_best_epochs
from asparagus.functional.collate import collate_return
from asparagus.modules.datasets.TrainDataset import SegTestDataset
from asparagus.modules.networks.fomo26_student import FomoStudentSegNet
from asparagus.modules.transforms.presets.train import CPU_seg_test_transforms
from asparagus.modules.transforms.cutout import Torch_Cutout3D
from asparagus.modules.transforms.modality_dropout import Torch_ModalityDropout

from gardening_tools.modules.transforms.spatial import Torch_Spatial
from gardening_tools.modules.transforms.mirror import Torch_Mirror
from gardening_tools.functional.transforms.blur import torch_blur
from gardening_tools.functional.transforms.bias_field import torch_bias_field
from gardening_tools.functional.transforms.motion_ghosting import torch_motion_ghosting
from gardening_tools.functional.transforms.ringing import torch_gibbs_ringing
from gardening_tools.functional.transforms.sampling import torch_simulate_lowres
from gardening_tools.functional.transforms.noise import torch_additive_noise, torch_multiplicative_noise

VARIANT = "CUTOUT_MODDROP"
DEVICE = torch.device("cuda")
PATCH_SIZE = (128, 128, 128)
NSD_TOLERANCE_MM = 1.0
OUT_CSV = Path("/root/fomo26_segm_mk5/task2_archsearch/output/tta_sensitivity_cutout_moddrop.csv")


def make_batch(subject, pre_dir):
    pre = ROOT / "asparagus_data/SEG009i_FOMO26_Meningioma_ISO1MM" / pre_dir
    path = str(pre / subject / "ses-01" / "image.pt")
    transform = CPU_seg_test_transforms(PATCH_SIZE, normalize=False)
    return collate_return([SegTestDataset([path], transforms=transform)[0]])


def surface_dice(pred_hard, gt_hard):
    pred_t = torch.as_tensor(pred_hard).long().unsqueeze(0)
    gt_t = torch.as_tensor(gt_hard).long().unsqueeze(0)
    pred_onehot = F.one_hot(pred_t, num_classes=2).permute(0, 4, 1, 2, 3).float()
    gt_onehot = F.one_hot(gt_t, num_classes=2).permute(0, 4, 1, 2, 3).float()
    nsd = compute_surface_dice(
        pred_onehot, gt_onehot, class_thresholds=[NSD_TOLERANCE_MM],
        include_background=False, spacing=[1.0, 1.0, 1.0],
    )
    return float(nsd[0, 0])


def gamma_transform(image, gamma):
    """Direct power-law gamma (per-channel min/max), bypassing
    torch_gamma's quirky degenerate-range branch -- see module docstring."""
    out = image.clone()
    for c in range(out.shape[0]):
        ch = out[c]
        lo, hi = ch.min(), ch.max()
        rng = (hi - lo).clamp_min(1e-7)
        norm = (ch - lo) / rng
        out[c] = norm.pow(gamma) * rng + lo
    return out


# ---------- perturbation registry: name -> fn(image, label) -> (image', label') ----------
def perturb_none(image, label):
    return image, label


def perturb_blur(image, label):
    out = image.clone()
    for c in range(out.shape[0]):
        out[c] = torch_blur(out[c], sigma=0.7, clip_to_input_range=True)
    return out, label


def perturb_bias_field(image, label):
    out = image.clone()
    for c in range(out.shape[0]):
        out[c] = torch_bias_field(out[c], clip_to_input_range=False)
    return out, label


def perturb_gamma_bright(image, label):
    return gamma_transform(image, gamma=0.9), label


def perturb_gamma_dark(image, label):
    return gamma_transform(image, gamma=1.1), label


def perturb_motion_ghosting(image, label):
    out = image.clone()
    for c in range(out.shape[0]):
        out[c] = torch_motion_ghosting(out[c], alpha=0.9, num_reps=3, axis=0, clip_to_input_range=False)
    return out, label


def perturb_gibbs_ringing(image, label):
    out = image.clone()
    for c in range(out.shape[0]):
        out[c] = torch_gibbs_ringing(out[c], num_sample=112, mode="rect", axes=[0, 1, 2], clip_to_input_range=False)
    return out, label


def perturb_simulate_lowres(image, label):
    out = image.clone()
    for c in range(out.shape[0]):
        shape = out[c].shape
        target = tuple(max(1, int(round(s * 0.75))) for s in shape)
        out[c] = torch_simulate_lowres(out[c], target_shape=target, clip_to_input_range=True)
    return out, label


def perturb_multiplicative_noise(image, label):
    out = image.clone()
    for c in range(out.shape[0]):
        out[c] = torch_multiplicative_noise(out[c], mean=0.0, sigma=5e-4, clip_to_input_range=False)
    return out, label


def perturb_additive_noise(image, label):
    out = image.clone()
    for c in range(out.shape[0]):
        out[c] = torch_additive_noise(out[c], mean=0.0, sigma=5e-4, clip_to_input_range=False)
    return out, label


def perturb_cutout(image, label):
    cutout = Torch_Cutout3D(p_per_sample=1.0, size_range=(0.2, 0.2), n_cuts=1, batched=False)
    d = cutout({"image": image.clone()})
    return d["image"], label


def perturb_modality_dropout(image, label):
    moddrop = Torch_ModalityDropout(p_per_sample=1.0, channels=(2,), batched=False)
    d = moddrop({"image": image.clone()})
    return d["image"], label


def perturb_rotation(image, label):
    sp = Torch_Spatial(
        crop=False, patch_size=(1, 1, 1), interpolation_mode="bilinear",
        p_deform_all_channel=0.0,
        p_rot_all_channel=1.0, p_rot_per_axis=1.0,
        x_rot_in_degrees=(0.0, 0.0), y_rot_in_degrees=(0.0, 0.0), z_rot_in_degrees=(15.0, 15.0),
        p_scale_all_channel=0.0, scale_factor=(1.0, 1.0),
        skip_label=False,
    )
    d = sp({"image": image.clone().unsqueeze(0), "label": label.clone().unsqueeze(0)})
    return d["image"][0], d["label"][0]


def perturb_scale_in(image, label):
    sp = Torch_Spatial(
        crop=False, patch_size=(1, 1, 1), interpolation_mode="bilinear",
        p_deform_all_channel=0.0,
        p_rot_all_channel=0.0, p_rot_per_axis=0.0,
        x_rot_in_degrees=(0.0, 0.0), y_rot_in_degrees=(0.0, 0.0), z_rot_in_degrees=(0.0, 0.0),
        p_scale_all_channel=1.0, scale_factor=(1.1, 1.1),
        skip_label=False,
    )
    d = sp({"image": image.clone().unsqueeze(0), "label": label.clone().unsqueeze(0)})
    return d["image"][0], d["label"][0]


def perturb_scale_out(image, label):
    sp = Torch_Spatial(
        crop=False, patch_size=(1, 1, 1), interpolation_mode="bilinear",
        p_deform_all_channel=0.0,
        p_rot_all_channel=0.0, p_rot_per_axis=0.0,
        x_rot_in_degrees=(0.0, 0.0), y_rot_in_degrees=(0.0, 0.0), z_rot_in_degrees=(0.0, 0.0),
        p_scale_all_channel=1.0, scale_factor=(0.9, 0.9),
        skip_label=False,
    )
    d = sp({"image": image.clone().unsqueeze(0), "label": label.clone().unsqueeze(0)})
    return d["image"][0], d["label"][0]


def perturb_mirror_lr(image, label):
    mirror = Torch_Mirror(p_per_sample=1.0, axes=(0,), p_mirror_per_axis=1.0, skip_label=False)
    d = mirror({"image": image.clone().unsqueeze(0), "label": label.clone().unsqueeze(0)})
    return d["image"][0], d["label"][0]


PERTURBATIONS = {
    "none (baseline)": perturb_none,
    "blur": perturb_blur,
    "bias_field": perturb_bias_field,
    "gamma_brighten": perturb_gamma_bright,
    "gamma_darken": perturb_gamma_dark,
    "motion_ghosting": perturb_motion_ghosting,
    "gibbs_ringing": perturb_gibbs_ringing,
    "simulate_lowres": perturb_simulate_lowres,
    "multiplicative_noise": perturb_multiplicative_noise,
    "additive_noise": perturb_additive_noise,
    "rotation_15deg": perturb_rotation,
    "scale_zoom_in": perturb_scale_in,
    "scale_zoom_out": perturb_scale_out,
    "mirror_lr_flip": perturb_mirror_lr,
    "cutout3d": perturb_cutout,
    "modality_dropout": perturb_modality_dropout,
}


def main():
    cfg = VARIANTS[VARIANT]
    best_epochs = per_fold_best_epochs(VARIANT)

    model = FomoStudentSegNet(input_channels=3, output_channels=2, checkpoint_path=PRETRAIN,
                               teacher_name="brats", from_scratch=False, stem_mode="learnable",
                               freeze_decoder=False).eval().to(DEVICE)

    rows = []
    for fold in range(10):
        epoch = best_epochs[fold]
        step = (epoch + 1) * 250
        seed = cfg["seed_base"] + fold
        d = run_dir(VARIANT, fold, seed)
        ckpt = d / f"milestone_step={step:06d}.ckpt"
        if not ckpt.exists():
            print(f"[SKIP] fold={fold}: missing {ckpt}")
            continue
        load_state(model, ckpt)
        model.eval()

        subjects = held_out_subjects(cfg["test_prefix"], fold)
        for subject in subjects:
            batch = make_batch(subject, cfg["pre_dir"])
            image0 = batch["image"][0]  # (C,D,H,W)
            label0 = batch["label"][0]  # (1,D,H,W)

            for pname, pfn in PERTURBATIONS.items():
                img_p, lbl_p = pfn(image0, label0)
                x = img_p.unsqueeze(0).to(DEVICE)
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model.sliding_window_predict(x, patch_size=PATCH_SIZE, overlap=0.5, mirror=False)
                gt = lbl_p.unsqueeze(0).to(DEVICE)
                dice, tp, fp, fn = hard_dice(logits, gt)
                pred = logits.argmax(1)[0].cpu().numpy().astype("uint8")
                gt_np = lbl_p.squeeze().cpu().numpy().astype("uint8")
                nsd = surface_dice(pred, gt_np)

                rows.append({"fold": fold, "epoch": epoch, "subject": subject, "perturbation": pname, "dice": dice, "nsd": nsd})
                print(f"fold={fold} subject={subject} pert={pname:22s} dice={dice:.4f} nsd={nsd:.4f}", flush=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    from collections import defaultdict
    by_pert = defaultdict(lambda: {"dice": [], "nsd": []})
    for r in rows:
        by_pert[r["perturbation"]]["dice"].append(r["dice"])
        by_pert[r["perturbation"]]["nsd"].append(r["nsd"])

    baseline_dice = np.mean(by_pert["none (baseline)"]["dice"])
    baseline_nsd = np.mean(by_pert["none (baseline)"]["nsd"])

    print(f"\n=== TTA sensitivity, {VARIANT} fold-best.ckpt, n={len(rows)//len(PERTURBATIONS)} subjects x {len(PERTURBATIONS)} perturbations ===")
    print(f"{'perturbation':22s} {'mean_dice':>10s} {'d_dice':>9s} {'mean_nsd':>10s} {'d_nsd':>9s}")
    for pname in PERTURBATIONS:
        md = np.mean(by_pert[pname]["dice"])
        mn = np.mean(by_pert[pname]["nsd"])
        print(f"{pname:22s} {md:10.4f} {md-baseline_dice:+9.4f} {mn:10.4f} {mn-baseline_nsd:+9.4f}")

    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
