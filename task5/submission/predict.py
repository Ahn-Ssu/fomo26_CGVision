#!/usr/bin/env python3
"""FOMO26 Task 5 submission -- affine_iso10 + LV(lateral ventricle) fixed
masking, 30-epoch, 5-fold ensemble (2026-08-30).

Background: LR-asymmetry group analysis + SmoothGrad saliency on the
un-masked affine_iso10 classifier both independently pointed to the same
problem -- its discriminative signal concentrates on the lateral-ventricle
boundary, not the cortex, where the actual PMG pathology (a gray-matter
cortical-folding malformation) lives. A per-subject quantitative check
(Welch/Mann-Whitney, n=24 PMG vs 24 control) found NO statistically
significant ventricle-region asymmetry difference either -- whatever the
un-masked model was using inside that region wasn't even a robust
population-level signal, consistent with the real leaderboard score for
that variant being near/below random (0.396) despite locally-strong
pooled AUROC (~0.83).

Fix: during TRAINING, a fixed lateral-ventricle + periventricular bounding
box (x=[49,144], y=[55,172], z=[57,134] in this variant's 192x224x192
grid -- derived from a 48-subject population CSF-consistency mask, eroded
to drop the thin 3rd-ventricle/interhemispheric-fissure sliver, then
expanded by a 15-voxel margin) is ALWAYS corrupted with local-mean/std-
matched Gaussian noise, denying the model that shortcut and forcing it
onto the cortex. Confirmed via matched-masked evaluation (same masking
applied at eval time): official-style pooled TEST AUROC 0.6406 (clean,
mismatched distribution) -> 0.8681 (10ep) -> 0.8819 (20ep) -> 0.9010
(30ep), monotonically improving with NO overfitting (the un-masked
recipe, by contrast, peaked at 10ep and got WORSE by 20ep: 0.8299 ->
0.8142) -- the masking itself acts as a strong enough regularizer that
this small (n=48) dataset supports much longer training. SmoothGrad under
matched-masked evaluation confirmed attention genuinely shifts to the
cortical rim around the masked box, both for a single subject and
independently across 5 different PMG subjects (one per fold's own
held-out model).

Because the mask is filled with FRESH random noise each call, inference
averages N_TTA=8 independent draws per view (not a single draw) to match
how this was validated and reduce prediction variance. A subsequent
augmentation-sensitivity probe (seeded, cross-validated across two
different seeds) found LR-mirroring the input (before masking) ALSO gives
a small, consistent positive effect on top of the masking itself (+0.0017
to +0.0174 across the two seeds tested, at 20ep and 30ep respectively) --
so each fold's final probability is an average over 16 views: 8 masked
draws of the original orientation + 8 masked draws of the LR-mirrored
orientation. All 5 folds' 16-view-averaged probabilities are then
averaged together (80 forward passes total per subject) for the final
ensemble prediction -- note this outer 5-fold ensembling step itself
could NOT be locally validated against held-out data (every one of our 48
labeled subjects was in 4 of the 5 folds' TRAINING sets, so a naive
all-5-average on our own data leaks), but is safe and standard for
genuinely new subjects at real deployment time (same approach as the
Task1 champion container's fold ensemble).

Rotation was also tested as a TTA candidate but deliberately excluded:
mild rotation (+-10 deg) showed a positive effect similar in size to
mirroring, but moderate rotation (+-20 deg) reversed to a small negative
effect -- consistent with this variant's data already being affine-
registered (well-aligned) and the LV bbox being FIXED-coordinate, so
larger rotations risk shifting the true ventricle out from under the
mask. Given a fixed-magnitude rotation TTA candidate is inherently harder
to bound safely than a simple LR flip, it was not included here.

Preprocessing (unchanged from the un-masked affine_iso10 container): HD-BET
skull-strip, ANTs RIGID init -> chained AFFINE registration to
MNI152NLin2009cAsym, warped onto the fixed template-anchored 1mm crop box,
padded to (192, 224, 192). See preprocess.py for the (also unchanged)
error-handling cascade -- as of 2026-08-30 it degrades ONCE (registration
failure -> no-registration resample fallback) and then crashes on purpose
if that also fails, rather than silently returning an all-zero volume,
per the same policy change applied project-wide (see that file's
docstring for why)."""
import argparse
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "model"))

from preprocess import preprocess_task5
from net import ChampionTask1Net

N_FOLDS = 5
FOLD_CHECKPOINT_DIR = Path(__file__).parent / "model" / "fold_checkpoints"
TEACHER_NAME = "brats"

# Fixed LV+periventricular bbox in this variant's (192,224,192) grid --
# x0,x1,y0,y1,z0,z1, inclusive. Must exactly match training's
# _LV_BBOX_AFFINE (asparagus/modules/transforms/presets/train.py).
LV_BBOX = (49, 144, 55, 172, 57, 134)
N_TTA = 8  # independent noise-mask draws averaged per orientation


def lv_fixed_mask(x: torch.Tensor) -> torch.Tensor:
    """Standalone port of Torch_LVFixedMask (no asparagus dependency, see
    net.py's docstring for why submission containers don't import it).
    x: (C, X, Y, Z), single sample, no batch dim. Returns a NEW tensor
    with LV_BBOX corrupted by local-mean/std-matched Gaussian noise,
    clamped to the original image's own value range."""
    out = x.clone()
    x0, x1, y0, y1, z0, z1 = LV_BBOX
    vlo, vhi = out.min(), out.max()
    patch = out[..., x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
    local_mean = patch.mean()
    local_std = patch.std().clamp_min(1e-3)
    noise = (torch.randn_like(patch) * local_std + local_mean).clamp(vlo, vhi)
    out[..., x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = noise
    return out


def mirror_lr(x: torch.Tensor) -> torch.Tensor:
    """LR flip. x: (C, X, Y, Z) -- X (dim 1, the first spatial axis) is
    the confirmed left-right axis for this variant (validated via direct
    raw-data flip-difference analysis, 2026-08-29)."""
    return torch.flip(x, dims=[1])


def parse_args():
    parser = argparse.ArgumentParser(description="FOMO26 Task 5 - affine_iso10 + LV masking, 5-fold ensemble")
    parser.add_argument("--t1", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


@torch.no_grad()
def predict(args) -> float:
    # Fixed seed (2026-08-31): the LV mask is filled with fresh random
    # noise every call (by design -- N_TTA=8 draws are averaged per view
    # to match how this was validated), so without a fixed seed, two runs
    # on the IDENTICAL input give slightly different probabilities
    # (observed: 0.352 vs 0.334 on a real subject, two separate `apptainer
    # run` invocations). That's expected and shouldn't move AUROC (rank-
    # based, and the real harness scores each subject exactly once
    # anyway), but a submission should still be reproducible on rerun.
    torch.manual_seed(42)
    # torch.manual_seed alone wasn't enough on GPU (real-data retest still
    # showed 0.330 vs 0.325 across two separate `apptainer run` processes,
    # down from 0.352/0.334 but not eliminated) -- cuDNN's convolution
    # algorithm autotuner can pick slightly different (numerically
    # non-identical) algorithms across process invocations independent of
    # the RNG seed. Pinning it closes that second source of nondeterminism.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.TemporaryDirectory() as work_dir:
        image = preprocess_task5(args.t1, work_dir)

    net = ChampionTask1Net(input_channels=1, output_channels=2, teacher_name=TEACHER_NAME, dropout_rate=0.3)
    net.to(device)
    net.eval()

    fold_probs = []
    for fold in range(N_FOLDS):
        ckpt_path = FOLD_CHECKPOINT_DIR / f"fold{fold}.pt"
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        incompatible = net.load_state_dict(sd, strict=True)
        assert not incompatible.missing_keys and not incompatible.unexpected_keys, incompatible

        view_probs = []
        for _ in range(N_TTA):
            masked = lv_fixed_mask(image)
            x = masked.to(device).unsqueeze(0)
            view_probs.append(F.softmax(net(x), dim=1)[0, 1].item())
        mirrored = mirror_lr(image)
        for _ in range(N_TTA):
            masked = lv_fixed_mask(mirrored)
            x = masked.to(device).unsqueeze(0)
            view_probs.append(F.softmax(net(x), dim=1)[0, 1].item())

        fold_probs.append(sum(view_probs) / len(view_probs))

    ensemble_prob = sum(fold_probs) / len(fold_probs)
    assert 0.0 <= ensemble_prob <= 1.0
    return ensemble_prob


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    probability = predict(args)

    subject_id = output_path.stem
    output_file = output_path.parent / f"{subject_id}.txt"
    with open(output_file, "w") as f:
        f.write(f"{probability:.3f}")

    print(f"[predict] {subject_id}: p = {probability:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
