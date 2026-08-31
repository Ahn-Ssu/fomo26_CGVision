#!/usr/bin/env python3
"""
FOMO26 Challenge - Task 4: Multiclass Tissue Segmentation

Best-of-fold ensemble of 5 fomo26_student_seg checkpoints (encoder frozen,
decoder trained from scratch) -- for each of Task_4's 5 CV folds, whichever
pretrained teacher (brats vs vesselfm) scored higher val Dice was picked
(see checkpoints/manifest.json): brats won folds 1/2/4, vesselfm won
folds 0/3. anatomix+brains never won a fold and is not part of this
ensemble.

Pipeline per case (see the top-level task4/README.md for the full design
rationale):
  1. preprocess.py: HD-BET skull-strip -> ANTs rigid+affine registration to
     MNI152 -> resample onto the 256^3 0.5mm-iso ROI grid the models were
     trained on (single interpolation hop from the untouched --t2 input).
  2. Run all 5 checkpoints' sliding-window inference on that ROI (under the
     SAME bf16-mixed autocast precision asparagus's Lightning Trainer used
     for training/val/test -- see common/asparagus/configs/hardware/*.yaml
     `precision: "bf16-mixed"`; plain fp32 inference here would be a silent
     train/inference precision mismatch), with full 8-way mirror TTA
     (`mirror=True`, averaging over all flip combinations of the 3 spatial
     axes -- matches `SegmentationModule.test_step`'s own `mirror_tta=True`
     class default, which every "officially" reported fold Dice number this
     whole project was actually computed under; measured on unseen val
     subjects across all 15 (teacher, fold) checkpoints: +0.0114 mean Dice
     over no-TTA, 14/15 folds improved, a larger and more consistent gain
     than LR-flip-only TTA's +0.0050), then average all 5 checkpoints'
     softmax PROBABILITIES (standard k-fold ensembling: each fold's model
     saw a different 32/8 train/val split of the same subjects) -- kept as
     probabilities, NOT argmaxed yet.
  3. Inverse-transform EACH class's probability channel separately (Linear
     interpolation, ANTs' own ITK-based antsApplyTransforms -- physical-
     space-correct by construction) into the original --t2 scan's exact
     grid, THEN argmax in that native space. Argmaxing the ROI-space hard
     label first and NearestNeighbor-resampling *that* (the earlier
     approach) throws away the sub-voxel confidence gradient the
     probability field carries across the boundary and produces blockier
     NN-staircase edges -- confirmed empirically: a hard-label round-trip
     (warp a known label to ROI space and directly back) loses ~0.5-0.7%
     Dice to double-NN quantization alone (measured 0.9926-0.9952 on a real
     subject), which per-class probability inversion avoids.

Output convention (matches the FOMO26 container-validator contract):
  0 = background, 1 = structure_1, 2 = structure_2

Robustness: a sibling FOMO26 Task_3 container hit a real production
incident where a hidden test subject's HD-BET output was an empty mask,
which crashed ANTs registration downstream and lost that subject's
prediction entirely -- see preprocess.py's is_blank_image/hd_bet_safe/
register_safe for the corresponding defenses (retrying with a raw,
non-skull-stripped image is a genuine attempt at a real answer via an
alternate method, kept as-is).

This used to be wrapped in a top-level try/except that converted ANY
failure into a silent all-zero prediction. That was removed after it
directly caused a real submission to silently score 0: a $HOME/
network-access bug produced an all-zero output that looked structurally
valid to the local container-validator and took real effort to root-cause,
when the underlying problem should have crashed loudly instead. FOMO26's
evaluation harness marks a crashing container INVALID quickly, which is a
far more useful, fast diagnostic signal than a silently-wrong "score" that's
indistinguishable from a genuinely bad-but-real prediction.

So now: ONLY a genuinely blank/degenerate input (mean and std both ~0,
checked in preprocess_safe before HD-BET is even attempted -- a deliberate,
correct response, since a blank scan legitimately has no foreground) gets
an intentional all-background output. Every other failure (registration
failing on every candidate, a missing dependency, any other unexpected bug)
is NOT caught -- it propagates as a real exception, crashing the process
with a full traceback and non-zero exit code.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# CRITICAL, set before any other import: Apptainer (unlike Docker) preserves
# the invoking HOST user's HOME inside the container. HD_BET.paths hardcodes
# ~/hd-bet_params with no env-var override, so if runtime HOME != build-time
# HOME (/root, since %post runs as root), the bundled weights below go
# unfound and HD-BET tries to re-download them -- a real sibling FOMO26
# container was marked INVALID by the organizers for "attempts to access the
# internet" from exactly this mismatch. Setting os.environ["HOME"] directly,
# in Python, at the very top of this process guarantees it for THIS process
# and everything it inherits from (subprocess calls, matplotlib/CUDA lazy
# cache init) regardless of whether Apptainer.def's `%environment` took
# effect -- no dependency on the outer container/shell behaving as expected.
os.environ["HOME"] = "/root"

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR / "model"))

from preprocess import preprocess_safe  # noqa: E402
from postprocess import inverse_transform  # noqa: E402

# Deliberately NOT `pip install`-ing the full asparagus package: net.py's
# own imports are just torch + gardening_tools + this container's own
# student.py -- it never touches any other asparagus module. Installing the
# whole asparagus package pulls in lightning, hydra-core, omegaconf, and
# wandb as hard dependencies (asparagus's own pyproject.toml), none of which
# this container's code path ever imports, and can silently downgrade the
# base image's own CUDA-matched torch build via asparagus's own pinned
# torch/torchvision versions. Vendoring just this one file (+ student.py)
# removes that whole class of risk at the source, rather than relying on it
# being merely unreachable from this process's import graph.
from net import FomoStudentSegNet  # noqa: E402
from gardening_tools.functional.normalization import volume_wise_znorm  # noqa: E402

TEMPLATE = APP_DIR / "assets" / "MNI152_T1_1mm_brain.nii.gz"
ROI_REFERENCE = APP_DIR / "assets" / "roi_reference_iso05.nii.gz"
CHECKPOINT_DIR = APP_DIR / "checkpoints"
MANIFEST_PATH = CHECKPOINT_DIR / "manifest.json"
PRETRAIN_CHECKPOINT = CHECKPOINT_DIR / "pretrain_step_280000.pt"

PATCH_SIZE = [128, 128, 128]
N_CLASSES = 3  # background + 2 lesion types


def parse_args():
    parser = argparse.ArgumentParser(
        description="FOMO26 Task 4 Multiclass Segmentation (best-of-fold 5-model ensemble)"
    )
    parser.add_argument("--t2", type=str, required=True, help="Path to T2-weighted image")
    parser.add_argument("--output", type=str, required=True, help="Path to save segmentation NIfTI")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto-detect)")
    return parser.parse_args()


def load_fold_model(teacher_name: str, trained_ckpt_path: Path, device: str) -> FomoStudentSegNet:
    model = FomoStudentSegNet(
        input_channels=1, output_channels=N_CLASSES,
        checkpoint_path=str(PRETRAIN_CHECKPOINT), teacher_name=teacher_name,
        from_scratch=False, stem_mode="frozen",
        freeze_decoder=False, encoder_convpass_frozen=True, decoder_scheme="scratch",
    )
    ckpt = torch.load(trained_ckpt_path, map_location="cpu", weights_only=False)
    model_state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    current_keys = set(model.state_dict().keys())
    loaded_keys = set(model_state.keys())
    assert current_keys == loaded_keys, (
        f"key mismatch loading {trained_ckpt_path}: "
        f"missing={current_keys - loaded_keys} unexpected={loaded_keys - current_keys}"
    )
    model.load_state_dict(model_state)
    model.to(device).eval()
    return model


def ensemble_predict_roi(roi_path: Path, manifest: list, device: str):
    """Returns mean class-probability volume [N_CLASSES, D, H, W] (float32,
    numpy) in the ROI grid -- deliberately NOT argmaxed, see module
    docstring point 3."""
    img_nii = nib.load(str(roi_path))
    img = np.asarray(img_nii.dataobj).astype(np.float32)
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    x[0, 0] = volume_wise_znorm(x[0, 0])

    use_amp = device.startswith("cuda")
    prob_sum = None
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        for entry in manifest:
            ckpt_path = CHECKPOINT_DIR / entry["checkpoint"]
            model = load_fold_model(entry["teacher_name"], ckpt_path, device)
            logits = model.sliding_window_predict(x, patch_size=PATCH_SIZE, overlap=0.5, mirror=True)
            probs = F.softmax(logits.float(), dim=1)  # upcast for a numerically stable softmax/sum
            prob_sum = probs if prob_sum is None else prob_sum + probs
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            print(f"  + {entry['teacher_name']} fold{entry['fold']} done", flush=True)

    prob_mean = (prob_sum / len(manifest))[0].cpu().numpy().astype(np.float32)
    return prob_mean, img_nii.affine


def write_fallback_output(t2_path: Path, output_path: Path, reason: str):
    """Last-resort output: an all-background (all-zero) segmentation in the
    input's own exact shape/affine/header. No resampling or cropping is
    needed since Task_4's output contract already requires matching the
    input's native grid exactly, so "all zeros in the input's own grid" is
    trivially spacing/orientation-safe by construction."""
    print(f"[FALLBACK] {reason} -- writing all-background segmentation", flush=True)
    ref = nib.load(str(t2_path))
    zeros = np.zeros(ref.shape[:3], dtype=np.int16)
    out_img = nib.Nifti1Image(zeros, ref.affine, ref.header)
    out_img.header.set_data_dtype(np.int16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(output_path))
    print(f"wrote {output_path} shape={zeros.shape} labels=[0,0] (fallback)", flush=True)


def run_pipeline(t2_path: Path, output_path: Path, device: str, manifest: list):
    """Writes the real prediction to output_path in the normal case. For a
    genuinely blank/degenerate input, writes an intentional all-background
    result instead (see preprocess_safe -- a deliberate, correct response,
    not error-masking) and returns normally. Any OTHER failure is NOT caught
    here -- it propagates as a real exception; the caller (main) does not
    catch it either, so it crashes the process on purpose."""
    t_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="fomo26_task4_") as tmp:
        work_dir = Path(tmp)
        t0 = time.perf_counter()
        result = preprocess_safe(t2_path, TEMPLATE, ROI_REFERENCE, work_dir, device=device)
        t_preprocess = time.perf_counter() - t0
        if result is None:
            # ONLY reached for a genuinely blank/degenerate input.
            write_fallback_output(t2_path, output_path, "blank/degenerate input (mean and std both ~0)")
            return
        print(f"preprocessed -> {result['roi']}  ({t_preprocess:.1f}s)", flush=True)

        t0 = time.perf_counter()
        prob_mean, affine = ensemble_predict_roi(result["roi"], manifest, device)
        t_inference = time.perf_counter() - t0

        # Inverse-transform each class's probability channel SEPARATELY (Linear
        # interpolation) before deciding the label, rather than argmaxing in ROI
        # space and NearestNeighbor-resampling the hard label -- see module
        # docstring point 3 for why this preserves more boundary fidelity.
        t0 = time.perf_counter()
        native_probs = []
        for c in range(N_CLASSES):
            tc0 = time.perf_counter()
            # Uncompressed .nii (not .nii.gz) for these purely-intermediate,
            # never-leave-the-container scratch files -- gzip compression of a
            # 94M-voxel float32 native-resolution probability map (both the
            # write here AND antsApplyTransforms' own write of its output) was
            # measured to cost ~14s of the ~31s postprocess stage, with
            # negligible benefit since nothing reads these files outside this
            # process. The final segmentation written to --output still uses
            # .nii.gz (small, mostly-zero int16 labels -- compresses fast --
            # and the container-validator's contract names that exact file).
            roi_prob_path = work_dir / f"prob_class{c}_roi.nii"
            nib.save(nib.Nifti1Image(prob_mean[c], affine), str(roi_prob_path))
            tc_save = time.perf_counter() - tc0
            native_prob_path = work_dir / f"prob_class{c}_native.nii"
            tc1 = time.perf_counter()
            inverse_transform(roi_prob_path, result["transform"], t2_path, native_prob_path,
                               interpolation="Linear")
            tc_xform = time.perf_counter() - tc1
            tc2 = time.perf_counter()
            arr = np.asanyarray(nib.load(str(native_prob_path)).dataobj).astype(np.float32)
            tc_load = time.perf_counter() - tc2
            native_probs.append(arr)
            print(f"  inverse-transformed class {c} probability map -> {native_prob_path}  "
                  f"(save={tc_save:.1f}s xform={tc_xform:.1f}s load={tc_load:.1f}s)", flush=True)

        stacked = np.stack(native_probs, axis=0)  # [N_CLASSES, D, H, W] in native space
        pred_native = np.argmax(stacked, axis=0).astype(np.int16)

        ref = nib.load(str(t2_path))
        assert pred_native.shape == ref.shape[:3], (pred_native.shape, ref.shape)
        assert pred_native.min() >= 0 and pred_native.max() <= 2, \
            f"unexpected label range [{pred_native.min()},{pred_native.max()}]"

        out_img = nib.Nifti1Image(pred_native, ref.affine, ref.header)
        out_img.header.set_data_dtype(np.int16)
        nib.save(out_img, str(output_path))
        t_postprocess = time.perf_counter() - t0

    t_total = time.perf_counter() - t_start
    print(f"wrote {output_path} shape={pred_native.shape} "
          f"labels=[{int(pred_native.min())},{int(pred_native.max())}]", flush=True)
    print(f"TIMING: preprocess={t_preprocess:.1f}s inference={t_inference:.1f}s "
          f"postprocess={t_postprocess:.1f}s total={t_total:.1f}s", flush=True)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    t2_path = Path(args.t2)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    print(f"ensembling {len(manifest)} checkpoints: "
          f"{[(m['teacher_name'], m['fold']) for m in manifest]}", flush=True)

    # No top-level try/except: any unexpected failure crashes with a full
    # traceback + non-zero exit rather than being silently converted to an
    # all-zero prediction -- see run_pipeline()'s and this module's
    # docstrings.
    run_pipeline(t2_path, output_path, device, manifest)


if __name__ == "__main__":
    main()
