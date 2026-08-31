#!/usr/bin/env python3
"""FOMO26 Challenge - Task 6/7: Linear Probing / Fairness Embeddings.

Submission #1 (per PREREGISTRATION.md's plan step 7 -- see
/root/fomo-lp/PREREGISTRATION.md): neutral teacher-context (Convpass gate=0,
verified bit-identical to a Convpass-free forward -- see
verify_neutral_gate*.py) on ver5, with the "iso" preprocessing (1mm resample
+ foreground crop + masked znorm, matching pretraining exactly). This is the
most defensible first submission -- confirms the pipeline scores correctly
before spending validation submissions on the other 11 combos.

Input is single-modality, contrast-agnostic (the official manifest tests
both t2w and t1w scans for this task -- see container-validator's
data/manifest.yaml -- so preprocessing must not assume a specific contrast).
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from skimage import exposure

from model.student import StudentResEncUNet

CHECKPOINT_PATH = Path(__file__).parent / "model" / "checkpoint.pt"
NEUTRAL_REFERENCE_TEACHER = "brats"  # arbitrary; see student.py's gate=0 == Convpass-free proof


def parse_args():
    parser = argparse.ArgumentParser(description="FOMO26 Task 6/7 Linear Probing Embeddings")
    parser.add_argument("--input", type=str, required=True, help="Path to input NIfTI")
    parser.add_argument("--output", type=str, required=True, help="Path to save embedding .npy")
    return parser.parse_args()


def load_model(device: torch.device) -> StudentResEncUNet:
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
    sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    teachers = sorted({k.split(".convpass.")[1].split(".")[0] for k in sd if ".convpass." in k})
    model = StudentResEncUNet(in_channels=1, teachers=teachers, convpass=True,
                               convpass_encoder=True, skip_alpha=False, norm_conditional=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert len(missing) == 0, f"checkpoint transplant incomplete: missing={missing}"

    # Neutral path: zero the reference teacher's Convpass gate at every
    # stage. out = block_out + gate * convpass_out collapses to out =
    # block_out exactly (gate is a plain float multiply by 0.0) -- verified
    # torch.equal against a Convpass-free forward on this same ver5
    # checkpoint (encoder AND decoder), see verify_neutral_gate*.py.
    with torch.no_grad():
        for stage in list(model.encoder.stages) + list(model.decoder.stages):
            if getattr(stage, "convpass_enabled", False) and NEUTRAL_REFERENCE_TEACHER in stage.convpass_gate:
                stage.convpass_gate[NEUTRAL_REFERENCE_TEACHER].fill_(0.0)

    model.eval()
    model.to(device)
    return model


def _sitk_resample(np_vol: np.ndarray, spacing, target_spacing=(1.0, 1.0, 1.0),
                    interpolator=sitk.sitkBSpline, default_value: float = 0.0) -> np.ndarray:
    """Verbatim copy of FOMO26/data/preprocess_raw.py's _sitk_resample (same
    function pretraining's own Issue A fix uses -- see
    _asparagus_volume_wise_znorm's docstring below for why this is copied
    rather than imported)."""
    img = sitk.GetImageFromArray(np.transpose(np_vol, (2, 1, 0)))
    img.SetSpacing([float(s) for s in spacing])
    orig_size = img.GetSize()
    new_size = [max(1, int(round(osz * ospc / tspc))) for osz, ospc, tspc in zip(orig_size, spacing, target_spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetInterpolator(interpolator)
    out = resampler.Execute(img)
    return np.transpose(sitk.GetArrayFromImage(out), (2, 1, 0))


def _asparagus_volume_wise_znorm(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Verbatim copy of FOMO26/data/normalize.py's asparagus_volume_wise_znorm
    (mask branch only -- this container always has an explicit foreground
    mask, never the array!=array.min() heuristic branch). Copied rather than
    imported because the container has no access to the FOMO26 repo outside
    what's bundled here -- see this project's standing rule against
    reimplementing pretraining's preprocessing functions (a past preprocessing
    mismatch was a real bug root cause). Byte-for-byte diffed against the
    original on 2026-08-11."""

    def clamp(x, m, q=0.99):
        q_val = np.quantile(x[m], q)
        return np.clip(x, a_min=None, a_max=q_val)

    def znormalize(x, m):
        values = x[m]
        mean, std = np.mean(values), np.std(values)
        assert std > 0
        x = x.astype(np.float64, copy=True)
        x -= mean
        x /= std
        return x

    def rescale(x, out_range=(0, 1)):
        return exposure.rescale_intensity(x, out_range=out_range)

    m = mask.astype(bool)
    if m.sum() == 0:
        return array.astype(np.float32)
    out = clamp(array, m)
    out = znormalize(out, m)
    out = rescale(out, out_range=(0, 1))
    return out.astype(np.float32)


def preprocess_iso(nifti_path: str) -> torch.Tensor:
    """Issue A fix (2026-08-11, see /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/03_PREPROCESSING_REPORT.md
    Sec B.2 and preprocess_raw.py:159-173): cubic (order=3) resampling of the intensity
    image can leak small nonzero values into what should be pure background near the
    boundary. Thresholding the RESAMPLED array directly (this function's original
    approach) let that leakage contaminate both the crop bbox and the znorm statistics.
    Fixed to match pretraining exactly: foreground mask computed at NATIVE resolution
    (before interpolation), resampled with nearest-neighbor (never blurred), then the
    resampled intensity image has its background force-cleaned to exact 0."""
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)
    spacing = [float(s) for s in img.header.get_zooms()[:3]]
    native_mask = (data > 1e-6).astype(np.float32)

    resampled = _sitk_resample(data, spacing, target_spacing=(1.0, 1.0, 1.0),
                                interpolator=sitk.sitkBSpline, default_value=0.0)
    resampled_mask = _sitk_resample(native_mask, spacing, target_spacing=(1.0, 1.0, 1.0),
                                     interpolator=sitk.sitkNearestNeighbor, default_value=0.0)
    mask_bin = resampled_mask > 0.5
    resampled = np.where(mask_bin, resampled, 0.0)

    if not mask_bin.any():
        raise ValueError("no foreground voxels after resample")
    coords = np.argwhere(mask_bin)
    mins, maxs = coords.min(axis=0), coords.max(axis=0) + 1
    cropped = resampled[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    own_mask = mask_bin[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

    normed = _asparagus_volume_wise_znorm(cropped, mask=own_mask).astype(np.float32)

    # Pad each dim UP to the next multiple of 32 (5 stride-2 encoder stages),
    # with a 64 MINIMUM (2026-08-31): container-validator's own Task6 test
    # fixture is a tiny (32,32,16) synthetic volume -- padding that to
    # "next multiple of 32" gives exactly (32,32,32), which after 5
    # halvings collapses to a literal 1x1x1 spatial size at the deepest
    # stage, and InstanceNorm3d (student.py, no running stats -- it
    # normalizes over the CURRENT input's own spatial extent every call)
    # raises "Expected more than 1 spatial element" on that degenerate
    # size. Real scans are always far larger than 64 already, so this
    # floor only ever activates on tiny/synthetic test inputs -- it's a
    # crash guard, not a change to real-data behavior.
    target_shape = tuple(max(64, ((s + 31) // 32) * 32) for s in normed.shape)
    padded = np.zeros(target_shape, dtype=np.float32)
    starts = [(t - s) // 2 for s, t in zip(normed.shape, target_shape)]
    padded[starts[0]:starts[0] + normed.shape[0],
           starts[1]:starts[1] + normed.shape[1],
           starts[2]:starts[2] + normed.shape[2]] = normed

    return torch.from_numpy(padded).unsqueeze(0)  # (1, D, H, W)


@torch.no_grad()
def predict(args) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    x = preprocess_iso(args.input).unsqueeze(0).to(device)  # (1, 1, D, H, W)
    feats = model.forward_encoder_only(x, teacher_name=NEUTRAL_REFERENCE_TEACHER)
    deepest = feats[f"enc_stage_{len(model.encoder.stages) - 1}"]
    pooled = F.adaptive_avg_pool3d(deepest, (1, 1, 1))
    embedding = pooled.flatten(1).squeeze(0).cpu().numpy().astype(np.float32)

    assert embedding.ndim == 1, f"expected 1-D embedding, got shape {embedding.shape}"
    assert np.all(np.isfinite(embedding)), "embedding contains NaN/Inf"
    return embedding


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embedding = predict(args)
    np.save(output_path, embedding)
    return 0


if __name__ == "__main__":
    exit(main())
