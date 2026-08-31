"""Inverse: warp a segmentation network's prediction (living in the ROI
crop grid produced by preprocess.py) back onto the ORIGINAL input scan's
own shape/spacing/orientation, using the SAME affine transform preprocess.py
computed -- inverted via antsApplyTransforms' `[transform,1]` syntax (no
separate inverse-transform computation needed; ANTs affine transforms are
invertible in closed form).

Verified round-trip on real data (chat 2026-08-19/20): warping a known
label through preprocess.py's ROI grid and back with this script reproduced
the original label at Dice 0.9998 (voxel count 2277 -> 2278).

Usage:
  python postprocess.py \
    --prediction /work/case001/prediction_roi.nii.gz \
    --transform /work/case001/reg_0GenericAffine.mat \
    --original-input /data/case001/t2w.nii.gz \
    --output /results/case001/prediction_native.nii.gz
"""
import argparse
import subprocess
from pathlib import Path


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def inverse_transform(prediction_path: Path, transform_mat: Path, original_input: Path,
                       output_path: Path, interpolation: str = "NearestNeighbor"):
    run([
        "antsApplyTransforms", "-d", "3",
        "-i", str(prediction_path),
        "-r", str(original_input),
        "-o", str(output_path),
        "-t", f"[{transform_mat},1]",  # ",1" = use the inverse of this affine
        "-n", interpolation,
    ], timeout=120)
    return output_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prediction", required=True, type=Path, help="Network output, in the ROI crop grid.")
    ap.add_argument("--transform", required=True, type=Path, help="The *_0GenericAffine.mat preprocess.py wrote.")
    ap.add_argument("--original-input", required=True, type=Path, help="The untouched input scan (defines the output grid).")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--interpolation", default="NearestNeighbor",
                     choices=["NearestNeighbor", "Linear", "BSpline"],
                     help="NearestNeighbor for hard label maps; Linear/BSpline if the prediction is a soft/probability map.")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    inverse_transform(args.prediction, args.transform, args.original_input, args.output, args.interpolation)
    print(f"\nwrote {args.output} (same shape/spacing/orientation as {args.original_input})")


if __name__ == "__main__":
    main()
