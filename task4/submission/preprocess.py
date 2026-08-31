"""Forward preprocessing: raw input MRI -> network-ready ROI crop.

    input.nii.gz
      -> ResampleImage to 1mm isotropic (much smaller/faster HD-BET+registration)
      -> HD-BET (skull-strip, brain mask) -- run at 1mm
      -> ANTs rigid+affine registration (skull-stripped 1mm brain -> template) -- at 1mm
      -> antsApplyTransforms applies the resulting (resolution-independent,
         physical-space) transform DIRECTLY to the ORIGINAL, untouched
         native-resolution input onto a custom ROI reference grid (BSpline
         for the image; the transform + registration are computed once, then
         reused for as many reference grids as needed -- e.g. run this
         twice with different --roi-reference to get both a native-spacing
         and a 0.5mm-iso crop from the same registration)

The 1mm-downsample-first step (2026-08-21) is a measured ~53% speedup for
HD-BET+registration (mean 60.2s -> 28.2s across a 10-subject sample spanning
all 5 CV folds) with no measurable quality cost: round-trip label Dice across
those same 10 subjects was statistically indistinguishable between the native-
resolution path (dice_1 0.9976+/-0.0044, dice_2 0.9986+/-0.0022) and this 1mm
path (dice_1 0.9978+/-0.0030, dice_2 0.9978+/-0.0028), with no subject showing
a systematic advantage either way. Valid because ANTs affine transforms are
resolution-independent physical-space transforms -- the transform computed at
1mm is applied straight to the untouched native input, never to a downsampled
copy of it, so no accuracy is given up in the actual ROI crop that reaches the
model. See NOTES.md for the full before/after numbers.

Every intermediate artifact (brain mask, affine transform .mat, ROI crop)
is written under --work-dir so postprocess.py can invert the exact same
transform later. This is the case-level unit of work an Apptainer entry
point would call once per input scan.

Requires on PATH: hd-bet (pip), antsRegistrationSyNQuick.sh, antsApplyTransforms
(ANTs). See ../Apptainer.def.notes for the container dependency list.

Usage:
  python preprocess.py \
    --input /data/case001/t2w.nii.gz \
    --template assets/MNI152_T1_1mm_brain.nii.gz \
    --roi-reference assets/roi_reference_iso05.nii.gz \
    --work-dir /work/case001 \
    --device cuda
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

# See predict.py's own top-of-file comment: %environment's `export HOME=/root`
# isn't guaranteed to reach this process in every execution environment (a
# sibling FOMO26 container was marked INVALID for an internet access attempt
# traced to exactly this). Setting it here too makes this module self-
# contained even when preprocess.py's own CLI (main(), below) is run
# standalone rather than imported by predict.py.
os.environ["HOME"] = "/root"


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def is_blank_image(array: np.ndarray, eps: float = 1e-6) -> bool:
    """Degenerate-input guard: True only if the raw input's mean AND std are
    BOTH essentially zero -- a genuinely blank/empty scan. Checked BEFORE
    even attempting HD-BET so a truly blank input takes the cheap,
    always-safe fallback path immediately instead of burning time on
    registration that's certain to fail or be meaningless anyway.

    This is a DELIBERATE, narrow decision (not general error-masking, see
    predict.py's module docstring, 2026-08-22 policy change): real
    T2-weighted MRI intensities are non-negative, so any actual scan -- even
    a poor-quality or unusually-oriented one -- has a clearly nonzero mean;
    only a genuinely blank/all-zero volume collapses both statistics to
    zero. Any OTHER kind of bad input (corrupt file, wrong orientation,
    weird-but-nonzero intensities, etc.) is NOT caught here -- it's expected
    to surface downstream as a real, visible failure rather than being
    silently absorbed into "blank input" handling."""
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return True
    mean = float(finite.mean())
    std = float(finite.std())
    return abs(mean) < eps and std < eps


def downsample_to_1mm(input_path: Path, output_path: Path):
    """ANTs' ResampleImage to 1mm isotropic -- HD-BET and registration both run
    much faster on this than on the native (typically 0.5mm-ish) resolution,
    and the affine transform ANTs fits is a physical-space transform, so it
    can be applied directly to the untouched native-resolution input later
    with no accuracy cost (see module docstring)."""
    run(["ResampleImage", "3", str(input_path), str(output_path), "1x1x1", "0", "0"], timeout=120)
    return output_path


def hd_bet(input_path: Path, out_brain: Path, device: str):
    # HD_BET.checkpoint_download.maybe_download_parameters() runs
    # UNCONDITIONALLY on every `hd-bet` invocation (HD_BET/entry_point.py)
    # and checks os.path.expanduser('~')/hd-bet_params/release_2.0.0/... --
    # if that resolves to anywhere other than where we bundled the weights,
    # it falls back to a live `requests.get()` download from zenodo.org.
    # Apptainer.def's `%environment` sets `export HOME=/root` for exactly
    # this reason, but a sibling FOMO26 container (Task 6/7) was marked
    # INVALID by the organizers for "attempts to access the internet" --
    # root-caused to `%environment`'s HOME export not reliably reaching the
    # process env in the actual Synapse execution environment. Passing
    # HOME="/root" directly as this subprocess's own env (not relying on
    # %environment at all) guarantees it regardless of how the outer
    # container environment behaves.
    #
    # timeout=120: belt-and-suspenders against the HOME fix above somehow
    # still not applying (or any other cause of HD-BET blocking on a network
    # call) -- normal HD-BET runs take ~2-13s end to end, so 120s is already
    # generous headroom, not a tight bound. Without this, a hung network
    # call inside HD-BET would hang subprocess.run() itself with NO
    # exception ever raised, meaning `hd_bet_safe()`'s own try/except below
    # would never even get a chance to catch it and fall back gracefully --
    # the whole process would just sit blocked until an external watchdog
    # (if any) kills it, producing no output at all rather than a handled
    # fallback. A `subprocess.TimeoutExpired` here IS a normal Exception
    # subclass, so it's caught by hd_bet_safe() exactly like any other
    # HD-BET failure.
    run([
        "hd-bet", "-i", str(input_path), "-o", str(out_brain),
        "-device", device, "--save_bet_mask",
    ], env=dict(os.environ, HOME="/root"), timeout=120)
    # HD-BET writes the mask as <out_brain stem>_bet.nii.gz next to out_brain
    mask_path = out_brain.parent / (out_brain.name.replace(".nii.gz", "") + "_bet.nii.gz")
    return mask_path


def hd_bet_safe(input_path: Path, out_brain: Path, device: str):
    """Returns out_brain on success, or None if HD-BET raised OR produced a
    zero/near-zero-mass (empty) brain -- signals the caller to fall back to
    the raw (non-skull-stripped) input for registration instead. See Task_3
    incident report, issue 4 (RuntimeError: Registration failed with error
    code 1 / ITK "Total Mass of the image was zero" on a hidden test subject
    whose HD-BET output was an empty mask)."""
    try:
        hd_bet(input_path, out_brain, device)
    except Exception as e:
        print(f"[WARNING] HD-BET raised {type(e).__name__}: {e} -- falling back to raw input", flush=True)
        return None

    try:
        data = np.asanyarray(nib.load(str(out_brain)).dataobj)
        mass = float(np.nansum(data))
    except Exception as e:
        print(f"[WARNING] could not read HD-BET output ({e}) -- falling back to raw input", flush=True)
        return None

    if mass <= 0 or not np.isfinite(mass):
        print(f"[WARNING] HD-BET output has zero/invalid mass ({mass}) -- falling back to raw input", flush=True)
        return None
    return out_brain


def affine_register(moving_brain: Path, template: Path, out_prefix: Path, threads: int):
    run([
        "antsRegistrationSyNQuick.sh", "-d", "3",
        "-f", str(template), "-m", str(moving_brain),
        "-t", "a",  # rigid + affine ONLY -- no nonlinear SyN, see chat 2026-08-19
        "-o", str(out_prefix) + "_", "-n", str(threads),
    ], timeout=300)  # measured 16-70s normally -- generous margin, not a tight bound
    return Path(str(out_prefix) + "_0GenericAffine.mat")


def register_safe(candidates: list, template: Path, out_prefix: Path, threads: int):
    """Tries each (name, moving_image_path) candidate in order, returning the
    first successful registration's transform, or None if all candidates
    fail. `candidates` is built EXPLICITLY by the caller (not deduplicated
    via path-equality checks) -- the sibling Task_3 container's first
    implementation of this used an `if candidate == brain_path and
    brain_path == raw_path: continue` equality check that, when HD-BET had
    ALREADY internally fallen back to the raw path, collapsed both
    candidates to the same value and skipped both. Building the candidate
    list explicitly (only adding the HD-BET brain if it actually succeeded)
    avoids that bug class entirely -- there is never an ambiguous
    "are these the same candidate" comparison to get wrong."""
    for name, moving_path in candidates:
        try:
            transform = affine_register(moving_path, template, out_prefix, threads)
            if transform.exists() and transform.stat().st_size > 0:
                print(f"[INFO] registration succeeded using candidate: {name}", flush=True)
                return transform
            print(f"[WARNING] registration for candidate {name!r} produced no transform file", flush=True)
        except Exception as e:
            print(f"[WARNING] registration failed for candidate {name!r}: {type(e).__name__}: {e}", flush=True)
    return None


def apply_to_roi(input_path: Path, transform_mat: Path, roi_reference: Path,
                  output_path: Path, interpolation: str):
    run([
        "antsApplyTransforms", "-d", "3",
        "-i", str(input_path), "-r", str(roi_reference),
        "-o", str(output_path), "-t", str(transform_mat),
        "-n", interpolation,
    ], timeout=120)


def preprocess(input_path: Path, template: Path, roi_reference: Path, work_dir: Path,
                device: str = "cuda", threads: int = 32, interpolation: str = "BSpline") -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)

    input_1mm = work_dir / "input_1mm.nii.gz"
    downsample_to_1mm(input_path, input_1mm)

    brain_path = work_dir / "brain.nii.gz"
    mask_path = hd_bet(input_1mm, brain_path, device)

    reg_prefix = work_dir / "reg"
    transform_mat = affine_register(brain_path, template, reg_prefix, threads)

    # Apply the 1mm-computed transform DIRECTLY to the ORIGINAL, untouched
    # native-resolution input_path -- never to the 1mm-downsampled copy.
    roi_path = work_dir / "roi.nii.gz"
    apply_to_roi(input_path, transform_mat, roi_reference, roi_path, interpolation)

    return {
        "brain": brain_path,
        "brain_mask": mask_path,
        "transform": transform_mat,
        "roi": roi_path,
        "original_input": input_path,
    }


def preprocess_safe(input_path: Path, template: Path, roi_reference: Path, work_dir: Path,
                     device: str = "cuda", threads: int = 32, interpolation: str = "BSpline") -> dict | None:
    """Returns None ONLY for a genuinely blank/degenerate input (mean and std
    both ~0, see is_blank_image) -- a deliberate, correct response (a blank
    scan legitimately has no foreground to segment), not error-masking. Any
    OTHER failure (unreadable input, registration failing on every
    candidate, etc.) now RAISES instead of returning None.

    2026-08-22 policy change: this function used to swallow every failure
    mode into a `return None` so predict.py could fall back to an all-zero
    prediction no matter what went wrong. That made real bugs
    indistinguishable from "the model just didn't find anything" -- e.g. the
    HD-BET/$HOME network-access incident produced a silent all-zero
    prediction that took real effort to root-cause, when the underlying
    problem should have crashed loudly instead. FOMO26's evaluation harness
    marks a crashing container INVALID quickly, which is a far more useful,
    fast diagnostic signal than a silently-wrong "score" -- so now only the
    ONE well-understood, deliberate case (blank input) gets the graceful
    all-zero treatment; everything else is allowed to fail loudly.

    hd_bet_safe()/register_safe() still internally retry with a raw
    (non-skull-stripped) image if HD-BET fails -- that's a genuine attempt
    at a real answer via an alternate method, not error-masking, so it's
    unchanged. Only the case where EVERY registration candidate has
    genuinely failed now raises rather than returning None.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # Let a read failure (corrupt file, wrong format, etc.) raise naturally --
    # that's a real problem, not "blank input".
    input_data = np.asanyarray(nib.load(str(input_path)).dataobj)

    if is_blank_image(input_data):
        print("[INFO] input image is blank/degenerate (mean and std both ~0) "
              "-- skipping HD-BET+registration, writing all-background output", flush=True)
        return None

    # 1mm-downsample-first (see module docstring): run HD-BET+registration on a
    # much cheaper 1mm copy, then apply the resulting transform to the
    # untouched native-resolution input_path. If the downsample step itself
    # fails for some reason, fall back to running HD-BET+registration directly
    # at native resolution instead of failing the whole subject over a
    # speed optimization.
    hdbet_input = input_path
    downsample_tag = ""
    try:
        input_1mm = work_dir / "input_1mm.nii.gz"
        downsample_to_1mm(input_path, input_1mm)
        hdbet_input = input_1mm
        downsample_tag = "_1mm"
    except Exception as e:
        print(f"[WARNING] 1mm downsample failed ({type(e).__name__}: {e}) -- "
              f"falling back to native-resolution HD-BET+registration", flush=True)

    brain_path = work_dir / "brain.nii.gz"
    hdbet_result = hd_bet_safe(hdbet_input, brain_path, device)

    candidates = []
    if hdbet_result is not None:
        candidates.append(("hdbet_brain" + downsample_tag, hdbet_result))
    candidates.append(("raw_input" + downsample_tag, hdbet_input))  # always available as a last-resort registration target

    reg_prefix = work_dir / "reg"
    transform_mat = register_safe(candidates, template, reg_prefix, threads)
    if transform_mat is None:
        # Every registration candidate genuinely failed -- a real failure,
        # not a deliberate decision. Raise instead of silently returning
        # None so this surfaces as a crash (see docstring).
        raise RuntimeError(
            "registration failed for every candidate (hdbet_brain and raw_input) -- "
            "cannot produce a ROI crop"
        )

    roi_path = work_dir / "roi.nii.gz"
    # ALWAYS resample the ORIGINAL untouched native-resolution input_path
    # (never the 1mm downsampled copy) onto the ROI grid -- the transform
    # is a resolution-independent physical-space transform, so this loses
    # no accuracy versus computing it at native resolution directly. Let a
    # failure here raise naturally too.
    apply_to_roi(input_path, transform_mat, roi_reference, roi_path, interpolation)

    return {
        "brain": hdbet_result,
        "transform": transform_mat,
        "roi": roi_path,
        "original_input": input_path,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--template", required=True, type=Path)
    ap.add_argument("--roi-reference", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--interpolation", default="BSpline", choices=["BSpline", "Linear", "NearestNeighbor"])
    args = ap.parse_args()

    if shutil.which("antsRegistrationSyNQuick.sh") is None:
        raise SystemExit("antsRegistrationSyNQuick.sh not on PATH -- ANTs must be installed in this environment/container.")
    if shutil.which("hd-bet") is None:
        raise SystemExit("hd-bet not on PATH -- `pip install HD-BET` in this environment/container.")

    result = preprocess(args.input, args.template, args.roi_reference, args.work_dir,
                         args.device, args.threads, args.interpolation)
    print("\n=== preprocess done ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
