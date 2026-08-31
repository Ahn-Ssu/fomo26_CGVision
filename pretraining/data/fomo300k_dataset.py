"""FOMO300K Dataset -- two implementations.

`FOMO300KPreprocessedDataset` (default, added 2026-07-16): reads the
raw-preserving `.npz` cache produced by data/preprocess_raw.py at
/root/data/FOMO300K_preprocessed/ (full corpus preprocessed 2026-07-15:
165,790 scans, 1.27TB -- see /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/03_PREPROCESSING_REPORT.md). Each file is
already 1mm-isotropic with an explicit foreground mask, so __getitem__ only
needs to crop + normalize -- no zip extraction or SimpleITK resampling per
sample, which were the two most expensive steps in the original pipeline.

`FOMO300KDataset` (original, zip-based): kept for reference / for any future
data that hasn't been through preprocess_raw.py yet. Reads directly from the
per-subject/session zip archives at
/root/data/FOMO300K/<PTxxx_Name>/sub-*/ses-*.zip.

Both return modality + subject/session identifiers alongside the image
tensor (unlike asparagus's own PretrainDataset -- /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/01_ASPARAGUS_ANALYSIS.md Sec
C.3 -- which returns only {file_path, image, transforms_applied}), since the
episodic teacher sampler (sampler/episodic.py) and future teacher-confidence
weighting (PRE_ANALYSIS.md Sec 5.1's BraTS modality-Dice table) need to know
what was sampled. `FOMO300KPreprocessedDataset` additionally returns `mask`,
since preprocess_raw.py's explicit foreground mask (computed pre-resample,
not contaminated by cubic-interpolation boundary leakage -- see
/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/03_PREPROCESSING_REPORT.md Issue A) is preferred over teachers re-deriving their
own `x != x.min()` mask; run_pretrain.py passes it to every teacher via
`meta={"mask": ...}` (BaseTeacher.preprocess()'s meta contract).

Normalization: both apply asparagus's own volume_wise_znorm (data/normalize.py,
one shared implementation -- previously duplicated in 3 places, consolidated
2026-07-16) so the student sees the exact same input distribution the
teachers' own preprocess() methods are built to re-derive from (see
/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/02_TEACHER_REPORT.md Sec D). This happens on-the-fly at __getitem__ time, every
epoch, by design (Task 3's whole point is NOT baking normalization into the
stored data -- see /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/03_PREPROCESSING_REPORT.md).
"""

import csv
import json
import os
import random
import re
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from FOMO26.data.normalize import asparagus_volume_wise_znorm

FOMO300K_ROOT = "/root/data/FOMO300K"
INDEX_CACHE = "/root/FOMO26/data/fomo300k_index.json"

PREPROCESSED_ROOT = "/root/data/FOMO300K_preprocessed"
# Two separate log files -- the original 165,790-scan run (2026-07-15) plus
# the PT030_OpenNeuro backfill (140,300 scans, 2026-07-16, see
# /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/03_PREPROCESSING_REPORT.md for why OpenNeuro needed a separate run) -- kept
# separate rather than merged on disk; load_preprocessed_index() reads both.
PREPROCESSED_LOG_CSV = [
    os.path.join(PREPROCESSED_ROOT, "preprocess_log.csv"),
    os.path.join(PREPROCESSED_ROOT, "preprocess_log_openneuro.csv"),
]

_MODALITY_PATTERNS = [
    ("FLAIR", re.compile(r"FLAIR", re.I)),
    ("T1w", re.compile(r"T1w", re.I)),
    ("T2w", re.compile(r"T2w", re.I)),
    ("T2star", re.compile(r"T2star|T2\*", re.I)),
    ("SWI", re.compile(r"SWI", re.I)),
    ("dwi", re.compile(r"dwi", re.I)),
    ("angio", re.compile(r"angio|MRA|TOF", re.I)),
]


def guess_modality(filename: str) -> str:
    for name, pat in _MODALITY_PATTERNS:
        if pat.search(filename):
            return name
    return "unknown"


def build_index(root: str = FOMO300K_ROOT, cache_path: str = INDEX_CACHE, limit_zips: Optional[int] = None) -> List[dict]:
    """Scans root for **/sub-*/ses-*.zip (recursive -- see note below), reads
    each zip's central directory (fast, no decompression) to list NIfTI
    members, and caches the result. Re-scanning all 81k zips is slow
    (network storage); use the cache unless `force_rebuild`.

    BUG FOUND 2026-07-16 (via user question about mri_info.tsv's row count
    not matching our preprocessed output count): the original pattern
    `*/sub-*/ses-*.zip` (fixed 3 path segments under root) silently missed
    `PT030_OpenNeuro/<accession-id>/sub-*/ses-*.zip` -- OpenNeuro bundles many
    independent datasets under one PT030_OpenNeuro folder, each in its own
    accession-numbered subdirectory (e.g. `ds004271/`), adding one extra path
    level. This is 45,377 of the corpus's 81,190 zip files (56%), 140,389 of
    ~306,203 total scans (45.8%) per a from-scratch recursive-glob re-scan --
    confirmed ENTIRELY missing from the full preprocessing run completed
    2026-07-15 (which only produced 165,790 scans). Fixed here by using a
    recursive `**` glob (verified to find all 81,190 zips, matching
    examples/survey_geometry.py's independently-written recursive
    `glob.glob(root, "**", "*.zip", recursive=True)`, which is why
    examples/geo_survey/ correctly reported ~306K while this function did not).
    Also fixed: `dataset_name` must be the path component directly under
    `root` (`relative_to(root).parts[0]`), NOT `parts[-3]` -- the latter
    resolves to the OpenNeuro accession-id subdirectory instead of
    "PT030_OpenNeuro" once nesting depth varies, which would have scattered
    one dataset's output across thousands of wrongly-named "datasets"."""
    zip_paths = sorted(Path(root).glob("**/sub-*/ses-*.zip"))
    if limit_zips is not None:
        zip_paths = zip_paths[:limit_zips]

    entries = []
    for zp in zip_paths:
        try:
            zf = zipfile.ZipFile(zp)
        except (zipfile.BadZipFile, OSError):
            continue
        rel_parts = zp.relative_to(root).parts  # (dataset, [group...,] subject, "session.zip")
        dataset_name = rel_parts[0]
        subject = zp.parts[-2]
        session = zp.stem
        # "group": any path component(s) between dataset and subject (e.g.
        # OpenNeuro's accession id "ds004271" in
        # PT030_OpenNeuro/ds004271/sub-01/ses-01.zip). Empty for the normal
        # <dataset>/sub-*/ses-*.zip layout. MUST be included when building an
        # output filename downstream -- see preprocess_raw.py's 2026-07-16
        # collision-bug note: omitting it silently overwrote ~71,717 files
        # (different OpenNeuro accessions sharing generic sub-01/ses-01
        # naming all mapped to the same output path).
        group = "_".join(rel_parts[1:-2])
        for member in zf.namelist():
            if not member.endswith(".nii.gz"):
                continue
            entries.append({
                "zip_path": str(zp),
                "member": member,
                "dataset": dataset_name,
                "group": group,
                "subject": subject,
                "session": session,
                "modality": guess_modality(member),
            })
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(entries, f)
    return entries


_INDEX_SCHEMA_KEYS = {"zip_path", "member", "dataset", "group", "subject", "session", "modality"}


def load_or_build_index(cache_path: str = INDEX_CACHE, root: str = FOMO300K_ROOT,
                         limit_zips: Optional[int] = None) -> List[dict]:
    """BUG FOUND 2026-07-16: this function silently reused a cache file that
    predated the `group` field (added to fix the OpenNeuro filename-collision
    bug -- see build_index()'s docstring), so a second full OpenNeuro
    preprocessing run (248 min) reproduced the EXACT same collision even
    though the fixed code was running, because it never actually rebuilt the
    index. Now checks the cached entries' keys match the current schema and
    discards+rebuilds the cache if not, instead of trusting any existing
    cache file blindly."""
    if os.path.isfile(cache_path) and limit_zips is None:
        with open(cache_path) as f:
            cached = json.load(f)
        if cached and set(cached[0].keys()) == _INDEX_SCHEMA_KEYS:
            return cached
        # stale/mismatched schema -- fall through and rebuild
    return build_index(root=root, cache_path=cache_path if limit_zips is None else None, limit_zips=limit_zips)


def load_preprocessed_index(csv_path=PREPROCESSED_LOG_CSV, root: str = PREPROCESSED_ROOT,
                             modalities: Optional[List[str]] = None, limit: Optional[int] = None) -> List[dict]:
    """Reads preprocess_raw.py's preprocess_log*.csv (one row per attempted
    NIfTI, status in {ok,skipped,error}) and reconstructs each successfully-
    preprocessed entry's .npz path using the exact naming convention from
    preprocess_raw.py::process_one -- cheap (single CSV parse over ~300k
    rows), avoids opening every .npz just to build an index.

    `csv_path` accepts a single path or a list of paths (e.g. the original
    preprocess_log.csv plus preprocess_log_openneuro.csv from the 2026-07-16
    backfill -- see /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/03_PREPROCESSING_REPORT.md -- kept as separate files rather
    than merged on disk).

    Uses `row["group"]` (empty string if the column is absent, e.g. rows from
    the original pre-2026-07-16 preprocess_log.csv, which predates this field
    and only ever covered the non-nested layout where group is always "" --
    see build_index()'s docstring for why `group` exists) to reconstruct the
    exact output filename, matching preprocess_raw.py::process_one exactly."""
    csv_paths = [csv_path] if isinstance(csv_path, str) else list(csv_path)
    entries = []
    for path_ in csv_paths:
        with open(path_, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["status"] != "ok":
                    continue
                if modalities is not None and row["modality"] not in modalities:
                    continue
                group = row.get("group") or ""
                group_part = f"{group}__" if group else ""
                out_name = f"{row['dataset']}__{group_part}{row['subject']}__{row['session']}__{Path(row['member']).name}".replace(".nii.gz", "") + ".npz"
                out_path = os.path.join(root, row["dataset"], out_name)
                entries.append({
                    "path": out_path,
                    "modality": row["modality"],
                    "dataset": row["dataset"],
                    "subject": row["subject"],
                    "session": row["session"],
                    "member": row["member"],
                })
                if limit is not None and len(entries) >= limit:
                    break
        if limit is not None and len(entries) >= limit:
            break
    return entries


# Default: ~1/EVAL_HASH_MODULUS of subject-sessions held out for monitoring.
EVAL_HASH_MODULUS = 1000


def _subject_session_key(entry: dict) -> str:
    """Stable identity for a (dataset, group, subject, session) -- i.e. every
    modality of the same scan session shares one key, so held-out status is
    decided per SESSION, not per individual NIfTI file (avoids leaking e.g.
    a subject's T1 into training while their FLAIR sits in eval)."""
    return f"{entry['dataset']}|{entry.get('group', '')}|{entry['subject']}|{entry['session']}"


def is_eval_entry(entry: dict, modulus: int = EVAL_HASH_MODULUS) -> bool:
    """Deterministic, seed-independent (crc32, not Python's randomized
    string hash()) split: same held-out set on every run, forever, so
    different training versions (ver1, ver2 with LoRA, ...) can be compared
    on IDENTICAL monitoring data at matching steps -- the whole point being
    that eval_log.csv rows are comparable ACROSS run_names, not just within
    one run. This is a fixed holdout for MONITORING only; per user request
    2026-07-17, ALL other data (including everything not in this holdout)
    is still used for training, matching the FOMO25 precedent of using the
    full pretrain corpus with no train/val split for gradient updates."""
    key = _subject_session_key(entry)
    return (zlib.crc32(key.encode("utf-8")) % modulus) == 0


def split_train_eval(entries: List[dict], modulus: int = EVAL_HASH_MODULUS) -> Tuple[List[dict], List[dict]]:
    train, eval_ = [], []
    for e in entries:
        (eval_ if is_eval_entry(e, modulus) else train).append(e)
    return train, eval_


def _resample_to_iso(img: sitk.Image, target_spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()
    new_size = [int(round(osz * ospc / tspc)) for osz, ospc, tspc in zip(orig_size, orig_spacing, target_spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkBSpline)
    return resampler.Execute(img)


def _random_crop_or_pad(volume: np.ndarray, patch_size: int, rng: random.Random,
                         extra: Optional[np.ndarray] = None, extra2: Optional[np.ndarray] = None):
    """Crops/pads `volume` to a patch_size^3 cube at a random offset. If
    `extra` (e.g. a mask, same shape as volume) is given, it is cropped/padded
    at the SAME offsets so the two stay spatially aligned; returns
    (patch, extra_patch) in that case, else just patch.

    `extra2` (added for VoCo, 2026-08-02): a second optional array cropped at
    the same offsets -- for the RAW (pre-asparagus-znorm) intensities. VoCo's
    own preprocessing is a whole-volume z-score computed WITHOUT asparagus's
    clamp-then-mask-znorm applied first (composing the two isn't equivalent
    to VoCo's actual training-time normalization, since the clamp step isn't
    affine -- see teacher/voco_teacher.py). Requires `extra` to also be given
    when `extra2` is given; returns (patch, extra_patch, extra2_patch)."""
    out = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
    out_extra = np.zeros((patch_size, patch_size, patch_size), dtype=extra.dtype) if extra is not None else None
    out_extra2 = np.zeros((patch_size, patch_size, patch_size), dtype=extra2.dtype) if extra2 is not None else None
    src_slices, dst_slices = [], []
    for dim in range(3):
        n = volume.shape[dim]
        if n >= patch_size:
            start = rng.randint(0, n - patch_size)
            src_slices.append(slice(start, start + patch_size))
            dst_slices.append(slice(0, patch_size))
        else:
            src_slices.append(slice(0, n))
            pad_start = (patch_size - n) // 2
            dst_slices.append(slice(pad_start, pad_start + n))
    out[tuple(dst_slices)] = volume[tuple(src_slices)]
    if extra is not None:
        out_extra[tuple(dst_slices)] = extra[tuple(src_slices)]
    if extra2 is not None:
        out_extra2[tuple(dst_slices)] = extra2[tuple(src_slices)]
        return out, out_extra, out_extra2
    if extra is not None:
        return out, out_extra
    return out


class FOMO300KPreprocessedDataset(Dataset):
    """Default dataset -- reads /root/data/FOMO300K_preprocessed/*.npz."""

    def __init__(self, patch_size: int = 128, index: Optional[List[dict]] = None,
                 modalities: Optional[List[str]] = None, limit: Optional[int] = None,
                 seed: Optional[int] = None):
        self.patch_size = patch_size
        self.index = index if index is not None else load_preprocessed_index(modalities=modalities, limit=limit)
        assert self.index, "Empty preprocessed-FOMO300K index -- check PREPROCESSED_ROOT / preprocess_log.csv exist."
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        entry = self.index[idx]
        d = np.load(entry["path"])
        data = d["data"].astype(np.float32)  # raw intensity (* intensity_scale if applied), 1mm iso, bg=0
        mask = d["mask"].astype(bool)

        normed = asparagus_volume_wise_znorm(data, mask=mask)
        patch, mask_patch, raw_patch = _random_crop_or_pad(
            normed, self.patch_size, self._rng, extra=mask.astype(np.uint8), extra2=data)

        return {
            "image": torch.from_numpy(patch).unsqueeze(0),        # (1, P, P, P) float32, asparagus-normed
            "mask": torch.from_numpy(mask_patch).unsqueeze(0),     # (1, P, P, P) uint8
            "raw": torch.from_numpy(raw_patch).unsqueeze(0),       # (1, P, P, P) float32, RAW (pre-znorm) --
                                                                    # for teachers whose own normalization
                                                                    # isn't safe to compose with asparagus's
                                                                    # (e.g. VoCo's whole-volume z-score)
            "modality": entry["modality"],
            "dataset": entry["dataset"],
            "subject": entry["subject"],
            "session": entry["session"],
            "file_path": entry["path"],
        }


class FOMO300KDataset(Dataset):
    """Fallback / reference dataset -- reads directly from FOMO300K zip
    archives (no preprocessing step required, but ~1-1.5s/sample due to zip
    extraction + on-the-fly SimpleITK resampling; superseded by
    FOMO300KPreprocessedDataset now that the full corpus has been
    preprocessed -- see /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/03_PREPROCESSING_REPORT.md). Kept for any future data
    dropped into FOMO300K/ that hasn't been run through preprocess_raw.py yet."""

    def __init__(self, patch_size: int = 128, index: Optional[List[dict]] = None,
                 limit_zips: Optional[int] = None, modalities: Optional[List[str]] = None,
                 seed: Optional[int] = None):
        self.patch_size = patch_size
        self.index = index if index is not None else load_or_build_index(limit_zips=limit_zips)
        if modalities is not None:
            self.index = [e for e in self.index if e["modality"] in modalities]
        assert self.index, "Empty FOMO300K index -- check root path / limit_zips / modalities filter."
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.index)

    def _load_nifti_from_zip(self, zip_path: str, member: str) -> "nib.Nifti1Image":
        zf = zipfile.ZipFile(zip_path)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp.write(zf.read(member))
            tmp_path = tmp.name
        try:
            img = nib.load(tmp_path)
            img = nib.Nifti1Image(np.asarray(img.dataobj), img.affine, img.header)  # force load into memory
        finally:
            os.remove(tmp_path)
        return img

    def __getitem__(self, idx: int) -> dict:
        entry = self.index[idx]
        nib_img = self._load_nifti_from_zip(entry["zip_path"], entry["member"])
        data = np.asarray(nib_img.dataobj).astype(np.float32)

        sitk_img = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))  # nib (X,Y,Z) -> sitk (Z,Y,X) array order
        zooms = nib_img.header.get_zooms()[:3]
        sitk_img.SetSpacing([float(z) for z in zooms])
        sitk_resampled = _resample_to_iso(sitk_img, target_spacing=(1.0, 1.0, 1.0))
        resampled = np.transpose(sitk.GetArrayFromImage(sitk_resampled), (2, 1, 0))  # back to (X,Y,Z)

        # Simple post-resample threshold mask -- NOT Issue-A-safe (computed
        # after cubic interpolation, so boundary leakage can contaminate it
        # slightly; see /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/03_PREPROCESSING_REPORT.md Issue A). Acceptable for this
        # fallback path only; FOMO300KPreprocessedDataset's mask is the
        # correct pre-resample-computed one and should be preferred.
        mask = (resampled > 1e-6).astype(np.uint8)
        normed = asparagus_volume_wise_znorm(resampled, mask=mask.astype(bool))
        patch, mask_patch, raw_patch = _random_crop_or_pad(
            normed, self.patch_size, self._rng, extra=mask, extra2=resampled)

        return {
            "image": torch.from_numpy(patch).unsqueeze(0),  # (1, P, P, P)
            "mask": torch.from_numpy(mask_patch).unsqueeze(0),  # (1, P, P, P) uint8
            "raw": torch.from_numpy(raw_patch).unsqueeze(0),  # (1, P, P, P) float32, RAW (pre-znorm)
            "modality": entry["modality"],
            "dataset": entry["dataset"],
            "subject": entry["subject"],
            "session": entry["session"],
            "file_path": f"{entry['zip_path']}::{entry['member']}",
        }


if __name__ == "__main__":
    import sys
    import time

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    t0 = time.time()
    ds = FOMO300KPreprocessedDataset(patch_size=128, limit=limit, seed=0)
    print(f"Indexed {len(ds)} preprocessed entries (limit={limit}) in {time.time()-t0:.1f}s")
    t0 = time.time()
    sample = ds[0]
    print(f"Loaded one sample in {time.time()-t0:.2f}s:")
    for k, v in sample.items():
        print(" ", k, v.shape if torch.is_tensor(v) else v)
