"""
FOMO300K Geometry Survey
========================

목적: distillation 파이프라인 설계에 필요한 "영상 기하 정보"를 전수 조사한다.
     modality가 아니라 실제 shape / spacing / dimensionality를 본다.

핵심 질문:
  Q1. 2D vs 3D vs 4D 비율은?  (4D=fMRI/DWI series → 별도 처리 필요)
  Q2. Spacing 분포는?  → 1mm isotropic resampling이 얼마나 큰 변형을 주는가
  Q3. Isotropic vs anisotropic 비율은?  (anisotropic는 resampling 손실 큼)
  Q4. In-plane resolution vs slice thickness 분포
  Q5. Shape(voxel 크기) 분포 → patch sampling / 메모리 계획
  Q6. 극단적 케이스 (초고해상도 0.3mm, 초저해상도 5mm+) 비율

설계 원칙:
  - zip을 디스크에 풀지 않는다. zip 멤버의 nii.gz 헤더(첫 352바이트)만 읽는다.
    (306K 스캔 전량 압축해제는 디스크가 감당 못 함)
  - nibabel 의존성 없음. NIfTI-1/2 헤더를 직접 파싱한다.
    (검증 완료: 3D/2D/4D/anisotropic 모두 정확)
  - 멀티프로세싱으로 zip 병렬 처리.

디렉토리 구조 가정 (사용자 제공 예시):
  /root/data/FOMO300K/<dataset>/<...>/sub-XX/ses-YY.zip
  각 zip 안에 sub-XX/ses-YY/**/*.nii.gz 들이 있음.
  (일부는 zip이 아니라 이미 풀린 .nii.gz일 수도 있으므로 둘 다 처리)

사용법:
  python survey_geometry.py --root /root/data/FOMO300K --out_dir /root/FOMO26/examples/geo_survey
  python survey_geometry.py --root /root/data/FOMO300K --out_dir ... --workers 16
  python survey_geometry.py --root ... --limit 500   # 빠른 스모크 테스트 (zip 500개만)

출력:
  out_dir/per_scan_geometry.csv     — 스캔별 raw 기록 (재사용/재분석용)
  out_dir/*.csv                     — 각종 분포표
  out_dir/GEOMETRY_SUMMARY.md       — 요약
"""

import argparse
import csv
import glob
import gzip
import io
import os
import struct
import sys
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed


# ============================================================== #
# NIfTI header parsing (no nibabel)
# ============================================================== #

def _parse_nifti1(raw: bytes) -> dict:
    if struct.unpack_from('<i', raw, 0)[0] == 348:
        e = '<'
    elif struct.unpack_from('>i', raw, 0)[0] == 348:
        e = '>'
    else:
        return None
    dim = struct.unpack_from(e + '8h', raw, 40)
    pixdim = struct.unpack_from(e + '8f', raw, 76)
    datatype = struct.unpack_from(e + 'h', raw, 70)[0]
    ndim = dim[0]
    if ndim < 1 or ndim > 7:
        return None
    shape = tuple(int(dim[1 + i]) for i in range(ndim))
    spacing = tuple(round(float(pixdim[1 + i]), 4) for i in range(ndim))
    return {"nifti": 1, "ndim": ndim, "shape": shape,
            "spacing": spacing, "datatype": datatype}


def _parse_nifti2(raw: bytes) -> dict:
    # NIfTI-2: sizeof_hdr=540 at offset 0; magic 'n+2' at offset 4
    if struct.unpack_from('<i', raw, 0)[0] == 540:
        e = '<'
    elif struct.unpack_from('>i', raw, 0)[0] == 540:
        e = '>'
    else:
        return None
    # NIfTI-2 layout: datatype(int16)@12, dim(int64[8])@16, pixdim(float64[8])@104
    datatype = struct.unpack_from(e + 'h', raw, 12)[0]
    dim = struct.unpack_from(e + '8q', raw, 16)
    pixdim = struct.unpack_from(e + '8d', raw, 104)
    ndim = dim[0]
    if ndim < 1 or ndim > 7:
        return None
    shape = tuple(int(dim[1 + i]) for i in range(ndim))
    spacing = tuple(round(float(pixdim[1 + i]), 4) for i in range(ndim))
    return {"nifti": 2, "ndim": ndim, "shape": shape,
            "spacing": spacing, "datatype": datatype}


def parse_nifti_header(raw: bytes) -> dict:
    """raw: first >=544 decompressed bytes of a .nii. Returns None if not NIfTI."""
    if len(raw) < 8:
        return None
    info = _parse_nifti1(raw)
    if info is not None:
        return info
    return _parse_nifti2(raw)


def read_header_from_stream(fileobj) -> dict:
    """fileobj: a binary stream of .nii.gz content. Decompress only header bytes."""
    try:
        with gzip.GzipFile(fileobj=fileobj) as gz:
            head = gz.read(560)   # enough for NIfTI-2 (544) + margin
    except (OSError, EOFError, gzip.BadGzipFile):
        # maybe uncompressed .nii
        fileobj.seek(0)
        head = fileobj.read(560)
    return parse_nifti_header(head)


# ============================================================== #
# Per-source processing
# ============================================================== #

def process_zip(zip_path: str) -> list:
    """Return list of dicts, one per nii.gz inside the zip."""
    rows = []
    try:
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if not name.endswith(".nii.gz") and not name.endswith(".nii"):
                    continue
                try:
                    with z.open(name) as f:
                        # read into BytesIO so we can seek if needed
                        data = io.BytesIO(f.read(4096))  # header lives in first few KB
                    info = read_header_from_stream(data)
                except Exception as ex:
                    info = None
                rows.append(_make_row(zip_path, name, info))
    except (zipfile.BadZipFile, OSError) as ex:
        rows.append({"source": zip_path, "member": "", "status": f"bad_zip:{ex}",
                     "ndim": None, "shape": "", "spacing": "",
                     "sx": None, "sy": None, "sz": None,
                     "nx": None, "ny": None, "nz": None, "nt": None})
    return rows


def process_loose_nii(nii_path: str) -> list:
    """A .nii.gz sitting on disk (not in a zip)."""
    try:
        with open(nii_path, "rb") as f:
            data = io.BytesIO(f.read(4096))
        info = read_header_from_stream(data)
    except Exception as ex:
        info = None
    return [_make_row(nii_path, os.path.basename(nii_path), info)]


def _make_row(source: str, member: str, info: dict) -> dict:
    if info is None:
        return {"source": source, "member": member, "status": "parse_fail",
                "ndim": None, "shape": "", "spacing": "",
                "sx": None, "sy": None, "sz": None,
                "nx": None, "ny": None, "nz": None, "nt": None}
    shape = info["shape"]
    spacing = info["spacing"]
    # spatial dims: first 3
    nx = shape[0] if len(shape) > 0 else None
    ny = shape[1] if len(shape) > 1 else None
    nz = shape[2] if len(shape) > 2 else None
    nt = shape[3] if len(shape) > 3 else None
    sx = spacing[0] if len(spacing) > 0 else None
    sy = spacing[1] if len(spacing) > 1 else None
    sz = spacing[2] if len(spacing) > 2 else None
    return {
        "source": source, "member": member, "status": "ok",
        "ndim": info["ndim"],
        "shape": "x".join(str(s) for s in shape),
        "spacing": "x".join(str(s) for s in spacing),
        "sx": sx, "sy": sy, "sz": sz,
        "nx": nx, "ny": ny, "nz": nz, "nt": nt,
    }


# ============================================================== #
# Discovery
# ============================================================== #

def find_sources(root: str) -> list:
    """Find all session zips AND loose nii.gz under root."""
    zips = glob.glob(os.path.join(root, "**", "*.zip"), recursive=True)
    # loose nii.gz that are NOT inside any zip (already extracted datasets)
    loose = glob.glob(os.path.join(root, "**", "*.nii.gz"), recursive=True)
    return zips, loose


# ============================================================== #
# Analysis / reporting
# ============================================================== #

def classify_isotropy(sx, sy, sz, tol=0.15):
    """Return 'isotropic' | 'anisotropic' | 'in_plane_iso' based on spacing."""
    if sx is None or sy is None or sz is None:
        return "unknown"
    vals = [sx, sy, sz]
    mn, mx = min(vals), max(vals)
    if mx <= 0:
        return "unknown"
    if (mx - mn) / mx <= tol:
        return "isotropic"
    # in-plane isotropic but thick slices (common clinical 2D)
    if abs(sx - sy) / max(sx, sy) <= tol and sz > max(sx, sy) * (1 + tol):
        return "in_plane_iso_thick_slice"
    return "anisotropic"


def summarize(csv_path: str, out_dir: str) -> None:
    import pandas as pd
    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "ok"].copy()

    print("\n" + "=" * 60)
    print("GEOMETRY SURVEY SUMMARY")
    print("=" * 60)
    print(f"  Total records     : {len(df):,}")
    print(f"  Successfully read  : {len(ok):,} ({100*len(ok)/max(len(df),1):.1f}%)")
    n_fail = (df["status"] != "ok").sum()
    if n_fail:
        print(f"  Failed / bad      : {n_fail:,}")
        df[df["status"] != "ok"]["status"].value_counts().head(10).to_csv(
            os.path.join(out_dir, "failures.csv"))

    if len(ok) == 0:
        print("  No readable NIfTI headers — check paths.")
        return

    # ---- Q1: dimensionality ----
    print("\n  [Q1] Dimensionality (ndim):")
    dim_counts = ok["ndim"].value_counts().sort_index()
    for d, c in dim_counts.items():
        label = {2: "2D", 3: "3D", 4: "4D (series: fMRI/DWI)"}.get(int(d), f"{int(d)}D")
        print(f"    {label:25s} {c:>8,} ({100*c/len(ok):.1f}%)")
    dim_counts.to_csv(os.path.join(out_dir, "dimensionality.csv"))

    # Focus spatial analysis on 3D volumes (main distillation target)
    vol3d = ok[ok["ndim"] == 3].copy()
    print(f"\n  → 3D volumes (distillation 대상): {len(vol3d):,}")

    # ---- Q2: spacing distribution ----
    print("\n  [Q2] Spacing distribution (3D volumes, mm):")
    for axis, col in [("in-plane X", "sx"), ("in-plane Y", "sy"), ("slice Z", "sz")]:
        s = pd.to_numeric(vol3d[col], errors="coerce").dropna()
        s = s[(s > 0) & (s < 20)]  # sanity
        if len(s):
            print(f"    {axis:12s}: median={s.median():.2f}  "
                  f"IQR=[{s.quantile(.25):.2f},{s.quantile(.75):.2f}]  "
                  f"range=[{s.min():.2f},{s.max():.2f}]")

    # spacing binning on Z (slice thickness — most variable)
    sz = pd.to_numeric(vol3d["sz"], errors="coerce")
    sz = sz[(sz > 0) & (sz < 20)]
    bins = [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 20]
    labels = ["≤0.5", "0.5-0.8", "0.8-1.0", "1.0-1.2", "1.2-1.5",
              "1.5-2.0", "2.0-3.0", "3.0-5.0", ">5.0"]
    sz_bin = pd.cut(sz, bins=bins, labels=labels)
    print("\n    Slice spacing (Z) 분포:")
    for lbl, c in sz_bin.value_counts().sort_index().items():
        pct = 100 * c / len(sz)
        bar = "█" * int(pct / 2)
        print(f"      {str(lbl):10s} {c:>8,} ({pct:>5.1f}%) {bar}")
    sz_bin.value_counts().sort_index().to_csv(os.path.join(out_dir, "z_spacing_bins.csv"))

    # ---- Q3: isotropy ----
    print("\n  [Q3] Isotropy (3D volumes):")
    vol3d["_iso"] = vol3d.apply(
        lambda r: classify_isotropy(r["sx"], r["sy"], r["sz"]), axis=1)
    iso_counts = vol3d["_iso"].value_counts()
    for k, c in iso_counts.items():
        print(f"    {str(k):28s} {c:>8,} ({100*c/len(vol3d):.1f}%)")
    iso_counts.to_csv(os.path.join(out_dir, "isotropy.csv"))

    # near-1mm-isotropic (친화적 케이스)
    def is_near_1mm(r):
        try:
            return (max(abs(r["sx"]-1), abs(r["sy"]-1), abs(r["sz"]-1)) <= 0.15)
        except Exception:
            return False
    n_1mm = vol3d.apply(is_near_1mm, axis=1).sum()
    print(f"\n    near-1mm-isotropic (±0.15): {n_1mm:,} "
          f"({100*n_1mm/len(vol3d):.1f}%)  ← resampling 최소 변형")

    # ---- Q5: shape / voxel count ----
    print("\n  [Q5] Volume shape (3D):")
    for axis, col in [("nx", "nx"), ("ny", "ny"), ("nz", "nz")]:
        s = pd.to_numeric(vol3d[col], errors="coerce").dropna()
        if len(s):
            print(f"    {axis}: median={int(s.median())}  "
                  f"range=[{int(s.min())},{int(s.max())}]")
    # most common shapes
    print("\n    Top 15 exact shapes (3D):")
    top_shapes = vol3d["shape"].value_counts().head(15)
    for shp, c in top_shapes.items():
        print(f"      {shp:20s} {c:>8,} ({100*c/len(vol3d):.1f}%)")
    vol3d["shape"].value_counts().to_csv(os.path.join(out_dir, "shape_distribution.csv"))

    # top spacing tuples
    print("\n    Top 15 exact spacings (3D):")
    top_sp = vol3d["spacing"].value_counts().head(15)
    for sp, c in top_sp.items():
        print(f"      {sp:22s} {c:>8,} ({100*c/len(vol3d):.1f}%)")
    vol3d["spacing"].value_counts().to_csv(os.path.join(out_dir, "spacing_distribution.csv"))

    # ---- 1mm resampling impact estimate ----
    print("\n  [1mm resampling 영향 추정]")
    hi_res = (pd.to_numeric(vol3d["sz"], errors="coerce") < 0.7).sum()
    lo_res = (pd.to_numeric(vol3d["sz"], errors="coerce") > 2.0).sum()
    print(f"    초고해상도 (Z<0.7mm, downsample 손실): {hi_res:,} ({100*hi_res/len(vol3d):.1f}%)")
    print(f"    저해상도    (Z>2.0mm, upsample 흐림)  : {lo_res:,} ({100*lo_res/len(vol3d):.1f}%)")
    print(f"    → VesselFM 등 fine-structure teacher는 hi-res 스캔에서 1mm 손실 주의")

    _write_md(out_dir, len(df), len(ok), dim_counts, iso_counts, n_1mm, len(vol3d))


def _write_md(out_dir, n_total, n_ok, dim_counts, iso_counts, n_1mm, n_3d):
    lines = ["# FOMO300K Geometry Survey\n"]
    lines.append(f"- Total records: {n_total:,}")
    lines.append(f"- Readable: {n_ok:,}")
    lines.append(f"- 3D volumes: {n_3d:,}\n")
    lines.append("## Dimensionality")
    for d, c in dim_counts.items():
        lines.append(f"- {int(d)}D: {c:,}")
    lines.append(f"\n## Isotropy (3D)")
    for k, c in iso_counts.items():
        lines.append(f"- {k}: {c:,}")
    lines.append(f"\n- near-1mm-isotropic: {n_1mm:,} ({100*n_1mm/max(n_3d,1):.1f}%)")
    lines.append("\n## CSVs")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".csv"):
            lines.append(f"- `{f}`")
    with open(os.path.join(out_dir, "GEOMETRY_SUMMARY.md"), "w") as fh:
        fh.write("\n".join(lines))


# ============================================================== #
# Main
# ============================================================== #

def _worker(source_and_kind):
    source, kind = source_and_kind
    if kind == "zip":
        return process_zip(source)
    else:
        return process_loose_nii(source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/data/FOMO300K")
    ap.add_argument("--out_dir", default="/root/FOMO26/examples/geo_survey")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--limit", type=int, default=None,
                    help="process only first N sources (smoke test)")
    ap.add_argument("--skip_loose", action="store_true",
                    help="only scan zips, ignore loose .nii.gz (faster if all data is zipped)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Scanning {args.root} for sources ...")
    zips, loose = find_sources(args.root)
    print(f"  found {len(zips):,} zip archives")
    if args.skip_loose:
        loose = []
    print(f"  found {len(loose):,} loose .nii.gz")

    sources = [(z, "zip") for z in zips] + [(n, "loose") for n in loose]
    if args.limit:
        sources = sources[:args.limit]
        print(f"  LIMIT: processing only first {len(sources)} sources")

    if not sources:
        print("No sources found. Check --root path and structure.")
        sys.exit(1)

    csv_path = os.path.join(args.out_dir, "per_scan_geometry.csv")
    fieldnames = ["source", "member", "status", "ndim", "shape", "spacing",
                  "sx", "sy", "sz", "nx", "ny", "nz", "nt"]

    n_written = 0
    with open(csv_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_worker, s): s for s in sources}
            done = 0
            for fut in as_completed(futures):
                try:
                    rows = fut.result()
                except Exception as e:
                    src = futures[fut][0]
                    rows = [{"source": src, "member": "", "status": f"worker_err:{e}",
                             "ndim": None, "shape": "", "spacing": "",
                             "sx": None, "sy": None, "sz": None,
                             "nx": None, "ny": None, "nz": None, "nt": None}]
                for r in rows:
                    writer.writerow(r)
                    n_written += 1
                done += 1
                if done % 500 == 0:
                    print(f"  processed {done:,}/{len(sources):,} sources "
                          f"({n_written:,} scans)", flush=True)

    print(f"\n  wrote {n_written:,} scan records → {csv_path}")

    # analysis
    try:
        summarize(csv_path, args.out_dir)
    except ImportError:
        print("  (pandas not available — raw CSV written, skip summary. "
              "Run summarize separately with pandas installed.)")

    print(f"\nAll outputs → {args.out_dir}")


if __name__ == "__main__":
    main()