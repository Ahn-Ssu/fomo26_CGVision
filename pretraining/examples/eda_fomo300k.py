"""
FOMO300K mri_info.tsv — Exploratory Data Analysis
=================================================

목적: teacher 적용 전략 수립에 필요한 통계를 뽑는다.

핵심 질문:
  Q1. Modality/sequence 분포는? → 각 teacher의 실질 coverage
  Q2. VesselFM 적용 가능한 MRA/TOF/angio 스캔이 얼마나 있는가? (실효성 판단)
  Q3. BraTS teacher episode weighting을 위한 modality 분류가 가능한가?
      (T1/T2/FLAIR/기타로 얼마나 깨끗하게 나뉘는가)
  Q4. SliceThickness / spacing 분포는? → 1mm resampling의 영향 범위
  Q5. 2D vs 3D acquisition 비율은?
  Q6. SeriesDescription이 얼마나 신뢰할 수 있는가 (결측/자유텍스트 정도)

사용법:
  python eda_fomo300k.py --tsv /root/data/FOMO300K/mri_info.tsv --out_dir /root/data/fomo_eda

출력:
  - 콘솔 요약 리포트
  - out_dir/*.csv  (각 분포표)
  - out_dir/EDA_SUMMARY.md
"""

import argparse
import os
import re
import numpy as np
import pandas as pd


# ============================================================== #
# Modality classification heuristics
# ============================================================== #
# FOMO300K는 임상 데이터라 Modality 컬럼이 대부분 "MR"로만 찍히고,
# 실제 시퀀스 종류는 SeriesDescription / ProtocolName / filename의
# BIDS suffix (_T1w, _T2w, _FLAIR 등)에서 추론해야 한다.
# 세 소스를 결합해서 최대한 robust하게 분류한다.

# BIDS suffix (filename 기반) — 가장 신뢰도 높음
BIDS_SUFFIX_PATTERNS = {
    "T1w":    re.compile(r"_T1w\.nii", re.I),
    "T2w":    re.compile(r"_T2w\.nii", re.I),
    "FLAIR":  re.compile(r"_FLAIR\.nii", re.I),
    "T1c":    re.compile(r"_(ce-\w+_)?T1w\.nii", re.I),   # contrast-enhanced 별도 처리
    "T2star": re.compile(r"_T2star\.nii", re.I),
    "PD":     re.compile(r"_PD\w*\.nii", re.I),
    "DWI":    re.compile(r"_dwi\.nii", re.I),
    "ADC":    re.compile(r"_adc\.nii", re.I),
    "SWI":    re.compile(r"_(swi|SWI)\.nii", re.I),
    "angio":  re.compile(r"_(angio|MRA|TOF|tof)\.nii", re.I),
    "bold":   re.compile(r"_bold\.nii", re.I),
}

# SeriesDescription / ProtocolName 자유텍스트 기반 키워드
# (BIDS suffix로 못 잡은 경우 fallback)
TEXT_KEYWORDS = {
    "FLAIR":  [r"\bflair\b", r"fluid.?attenuat"],
    "T1c":    [r"t1.?c\b", r"t1.?gd", r"t1.?post", r"\+c\b", r"contrast", r"gad", r"ce.?t1", r"mprage.?gd"],
    "T1w":    [r"\bt1\b", r"mprage", r"spgr", r"tfl", r"t1.?3d"],
    "T2w":    [r"\bt2\b", r"tse", r"turbo.?spin", r"t2.?3d"],
    "T2star": [r"t2.?\*", r"t2.?star", r"gre"],
    "PD":     [r"\bpd\b", r"proton.?density"],
    "DWI":    [r"\bdwi\b", r"diffusion", r"\bdti\b", r"trace"],
    "ADC":    [r"\badc\b", r"apparent.?diffusion"],
    "SWI":    [r"\bswi\b", r"suscept", r"\bswan\b"],
    "angio":  [r"\bmra\b", r"\btof\b", r"angio", r"time.?of.?flight", r"venogram", r"\bmrv\b"],
    "bold":   [r"\bbold\b", r"resting.?state", r"\bfmri\b", r"\brs.?fmri\b"],
    "perfusion": [r"perfusion", r"\bpwi\b", r"\basl\b", r"\bdsc\b", r"\bdce\b"],
}


def classify_modality(row: pd.Series) -> str:
    """
    filename BIDS suffix → SeriesDescription → ProtocolName 순으로
    modality를 추론한다. contrast-enhanced T1은 T1c로 구분.
    """
    fname = str(row.get("filename", "") or "")
    series = str(row.get("SeriesDescription", "") or "").lower()
    protocol = str(row.get("ProtocolName", "") or "").lower()

    # 1) BIDS suffix — contrast-enhanced 우선 체크
    # BIDS: ce-<label> entity가 있으면 contrast-enhanced
    if re.search(r"_ce-\w+_T1w\.nii", fname, re.I):
        return "T1c"
    for mod, pat in BIDS_SUFFIX_PATTERNS.items():
        if mod == "T1c":
            continue  # 위에서 처리
        if pat.search(fname):
            # angio suffix는 그대로
            return mod

    # 2) 자유텍스트 (SeriesDescription 우선, 그다음 ProtocolName)
    combined = series + " || " + protocol
    # 우선순위: 더 구체적인 것부터 (T1c, FLAIR, angio를 T1/T2보다 먼저)
    priority_order = ["T1c", "FLAIR", "angio", "SWI", "T2star", "ADC", "DWI",
                      "PD", "perfusion", "bold", "T2w", "T1w"]
    for mod in priority_order:
        for kw in TEXT_KEYWORDS.get(mod, []):
            if re.search(kw, combined):
                return mod

    return "unknown"


# BraTS teacher가 신뢰할 수 있는 modality → confidence weight
# (fold=all 재학습 후 값 업데이트 필요; 현재는 5-fold 기준)
BRATS_TEACHER_MODALITIES = {"T1w", "T2w", "FLAIR", "T1c"}
VESSELFM_MODALITIES = {"angio"}   # MRA/TOF만 실질적으로 의미
ANATOMIX_MODALITIES = "any"       # modality-agnostic


# ============================================================== #
# Analysis functions
# ============================================================== #

def load_tsv(path: str) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")
    print(f"  columns: {list(df.columns)}")
    return df


def numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    """문자열 컬럼을 float로 강제 변환 (실패는 NaN)."""
    return pd.to_numeric(df[col], errors="coerce")


def basic_counts(df: pd.DataFrame, out_dir: str) -> dict:
    summary = {}
    summary["n_scans"] = len(df)
    summary["n_datasets"] = df["dataset"].nunique() if "dataset" in df else None
    summary["n_participants"] = (
        df.groupby("dataset")["participant_id"].nunique().sum()
        if {"dataset", "participant_id"}.issubset(df.columns) else None
    )
    summary["n_sessions"] = (
        df.groupby(["dataset", "participant_id"])["session_id"].nunique().sum()
        if {"dataset", "participant_id", "session_id"}.issubset(df.columns) else None
    )

    print("\n" + "="*60)
    print("BASIC COUNTS")
    print("="*60)
    print(f"  Total scans      : {summary['n_scans']:,}")
    print(f"  Datasets         : {summary['n_datasets']}")
    print(f"  Participants     : {summary['n_participants']}")
    print(f"  Sessions         : {summary['n_sessions']}")

    # per-dataset scan counts
    if "dataset" in df:
        ds_counts = df["dataset"].value_counts()
        ds_counts.to_csv(os.path.join(out_dir, "scans_per_dataset.csv"))
        print(f"\n  Top 10 datasets by scan count:")
        for name, cnt in ds_counts.head(10).items():
            print(f"    {name:40s} {cnt:>8,}")
        print(f"  ... ({len(ds_counts)} datasets total)")

    return summary


def modality_analysis(df: pd.DataFrame, out_dir: str) -> dict:
    print("\n" + "="*60)
    print("MODALITY / SEQUENCE CLASSIFICATION")
    print("="*60)

    # raw Modality column (거의 다 'MR'일 것)
    if "Modality" in df:
        print("\n  Raw 'Modality' column (DICOM tag):")
        for val, cnt in df["Modality"].value_counts(dropna=False).head(10).items():
            print(f"    {str(val):20s} {cnt:>8,}")

    # 추론된 modality
    print("\n  Classifying modality (filename BIDS + SeriesDescription + ProtocolName)...")
    df["_modality"] = df.apply(classify_modality, axis=1)

    mod_counts = df["_modality"].value_counts(dropna=False)
    mod_pct = (mod_counts / len(df) * 100).round(2)
    mod_table = pd.DataFrame({"count": mod_counts, "pct": mod_pct})
    mod_table.to_csv(os.path.join(out_dir, "modality_distribution.csv"))

    print("\n  Inferred modality distribution:")
    for mod, row in mod_table.iterrows():
        print(f"    {str(mod):12s} {int(row['count']):>8,}  ({row['pct']:>5.1f}%)")

    # unknown 비율이 높으면 경고
    unk_pct = mod_pct.get("unknown", 0)
    if unk_pct > 15:
        print(f"\n  ⚠ WARNING: {unk_pct:.1f}% classified as 'unknown' — "
              f"classification heuristics may need tuning.")
        # unknown 샘플의 SeriesDescription 예시 출력
        unk = df[df["_modality"] == "unknown"]
        print("  Sample SeriesDescription of unknowns (for heuristic tuning):")
        sample_desc = unk["SeriesDescription"].dropna().value_counts().head(20)
        for desc, cnt in sample_desc.items():
            print(f"    [{cnt:>4}] {desc[:70]}")
        sample_desc.to_csv(os.path.join(out_dir, "unknown_series_descriptions.csv"))

    return {"modality_table": mod_table, "unknown_pct": float(unk_pct)}


def teacher_coverage(df: pd.DataFrame, out_dir: str) -> dict:
    print("\n" + "="*60)
    print("TEACHER COVERAGE ANALYSIS  ★ 핵심")
    print("="*60)

    total = len(df)
    mod = df["_modality"]

    # VesselFM: MRA/TOF/angio only
    vesselfm_mask = mod.isin(VESSELFM_MODALITIES)
    n_vesselfm = int(vesselfm_mask.sum())

    # BraTS teacher: T1/T2/FLAIR/T1c
    brats_mask = mod.isin(BRATS_TEACHER_MODALITIES)
    n_brats = int(brats_mask.sum())

    # Anatomix: any (modality-agnostic) → 전체
    n_anatomix = total

    # BraTS modality별 세부 (confidence weighting 근거)
    brats_by_mod = df[brats_mask]["_modality"].value_counts()

    print(f"\n  Total scans: {total:,}\n")
    print(f"  ┌─ Anatomix (modality-agnostic)")
    print(f"  │    coverage: {n_anatomix:,} ({100*n_anatomix/total:.1f}%)  — 전체")
    print(f"  │")
    print(f"  ├─ BraTS teacher (T1/T2/FLAIR/T1c)")
    print(f"  │    coverage: {n_brats:,} ({100*n_brats/total:.1f}%)")
    for m in ["FLAIR", "T2w", "T1w", "T1c"]:
        c = int(brats_by_mod.get(m, 0))
        print(f"  │      {m:8s}: {c:>8,} ({100*c/total:.1f}%)")
    print(f"  │")
    print(f"  └─ VesselFM (MRA/TOF/angio only)  ★ 실효성 판단")
    print(f"       coverage: {n_vesselfm:,} ({100*n_vesselfm/total:.2f}%)")

    # VesselFM 실효성 경고
    print()
    if n_vesselfm == 0:
        print("  ⚠⚠ CRITICAL: MRA/TOF/angio 스캔이 0개.")
        print("     → VesselFM을 vessel-visible 모달리티에 적용할 수 없음.")
        print("     → VesselFM teacher 채택을 재고해야 함.")
    elif 100*n_vesselfm/total < 1:
        print(f"  ⚠ WARNING: MRA/TOF 비율이 {100*n_vesselfm/total:.2f}%로 매우 낮음.")
        print("     → VesselFM teacher의 기여가 제한적. 다음 중 하나 고려:")
        print("        (a) angio 스캔에만 VesselFM episode 적용 (coverage 작음)")
        print("        (b) VesselFM을 T1/T2에도 적용 — 단 vessel이 잘 안 보여 pseudo-label 품질 저하")
        print("        (c) VesselFM 대신 다른 fine-grained teacher 고려")
    else:
        print(f"  ✓ MRA/TOF {100*n_vesselfm/total:.1f}% — VesselFM 적용 가능한 스캔 충분")

    coverage_table = pd.DataFrame({
        "teacher": ["Anatomix", "BraTS", "VesselFM"],
        "coverage_scans": [n_anatomix, n_brats, n_vesselfm],
        "coverage_pct": [100*n_anatomix/total, 100*n_brats/total, 100*n_vesselfm/total],
    })
    coverage_table.to_csv(os.path.join(out_dir, "teacher_coverage.csv"), index=False)

    return {
        "n_vesselfm": n_vesselfm, "n_brats": n_brats, "n_anatomix": n_anatomix,
        "vesselfm_pct": 100*n_vesselfm/total,
    }


def spacing_analysis(df: pd.DataFrame, out_dir: str) -> dict:
    print("\n" + "="*60)
    print("SPACING / RESOLUTION ANALYSIS  (1mm resampling 영향)")
    print("="*60)

    result = {}

    # SliceThickness
    if "SliceThickness" in df:
        st = numeric_col(df, "SliceThickness")
        # sanity filter: brain MRI slice thickness is physically < ~20mm.
        # Values above this are almost always mis-recorded (e.g. TR leaked
        # into the column, unit errors). Flag and exclude them.
        n_implausible = int((st > 20).sum())
        if n_implausible > 0:
            print(f"    (note) {n_implausible:,} scans have SliceThickness > 20mm "
                  f"(likely mis-recorded) — excluded from stats")
            st = st.where(st <= 20)
        st_valid = st.dropna()
        result["slice_thickness"] = {
            "n_valid": int(len(st_valid)),
            "median": float(st_valid.median()) if len(st_valid) else None,
            "p25": float(st_valid.quantile(0.25)) if len(st_valid) else None,
            "p75": float(st_valid.quantile(0.75)) if len(st_valid) else None,
        }
        print(f"\n  SliceThickness (mm):")
        print(f"    valid: {len(st_valid):,} / {len(df):,} "
              f"({100*len(st_valid)/len(df):.1f}%)")
        if len(st_valid):
            print(f"    median={st_valid.median():.2f}, "
                  f"IQR=[{st_valid.quantile(0.25):.2f}, {st_valid.quantile(0.75):.2f}]")
            # 두께 구간별 분포
            bins = [0, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf]
            labels = ["≤1.0", "1.0-1.5", "1.5-2.0", "2.0-3.0", "3.0-5.0", ">5.0"]
            st_binned = pd.cut(st_valid, bins=bins, labels=labels, right=True)
            print(f"\n    SliceThickness 분포:")
            for lbl, cnt in st_binned.value_counts().sort_index().items():
                pct = 100*cnt/len(st_valid)
                bar = "█" * int(pct/2)
                print(f"      {str(lbl):10s} {cnt:>8,} ({pct:>5.1f}%) {bar}")
            st_binned.value_counts().sort_index().to_csv(
                os.path.join(out_dir, "slice_thickness_bins.csv"))

            # clinical-grade (≥3mm) 비율 — FOMO300K 정의
            clinical = int((st_valid >= 3.0).sum())
            print(f"\n    clinical-grade (ST≥3mm): {clinical:,} "
                  f"({100*clinical/len(st_valid):.1f}%)")
            # FastSurfer 적용 가능 (≤1.5mm) 비율 — 참고용
            hires = int((st_valid <= 1.5).sum())
            print(f"    hi-res (ST≤1.5mm)       : {hires:,} "
                  f"({100*hires/len(st_valid):.1f}%)")

    # MRAcquisitionType (2D vs 3D)
    if "MRAcquisitionType" in df:
        print(f"\n  MRAcquisitionType (2D vs 3D):")
        acq = df["MRAcquisitionType"].value_counts(dropna=False)
        for val, cnt in acq.items():
            print(f"    {str(val):10s} {cnt:>8,} ({100*cnt/len(df):.1f}%)")
        acq.to_csv(os.path.join(out_dir, "acquisition_type.csv"))

    return result


def field_strength_analysis(df: pd.DataFrame, out_dir: str) -> None:
    if "MagneticFieldStrength" not in df:
        return
    print("\n" + "="*60)
    print("FIELD STRENGTH / SCANNER")
    print("="*60)

    fs = numeric_col(df, "MagneticFieldStrength").dropna()
    if len(fs):
        print(f"\n  MagneticFieldStrength (T):")
        # round to common values
        fs_rounded = fs.round(1)
        for val, cnt in fs_rounded.value_counts().sort_index().items():
            print(f"    {val:>4.1f}T  {cnt:>8,} ({100*cnt/len(fs):.1f}%)")
        fs_rounded.value_counts().sort_index().to_csv(
            os.path.join(out_dir, "field_strength.csv"))

    if "Manufacturer" in df:
        print(f"\n  Manufacturer:")
        for val, cnt in df["Manufacturer"].value_counts(dropna=False).head(8).items():
            print(f"    {str(val):25s} {cnt:>8,} ({100*cnt/len(df):.1f}%)")
        df["Manufacturer"].value_counts(dropna=False).to_csv(
            os.path.join(out_dir, "manufacturer.csv"))


def metadata_completeness(df: pd.DataFrame, out_dir: str) -> None:
    print("\n" + "="*60)
    print("METADATA COMPLETENESS  (어떤 필드를 신뢰할 수 있는가)")
    print("="*60)
    print()
    completeness = {}
    for col in df.columns:
        if col.startswith("_"):
            continue
        n_missing = df[col].isna().sum() + (df[col] == "").sum()
        pct_present = 100 * (1 - n_missing / len(df))
        completeness[col] = pct_present

    comp_series = pd.Series(completeness).sort_values(ascending=False)
    for col, pct in comp_series.items():
        flag = "✓" if pct > 90 else ("~" if pct > 50 else "✗")
        print(f"    {flag} {col:28s} {pct:>5.1f}%")
    comp_series.to_csv(os.path.join(out_dir, "metadata_completeness.csv"))


def cross_tab_modality_dataset(df: pd.DataFrame, out_dir: str) -> None:
    """어떤 dataset에 어떤 modality가 몰려있는지 — sampling 전략에 유용."""
    if not {"dataset", "_modality"}.issubset(df.columns):
        return
    print("\n" + "="*60)
    print("MODALITY × DATASET (angio/FLAIR가 어디 몰려있는가)")
    print("="*60)

    ct = pd.crosstab(df["dataset"], df["_modality"])
    ct.to_csv(os.path.join(out_dir, "modality_by_dataset.csv"))

    # angio가 있는 dataset만 출력 (VesselFM 적용 대상 파악)
    if "angio" in ct.columns:
        angio_datasets = ct[ct["angio"] > 0]["angio"].sort_values(ascending=False)
        if len(angio_datasets):
            print(f"\n  angio/MRA/TOF 스캔이 있는 dataset:")
            for name, cnt in angio_datasets.items():
                print(f"    {name:40s} {cnt:>6,} angio scans")
        else:
            print("\n  angio 스캔이 있는 dataset 없음.")

    # FLAIR 분포 (BraTS teacher 최고 신뢰 modality)
    if "FLAIR" in ct.columns:
        flair_total = int(ct["FLAIR"].sum())
        print(f"\n  FLAIR 총 {flair_total:,} scans "
              f"(BraTS teacher 최고 신뢰 modality, NSD 0.824)")


def write_summary_md(summary: dict, out_dir: str) -> None:
    lines = ["# FOMO300K EDA Summary\n"]
    lines.append(f"- Total scans: {summary.get('n_scans', 'N/A'):,}")
    lines.append(f"- Datasets: {summary.get('n_datasets', 'N/A')}")
    lines.append(f"- Participants: {summary.get('n_participants', 'N/A')}")
    lines.append(f"- Unknown modality: {summary.get('unknown_pct', 'N/A')}%\n")
    lines.append("## Teacher coverage\n")
    lines.append(f"- Anatomix: {summary.get('n_anatomix', 'N/A'):,} (100%)")
    lines.append(f"- BraTS: {summary.get('n_brats', 'N/A'):,}")
    lines.append(f"- VesselFM (angio only): {summary.get('n_vesselfm', 'N/A'):,} "
                 f"({summary.get('vesselfm_pct', 0):.2f}%)\n")
    lines.append("## Generated CSVs\n")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".csv"):
            lines.append(f"- `{f}`")
    with open(os.path.join(out_dir, "EDA_SUMMARY.md"), "w") as f:
        f.write("\n".join(lines))


# ============================================================== #
# Main
# ============================================================== #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="/root/data/FOMO300K/mri_info.tsv")
    ap.add_argument("--out_dir", default="/root/data/fomo_eda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_tsv(args.tsv)

    summary = {}
    summary.update(basic_counts(df, args.out_dir))
    mod_res = modality_analysis(df, args.out_dir)
    summary["unknown_pct"] = mod_res["unknown_pct"]
    cov = teacher_coverage(df, args.out_dir)
    summary.update(cov)
    spacing_analysis(df, args.out_dir)
    field_strength_analysis(df, args.out_dir)
    metadata_completeness(df, args.out_dir)
    cross_tab_modality_dataset(df, args.out_dir)

    # 분류 결과가 담긴 전체 테이블 저장 (downstream에서 modality별 필터링에 사용)
    keep_cols = ["dataset", "participant_id", "session_id", "filename", "_modality"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_csv(os.path.join(args.out_dir, "scan_modality_labels.csv"), index=False)
    print(f"\n  Saved per-scan modality labels → scan_modality_labels.csv")
    print(f"    (이 파일이 episodic sampler의 modality-aware weighting 입력이 됨)")

    write_summary_md(summary, args.out_dir)
    print(f"\n{'='*60}")
    print(f"All outputs → {args.out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()