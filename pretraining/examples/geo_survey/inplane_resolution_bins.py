"""In-plane (X,Y) resolution binning, requested 2026-07-23 as a follow-up to
survey_geometry.py's Z-spacing binning (z_spacing_bins.csv). Reuses the
already-collected per_scan_geometry.csv (no re-scan of the raw zips needed).

In-plane resolution per scan = mean(sx, sy) -- the two are almost always
equal in practice (see geo_survey/isotropy.csv: isotropic + in_plane_iso_thick_slice
= 291,237 / 306,090 3D volumes have sx==sy within tolerance), so mean vs.
either individual axis makes negligible difference.
"""

import os

import pandas as pd

GEO_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(GEO_DIR, "per_scan_geometry.csv")

df = pd.read_csv(CSV_PATH)
ok = df[df["status"] == "ok"]
vol3d = ok[ok["ndim"] == 3].copy()

sx = pd.to_numeric(vol3d["sx"], errors="coerce")
sy = pd.to_numeric(vol3d["sy"], errors="coerce")
inplane = (sx + sy) / 2
inplane = inplane[(inplane > 0) & (inplane < 20)]

bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 20]
labels = ["≤0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0", "1.0-1.5", ">1.5"]
inplane_bin = pd.cut(inplane, bins=bins, labels=labels)

print(f"3D volumes considered: {len(inplane):,}")
print("\nIn-plane (X,Y mean) resolution 분포:")
counts = inplane_bin.value_counts().sort_index()
for lbl, c in counts.items():
    pct = 100 * c / len(inplane)
    bar = "█" * int(pct / 2)
    print(f"  {str(lbl):10s} {c:>8,} ({pct:>5.1f}%) {bar}")

out_path = os.path.join(GEO_DIR, "inplane_resolution_bins.csv")
counts.to_csv(out_path)
print(f"\nsaved: {out_path}")
