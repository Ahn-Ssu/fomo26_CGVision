"""Shared asparagus-equivalent normalization, used identically by the
preprocessing script, the training Dataset, and verify_teachers.py so there
is exactly one implementation to keep in sync (previously duplicated in three
places -- consolidated here 2026-07-16).

Verbatim port of asparagus_preprocessing/asparagus_preprocessing/utils/
normalize.py's `scheme="volume_wise_znorm"`: clamp(foreground, q=0.99) ->
z-score(foreground) -> rescale to [0,1].
"""

from typing import Optional

import numpy as np
from skimage import exposure


def asparagus_volume_wise_znorm(array: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """mask: optional explicit foreground mask (bool array, same shape as
    array). If given, used directly instead of the `array != array.min()`
    heuristic -- preferred whenever an explicit mask is available (e.g. the
    one saved by data/preprocess_raw.py, computed pre-resample and therefore
    not contaminated by cubic-interpolation boundary leakage; see
    /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/03_PREPROCESSING_REPORT.md Issue A)."""

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

    if mask is not None:
        m = mask.astype(bool)
    else:
        empty_val = array.min()
        m = array != empty_val

    if m.sum() == 0:
        return array.astype(np.float32)
    out = clamp(array, m)
    out = znormalize(out, m)
    out = rescale(out, out_range=(0, 1))
    return out.astype(np.float32)
