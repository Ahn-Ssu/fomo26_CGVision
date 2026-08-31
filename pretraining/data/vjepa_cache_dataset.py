"""Loader for the precomputed V-JEPA feature cache (see
/root/external_teacher_probe/extract_vjepa_cache.py for how it was built -- 30,000 samples,
stratified across all FOMO300K_preprocessed datasets, one random foreground-biased 128^3 crop per
subject). No live V-JEPA model runs during training -- this dataset returns the frozen teacher's
output directly, already computed offline.

Axis-order fix (found 2026-08-02, during this integration -- see chat history): the cache script's
saved feature arrays are (T,H,W,D) with T = the tubelet/temporal axis, which corresponds to the
volume's Z axis (vit_forward permutes Z->T before tokenizing). Every OTHER feature tensor in this
codebase (student encoder/decoder stages, the raw image crop, the other 3 teachers' features) uses
(X,Y,Z,D)-equivalent ordering with Z LAST, matching probe_native_teachers.py's established
"(Dx,Hy,Wz,C), Wz~axial(Z)" convention. Saved-as-is, the cache's spatial axes are silently
mislabeled relative to the student's own axes (position [i,j,k] means physically different things
on each side) -- confirmed via direct comparison against a freshly-recomputed sample
(see chat history: recompute matched cached exactly, and permuting both by (1,2,0,3) still
matched, proving the fix commutes with the pooling already baked into the cache and doesn't
require re-extraction). This dataset applies that permute on load, so callers never see the bug.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

# V-JEPA hierarchical layer -> STAGE_MAP's "vjepa" teacher_idx (networks/projections.py) --
# single source of truth for this pairing lives in the STAGE_MAP comment; kept in sync here.
LAYER_TO_TEACHER_STAGE = {2: 0, 5: 1, 8: 2, 11: 3}


class VJEPACachedDataset(Dataset):
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".npz"))
        if not self.files:
            raise FileNotFoundError(f"no .npz files found under {cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        d = np.load(os.path.join(self.cache_dir, self.files[idx]))
        image = torch.from_numpy(d["image"].astype(np.float32)).unsqueeze(0)  # (1,128,128,128)

        teacher_feats = {}
        for layer, stage_idx in LAYER_TO_TEACHER_STAGE.items():
            feat = d[f"feat_layer{layer}"].astype(np.float32)  # saved as (T,H,W,D)
            feat = np.transpose(feat, (1, 2, 0, 3))  # -> (H,W,T,D) == (X,Y,Z,D), see module docstring
            feat = torch.from_numpy(np.ascontiguousarray(feat)).permute(3, 0, 1, 2)  # -> (D,X,Y,Z)
            teacher_feats[f"dec_stage_{stage_idx}"] = feat

        return {"image": image, "teacher_feats": teacher_feats}
