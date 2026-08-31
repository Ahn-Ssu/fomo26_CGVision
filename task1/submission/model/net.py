"""Standalone (no asparagus/lightning/hydra dependency) port of the champion
architecture -- modality_specific encoder + multistage feature source + GeM
pooling -- for FOMO26 Task 1 (CLS002_FOMO26_Infarct) submission containers.

Ported from /root/asparagus/asparagus/modules/networks/task1_arch18.py
(FomoTask1Arch18Net, Pool, StageConcatHead -- verbatim logic, only the
constructor is simplified) + gem_pooling.py (verbatim, copied alongside as
gem_pooling.py) + networks/student.py (verbatim, copied alongside as
student.py, byte-identical to /root/FOMO26/networks/student.py).

Why the constructor doesn't take checkpoint_path like the original: the
original loads the ver5 PRETRAIN checkpoint at construction time purely to
(a) infer the teacher roster / convpass_encoder / skip_alpha flags via
_infer_student_config, and (b) partially transplant its weights as an
initial value -- both irrelevant here, since a submission container always
immediately does a STRICT load_state_dict of a fully fine-tuned fold
checkpoint right after construction, which overwrites every parameter
regardless of its initial value. Skipping that step saves bundling the
382MB ver5 checkpoint in the container for zero functional difference.

TEACHERS/CONVPASS_ENCODER/SKIP_ALPHA below are hardcoded from ver5's own
checkpoint keys (verified 2026-08-17 via the same _infer_student_config
logic against /root/FOMO26/expr/pretraining/ver5/checkpoints/step_280000.pt)
rather than inferred at runtime -- if you ever point this at a different
pretrain checkpoint, the safety net is that a wrong architecture here will
make the fold checkpoint's strict load_state_dict fail loudly (missing/
unexpected keys), not silently produce wrong predictions.
"""
from typing import List

import torch
import torch.nn as nn

from gem_pooling import SignedGeM3d
from student import StudentResEncUNet

TEACHERS = ["anatomix+brains", "brats", "vesselfm", "vjepa", "voco"]
CONVPASS_ENCODER = True
SKIP_ALPHA = False
MULTISTAGE_ENC_IDXS = [1, 2, 3, 4, 5]


class Pool(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.gem = SignedGeM3d() if mode == "gem" else None

    def forward(self, x):
        if self.mode == "max":
            return torch.nn.functional.adaptive_max_pool3d(x, 1).flatten(1)
        if self.mode == "avg":
            return torch.nn.functional.adaptive_avg_pool3d(x, 1).flatten(1)
        return self.gem(x).flatten(1)


class StageConcatHead(nn.Module):
    def __init__(self, channels: List[int], classes: int, pooling: str, dropout: float):
        super().__init__()
        self.pools = nn.ModuleList(Pool(pooling) for _ in channels)
        self.stage_norms = nn.ModuleList(nn.LayerNorm(c) for c in channels)
        self.norm = nn.LayerNorm(sum(channels))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(sum(channels), classes)

    def forward(self, feats):
        z = [norm(pool(x)) for pool, norm, x in zip(self.pools, self.stage_norms, feats)]
        return self.fc(self.drop(self.norm(torch.cat(z, 1))))


class ChampionTask1Net(nn.Module):
    """modality_specific + multistage + gem only -- the champion config.
    Not the general 18-way wrapper; other arch18 combos are out of scope
    for a submission container."""

    def __init__(self, input_channels: int = 4, output_channels: int = 2,
                 teacher_name: str = "brats", dropout_rate: float = 0.3):
        super().__init__()
        self.teacher_name = teacher_name
        self.n_modalities = input_channels
        self.backbones = nn.ModuleList(
            StudentResEncUNet(in_channels=1, teachers=TEACHERS, convpass=True,
                               convpass_encoder=CONVPASS_ENCODER, skip_alpha=SKIP_ALPHA,
                               norm_conditional=False)
            for _ in range(input_channels)
        )
        # matches the original FomoTask1Arch18Net's `self.backbone =
        # self.backbones[0]` alias exactly -- both attribute names register
        # the SAME module object, so state_dict() emits the tensors under
        # both "backbone.*" and "backbones.0.*" prefixes. The fold
        # checkpoints were saved from that original class, so this alias is
        # required for a strict load_state_dict to match key-for-key
        # (confirmed empirically 2026-08-17: dropping it left 329
        # "unexpected key" errors, all the backbone.* duplicates of
        # backbones.0.*).
        self.backbone = self.backbones[0]
        base_ch = self.backbones[0].encoder.out_channels
        self.stage_indices = MULTISTAGE_ENC_IDXS
        self.head = StageConcatHead(
            [base_ch[i] * input_channels for i in self.stage_indices],
            output_channels, "gem", dropout_rate,
        )

    def forward(self, x):
        per = [self.backbones[m].forward_encoder_only(x[:, m:m + 1], teacher_name=self.teacher_name)
               for m in range(self.n_modalities)]
        feats = [torch.cat([d[f"enc_stage_{i}"] for d in per], 1) for i in self.stage_indices]
        return self.head(feats)
