"""Task-1 18-way architecture screening wrapper."""
import sys
# /root/FOMO26 MUST be on sys.path before `from networks.student import ...`
# below -- fomo26_student.py does this same insert itself, but only as a side
# effect of being imported, which happens AFTER that line here. Standalone
# scripts that import /root/task1_head_arch first (e.g. verify_arch18.py) can
# mask this by adding /root/FOMO26 to sys.path themselves before ever
# importing this module -- Hydra's instantiate() has no such head start, so
# this bit unconditionally (bug found 2026-08-15 running the real pipeline,
# not caught by the isolated verify script for exactly that reason).
sys.path.insert(0, "/root/FOMO26")
sys.path.insert(0, "/root/task1_head_arch")
from typing import List
import torch
import torch.nn as nn
from gem_pooling import SignedGeM3d
from networks.student import StudentResEncUNet
from asparagus.modules.networks.fomo26_student import (
    FomoStudentBackboneMixin, _load_checkpoint_state_dict, _infer_student_config,
    _load_with_channel_repeat,
)

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

class FomoTask1Arch18Net(FomoStudentBackboneMixin, nn.Module):
    stem_weight_name = None
    MULTISTAGE_ENC_IDXS = [1, 2, 3, 4, 5]
    def __init__(self, input_channels, output_channels, checkpoint_path, teacher_name,
                 from_scratch=False, dropout_rate=.3, pooling_mode="avg",
                 feature_source="encoder_output", modality_arch="early_learnable", **_):
        super().__init__()
        assert pooling_mode in ("max", "gem", "avg")
        assert feature_source in ("encoder_output", "multistage")
        assert modality_arch in ("early_learnable", "shared_modalitywise", "modality_specific")
        self.teacher_name, self.n_modalities = teacher_name, input_channels
        self.num_classes = output_channels  # required by ClassificationModule/RegressionModule (clsreg_module.py:59)
        self.pooling_mode, self.feature_source, self.modality_arch = pooling_mode, feature_source, modality_arch
        raw = _load_checkpoint_state_dict(checkpoint_path)
        teachers, convpass_encoder, skip_alpha = _infer_student_config(raw)
        assert teacher_name in teachers
        self.backbones = nn.ModuleList()
        if modality_arch == "early_learnable":
            self.backbone = self._make_backbone(checkpoint_path, input_channels, teachers, convpass_encoder, skip_alpha, True)
            self.backbones.append(self.backbone)
        elif modality_arch == "shared_modalitywise":
            self.backbone = self._make_backbone(checkpoint_path, 1, teachers, convpass_encoder, skip_alpha, False)
            self.backbones.append(self.backbone)
        else:
            for _ in range(input_channels):
                self.backbones.append(self._make_independent(checkpoint_path, raw, teachers, convpass_encoder, skip_alpha))
            self.backbone = self.backbones[0]
        base_ch = self.backbone.encoder.out_channels
        idxs = [len(base_ch)-1] if feature_source == "encoder_output" else self.MULTISTAGE_ENC_IDXS
        factor = input_channels if modality_arch != "early_learnable" else 1
        self.stage_indices = idxs
        self.head = StageConcatHead([base_ch[i]*factor for i in idxs], output_channels, pooling_mode, dropout_rate)
        trainable = [n for n,p in self.named_parameters() if p.requires_grad]
        print(f"[Task1Arch18] pooling={pooling_mode} feature={feature_source} modality={modality_arch}; trainable={len(trainable)}")
        for n in trainable: print("  requires_grad=True:", n)
    def _make_backbone(self, path, channels, teachers, ce, sa, unfreeze):
        b = StudentResEncUNet(in_channels=channels, teachers=teachers, convpass=True, convpass_encoder=ce, skip_alpha=sa, norm_conditional=False)
        self.backbone = b
        self._load_and_configure(path, channels, self.teacher_name, False, unfreeze_stem=unfreeze)
        return b
    def _make_independent(self, path, raw, teachers, ce, sa):
        b = StudentResEncUNet(in_channels=1, teachers=teachers, convpass=True, convpass_encoder=ce, skip_alpha=sa, norm_conditional=False)
        adapted, missing, _, _ = _load_with_channel_repeat(b, raw, 1)
        assert len(adapted) > len(b.state_dict())//2 and not missing
        for p in b.parameters(): p.requires_grad_(False)
        enabled = 0
        for n,p in b.named_parameters():
            if f".convpass.{self.teacher_name}." in n or f".convpass_gate.{self.teacher_name}" in n or f".skip_alpha.{self.teacher_name}" in n:
                p.requires_grad_(True); enabled += 1
        assert enabled
        return b
    def forward(self, x):
        if self.modality_arch == "early_learnable":
            d = self.backbone.forward_encoder_only(x, teacher_name=self.teacher_name)
            return self.head([d[f"enc_stage_{i}"] for i in self.stage_indices])
        # shared_modalitywise has only ONE backbone in self.backbones (reused
        # for every modality) vs modality_specific's N independent ones -- bug
        # fix 2026-08-15: iterating `enumerate(self.backbones)` directly only
        # ever ran m=0 for shared_modalitywise (len(self.backbones)==1), so 3
        # of 4 modalities were silently dropped from the forward pass and the
        # concatenated feature width didn't match StageConcatHead's expected
        # `channels*input_channels` (built via the `factor` in __init__) --
        # this crashed at the first LayerNorm with a shape mismatch. Iterate
        # by modality index instead, always falling back to backbones[0] when
        # there's only one (shared) backbone to reuse.
        per = []
        for m in range(self.n_modalities):
            backbone = self.backbones[m] if len(self.backbones) > 1 else self.backbones[0]
            per.append(backbone.forward_encoder_only(x[:, m:m + 1], teacher_name=self.teacher_name))
        feats = [torch.cat([d[f"enc_stage_{i}"] for d in per], 1) for i in self.stage_indices]
        return self.head(feats)
