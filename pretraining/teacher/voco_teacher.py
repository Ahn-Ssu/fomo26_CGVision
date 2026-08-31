"""VoCo-OpenMind teacher wrapper.

Source verified directly (not guessed) -- see /root/external_teacher_probe/NOTES.md Sec 1 for the
full investigation this is built on:
  - Checkpoint: `MIC-DKFZ/ResEncL-OpenMind-VoCo` on HuggingFace (the task-given name
    `AnonRes/ResEncL-OpenMind-VoCo` 307-redirects there). Self-supervised VoCo (Volume Contrastive,
    Wu et al. CVPR 2024) pretraining on `Dataset745_OpenNeuro_v2` (OpenMind benchmark, real brain
    MRI -- per-subject `modality: "T1w"/"inplaneT2"` entries confirmed in the checkpoint's own
    `init_args.pretrain_json`, NOT synthetic, NOT CT).
  - Architecture: `dynamic_network_architectures.building_blocks.residual_encoders.ResidualEncoder`,
    6 stages, features_per_stage=[32,64,128,256,320,320], strides=[1,2,2,2,2,2] -- IDENTICAL plan to
    our own student encoder (networks/student.py FEATURES_PER_STAGE) and to BraTS/VesselFM. Loaded
    with strict=True: 0 missing, 0 unexpected keys (empirically confirmed, not assumed).
  - checkpoint is ENCODER-ONLY (no decoder keys in the state dict at all) -- this teacher therefore
    only ever returns enc_stage_0..4 (stage 5, stride 32, is deliberately dropped: rank-deficient in
    the PCA probe even pooled across 30 volumes, AND has no encoder-side student counterpart worth
    the bottleneck's sparse voxel count -- see external_teacher_probe/REPORT.md).

Normalization (critical, verified in adaptation_plan.json / init_args.plan, NOT guessed):
`ZScoreNormalization`, `use_mask_for_norm: False` -- mean/std computed over the WHOLE crop
(background included), NOT foreground-masked like asparagus's own znorm. Composing asparagus's
clamp-then-masked-znorm with VoCo's own znorm on top would NOT reproduce VoCo's actual training-time
input distribution (the clamp step isn't affine, so the two don't cancel the way e.g. Anatomix's
percentile normalization does -- see anatomix_teacher.py's norm_type note for the contrast). So
norm_type here is "absolute": preprocess() requires the RAW (pre-asparagus) crop via
`meta={"raw": ...}` (see data/fomo300k_dataset.py's `raw` field, added for exactly this) rather than
using the `x` argument at all.
"""

import importlib.util
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BaseTeacher

_DEFAULT_CKPT = "/root/teachers/checkpoints/voco/checkpoint_final.pth"

FEATURES_PER_STAGE = [32, 64, 128, 256, 320, 320]
STRIDES = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
N_BLOCKS_PER_STAGE = [1, 3, 4, 6, 6, 6]
N_USABLE_STAGES = 5  # stage 5 (stride 32) dropped -- see module docstring


def _load_residual_encoder_class():
    # Imported via the SAME package used to verify/reconstruct this architecture in
    # external_teacher_probe/probe_external.py -- already a dependency of this venv (brats_teacher.py
    # also uses dynamic_network_architectures), no new install needed.
    from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
    return ResidualEncoder


class VoCoTeacher(BaseTeacher):
    def __init__(self, checkpoint_path: str = _DEFAULT_CKPT, device: str = "cuda"):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"VoCo checkpoint not found: {checkpoint_path}")
        super().__init__(checkpoint_path=checkpoint_path, device=device)

    def _build_model(self) -> None:
        ResidualEncoder = _load_residual_encoder_class()
        self.model = ResidualEncoder(
            input_channels=1, n_stages=6, features_per_stage=FEATURES_PER_STAGE,
            conv_op=nn.Conv3d, kernel_sizes=3, strides=STRIDES, n_blocks_per_stage=N_BLOCKS_PER_STAGE,
            conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={"affine": True, "eps": 1e-5},
            nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True}, return_skips=True,
        )
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt["network_weights"]
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        missing, unexpected = self.model.load_state_dict(enc_sd, strict=True)
        assert not missing and not unexpected, f"VoCo strict load failed: missing={missing} unexpected={unexpected}"
        self._n_params = sum(p.numel() for p in self.model.parameters())

    def _register_hooks(self) -> None:
        # No hooks needed: ResidualEncoder(return_skips=True).forward(x) natively returns the list
        # of per-stage skip tensors -- see extract_features() override below.
        pass

    def preprocess(self, x: torch.Tensor, meta: Optional[dict] = None) -> torch.Tensor:
        if meta is None or meta.get("raw") is None:
            raise ValueError("VoCoTeacher.preprocess requires meta={'raw': <pre-asparagus-znorm crop>} "
                              "-- see data/fomo300k_dataset.py's 'raw' field / module docstring for why "
                              "the normal (asparagus-normed) `x` can't be reused here.")
        raw = meta["raw"].to(x.device, dtype=x.dtype)
        b = raw.shape[0]
        flat = raw.view(b, -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True).clamp_min(1e-8)
        normed = (flat - mean) / std
        return normed.view_as(raw)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, meta: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        inp = self.preprocess(x, meta=meta)
        skips = self.model(inp)  # list of 6, stage 0 (stride1) .. stage 5 (stride32)
        result = {f"dec_stage_{i}": skips[i] for i in range(N_USABLE_STAGES)}
        self._features = result
        return dict(result)

    @property
    def feature_specs(self) -> Dict[str, dict]:
        scales = [1.0, 0.5, 0.25, 0.125, 0.0625]
        specs = {}
        for i in range(N_USABLE_STAGES):
            specs[f"dec_stage_{i}"] = {
                "channels": FEATURES_PER_STAGE[i],
                "spatial_scale": scales[i],
                "module_path": f"model.stages[{i}] (ResidualEncoder, return_skips=True)",
            }
        return specs

    @property
    def norm_type(self) -> str:
        return "absolute"

    @property
    def input_requirements(self) -> dict:
        return {
            "modalities": "any (brain MRI)",
            "spacing": "1mm iso (pretraining plan)",
            "patch_size": "any (fully-convolutional; verified at 128^3)",
            "skull_stripped": "any",
            "in_channels": 1,
            "is_3d": True,
        }
