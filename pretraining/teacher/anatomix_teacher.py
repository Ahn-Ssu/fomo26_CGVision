"""Anatomix teacher wrapper.

Source verified directly (not guessed):
  - Architecture: /root/anatomix/anatomix/model/network.py:195-522 (Unet class)
    Constructor args match README usage exactly:
    dimension=3, input_nc=1, output_nc=16, num_downs=4, ngf=16.
  - Weights: /root/anatomix/model-weights/anatomix.pth (or anatomix+brains.pth),
    strict=True load verified: <All keys matched successfully>, 5,899,344 params.
  - No forward hooks needed: Unet.forward(x, layers=[...], encode_only=False)
    is a NATIVE multi-feature API (network.py:447-521) that returns
    (final_output, [feat_at_each_requested_layer_id]).
  - model.encoder_idx = [8, 15, 22, 29]  (skip-connect source layers, post-ReLU
    pre-pool, one per encoder stage)
  - model.decoder_idx = [37, 44, 51, 58]  (Upsample layer ids, one per decoder
    stage -- these are the START of each decoder stage, not useful as hook
    targets directly)
  - Empirically verified decoder STAGE-END layer ids (last conv-block output
    before the next Upsample / the final conv), forward-pass confirmed on a
    128^3 input on GPU:
        layer 43: (1,128,16,16,16)  spatial_scale=0.125
        layer 50: (1, 64,32,32,32)  spatial_scale=0.25
        layer 57: (1, 32,64,64,64)  spatial_scale=0.5
        layer 64: (1, 16,128,128,128) spatial_scale=1.0
    peak GPU mem for this forward: ~1.6 GB. Forward time: ~0.15s (128^3, RTX 6000 Ada).

Normalization: pretraining/data/data_utils.py:4-46 `normalize_img(array,
percentile=99.99, zero_centered=True)` computes min/percentile over the FULL
volume (no foreground mask) and does (x-min)/(percentile-min), optionally
rescaled to [-1,1]. This is a percentile-based, monotonic, full-volume-scope
transform. Because it is invariant under any prior positive-scale affine
transform (algebraically: normalize_img(a*raw+b) == normalize_img(raw) for
a>0), applying it on top of asparagus's already foreground-z-normed volume
produces EXACTLY the same result (verified analytically, matching the
PRE_ANALYSIS.md Sec 3.2 empirical proof of ~1e-9 float-level agreement).
=> norm_type = "percentile" => Path (b) applies with no caveats.
"""

import importlib.util
import os
from typing import Dict, Optional

import torch

from .base import BaseTeacher

_NETWORK_PY = "/root/anatomix/anatomix/model/network.py"

# Layer ids whose OUTPUT is the last conv-block activation of each decoder
# stage (i.e. right before the next nn.Upsample, or before the final 1x1x1-ish
# conv for the last stage). Derived from model.decoder_idx = [37,44,51,58]
# (each entry is an Upsample layer id) by taking (next_decoder_idx - 1), and
# for the last stage taking (len(model.model) - 2) i.e. just before the final
# conv (index 65).
_DECODER_STAGE_END_LAYERS = [43, 50, 57, 64]
_ENCODER_STAGE_END_LAYERS = [8, 15, 22, 29]  # == model.encoder_idx, verified


def _load_unet_class():
    spec = importlib.util.spec_from_file_location("anatomix_network", _NETWORK_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Unet


class AnatomixTeacher(BaseTeacher):
    """checkpoint_path: 'anatomix' or 'anatomix+brains' (selects weight file)."""

    def __init__(self, checkpoint_path: str = "anatomix", device: str = "cuda"):
        self._variant = checkpoint_path
        weight_file = {
            "anatomix": "/root/anatomix/model-weights/anatomix.pth",
            "anatomix+brains": "/root/anatomix/model-weights/anatomix+brains.pth",
        }.get(checkpoint_path, checkpoint_path)
        if not os.path.isfile(weight_file):
            raise FileNotFoundError(f"Anatomix weight file not found: {weight_file}")
        super().__init__(checkpoint_path=weight_file, device=device)

    def _build_model(self) -> None:
        Unet = _load_unet_class()
        self.model = Unet(dimension=3, input_nc=1, output_nc=16, num_downs=4, ngf=16)
        sd = torch.load(self.checkpoint_path, map_location="cpu")
        result = self.model.load_state_dict(sd, strict=True)
        assert result is None or (len(result.missing_keys) == 0 and len(result.unexpected_keys) == 0), (
            f"Anatomix strict load failed: {result}"
        )
        self._n_params = sum(p.numel() for p in self.model.parameters())

    def _register_hooks(self) -> None:
        # No hooks needed: Unet.forward(x, layers=[...]) natively returns the
        # requested intermediate features. See extract_features() override below.
        pass

    def preprocess(self, x: torch.Tensor, meta: Optional[dict] = None) -> torch.Tensor:
        # x is assumed already asparagus foreground-z-normed (Path B).
        # normalize_img semantics, vectorized per-sample in the batch:
        #   (x - min) / (percentile(x, 99.99) - min), then *2-1 (zero_centered=True)
        b = x.shape[0]
        flat = x.view(b, -1)
        min_ = flat.min(dim=1, keepdim=True).values
        max_ = torch.quantile(flat.float(), 0.9999, dim=1, keepdim=True).to(flat.dtype)
        denom = (max_ - min_).clamp_min(1e-8)
        normed = (flat - min_) / denom
        normed = normed * 2 - 1
        return normed.view_as(x)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, meta: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        inp = self.preprocess(x, meta=meta)
        out, feats = self.model(inp, layers=_DECODER_STAGE_END_LAYERS, encode_only=False)
        names = [f"dec_stage_{i}" for i in range(len(_DECODER_STAGE_END_LAYERS))]
        result = dict(zip(names, feats))
        result["final"] = out
        self._features = result
        return dict(result)

    @property
    def feature_specs(self) -> Dict[str, dict]:
        specs = {}
        channels = [128, 64, 32, 16]
        scales = [0.125, 0.25, 0.5, 1.0]
        for i, (c, s) in enumerate(zip(channels, scales)):
            specs[f"dec_stage_{i}"] = {
                "channels": c,
                "spatial_scale": s,
                "module_path": f"model.model[{_DECODER_STAGE_END_LAYERS[i]}] (via native layers= API)",
            }
        specs["final"] = {"channels": 16, "spatial_scale": 1.0, "module_path": "model.model[65] (final conv output)"}
        return specs

    @property
    def norm_type(self) -> str:
        return "percentile"

    @property
    def input_requirements(self) -> dict:
        return {
            "modalities": "any",
            "spacing": "any",
            "patch_size": "any (fully-convolutional; verified at 128^3)",
            "skull_stripped": "any",
            "in_channels": 1,
            "is_3d": True,
        }
