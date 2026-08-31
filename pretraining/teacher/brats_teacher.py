"""BraTS teacher wrapper -- checkpoint delivered 2026-07-15 to
/root/teachers/BarTS_Teacher/ (nnUNet training-output directory). Verified
directly against real weights on GPU (RTX 6000 Ada), not guessed.

Source facts (all read directly from the delivered artifacts, not assumed):
  - Architecture: /root/teachers/BarTS_Teacher/plans.json,
    configurations["3d_fullres"]["architecture"] ->
    `dynamic_network_architectures.architectures.unet.PlainConvUNet`
    (NOT a residual-encoder UNet as PRE_ANALYSIS.md guessed -- corrected here).
    arch_kwargs: n_stages=6, features_per_stage=[32,64,128,256,320,320],
    conv_op=Conv3d, strides=[1,2,2,2,2,(2,2,1)], InstanceNorm3d, LeakyReLU,
    n_conv_per_stage=[2]*6, n_conv_per_stage_decoder=[2]*5.
  - dataset.json: channel_names={"0":"MRI"} (single pooled MRI channel across
    modalities), labels={"background":0,"whole_tumor":1} (binary),
    numTraining=5004, description: "BraTS2023 GLI -- 1ch pooled, binary
    labels. Preprocessing: p99.5 clip only; z-norm via nnUNet."
  - Checkpoint: /root/teachers/BarTS_Teacher/fold_all/checkpoint_best.pth
    (note: fold_all, not fold_0 as PRE_ANALYSIS.md assumed). Top-level dict
    has key `network_weights` (not `state_dict`) -- matches the format
    `asparagus`'s own `load_checkpoint_state_dict` already expects
    (asparagus/pipeline/auto_configuration/checkpoint.py, per
    /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/01_ASPARAGUS_ANALYSIS.md). `strict=True` load: <All keys matched
    successfully>, 30,785,994 params. current_epoch=998, _best_ema fg dice
    =0.9142737 (training curve info, not re-validated here).
  - named_children(): `encoder` (PlainConvEncoder, `.stages[0..5]`),
    `decoder` (UNetDecoder, `.stages[0..4]` + `.seg_layers[0..4]`).
    Decoder forward (dynamic_network_architectures/building_blocks/
    unet_decoder.py, read directly): for each stage s, computes feature
    `x = self.stages[s](...)`, appends `self.seg_layers[s](x)` to
    seg_outputs, THEN advances `lres_input = x` -- i.e. `decoder.stages[s]`
    is the actual multi-channel feature map, `decoder.seg_layers[s]` is a
    separate low-dim (num_classes) segmentation head applied on top. We hook
    `decoder.stages[*]`, not `seg_layers`, for feature-level distillation.
  - Feature shapes verified empirically at a 128^3 input (GPU forward,
    0.235s, peak 1.92GB -- comparable to VesselFM/Anatomix, not FastSurfer):
        decoder.stages[0] -> (1,320,8,8,8)     spatial_scale=0.0625
        decoder.stages[1] -> (1,256,16,16,16)  spatial_scale=0.125
        decoder.stages[2] -> (1,128,32,32,32)  spatial_scale=0.25
        decoder.stages[3] -> (1, 64,64,64,64)  spatial_scale=0.5
        decoder.stages[4] -> (1, 32,128,128,128) spatial_scale=1.0
    (deep_supervision=True also yields 5 low-res seg logit maps via
    seg_layers, exposed here as feature "seg_logits_stage_i" for anyone who
    wants an output-level signal too, but these are NOT the primary
    feature-distillation target.)

Normalization -- empirically verified, and DIFFERENT in kind from
Anatomix/VesselFM's invariance argument:
  dataset.json says preprocessing is "p99.5 clip only; z-norm via nnUNet" --
  i.e. clamp(foreground, q=0.995) -> z-score(foreground stats), computed
  PER-CASE (not a dataset-wide fingerprint rescale -- channel is named "MRI",
  and nnUNet's default per-channel normalization for non-CT modalities is
  per-case ZScoreNormalization). Critically, nnUNet's pipeline does NOT end
  with a final min-max rescale to [0,1] (unlike Anatomix's normalize_img and
  VesselFM's ScaleIntensityRangePercentiles, whose OWN final rescale step is
  exactly what makes them invariant to a prior affine transform of their
  input). So this wrapper's preprocess() does NOT simply pass through
  asparagus's already-normalized [0,1] input -- it RE-DERIVES the clamp+
  z-score fresh from whatever it is given (this recomputation is what
  restores the invariance property; see BaseTeacher docstring for the
  general principle). Verified on the real T1
  /root/data/FOMO-MRI/fomo-60k/sub_11043/ses_1/t1.nii.gz (asparagus
  volume_wise_znorm applied first, vs. raw): preprocessed-array max abs diff
  0.87, Pearson r=0.9995; through the real network, final-seg-logit cosine
  similarity=0.9994, **argmax label agreement=99.83%** (vs. Anatomix's exact
  algebraic identity, VesselFM's 1.0 cosine sim, and FastSurfer's 79.28% --
  BraTS sits between "exactly invariant" and FastSurfer's genuine mismatch,
  the small residual gap traced to asparagus's own upper clamp at q=0.99
  interacting slightly with this teacher's independent q=0.995 clamp; not
  investigated further). norm_type = "percentile" (practically, not exactly,
  invariant -- flagged explicitly, not overstated as exact).
"""

import json
import os
import pydoc
from typing import Dict, Optional

import numpy as np
import torch

from .base import BaseTeacher

_DEFAULT_DIR = "/root/teachers/BarTS_Teacher"
_DECODER_CHANNELS = [320, 256, 128, 64, 32]
_DECODER_SCALES = [0.0625, 0.125, 0.25, 0.5, 1.0]


def _build_plainconvunet(plans: dict, dataset: dict):
    cfg = plans["configurations"]["3d_fullres"]
    arch = cfg["architecture"]
    kwargs = dict(arch["arch_kwargs"])
    for key in arch["_kw_requires_import"]:
        if kwargs[key] is not None:
            kwargs[key] = pydoc.locate(kwargs[key])
    from dynamic_network_architectures.architectures.unet import PlainConvUNet

    return PlainConvUNet(
        input_channels=len(dataset["channel_names"]),
        num_classes=len(dataset["labels"]),
        deep_supervision=True,
        **kwargs,
    )


class BraTSTeacher(BaseTeacher):
    """checkpoint_path: nnUNet training-output directory (must contain
    plans.json, dataset.json, and fold_all/checkpoint_best.pth)."""

    def __init__(self, checkpoint_path: str = _DEFAULT_DIR, device: str = "cuda"):
        super().__init__(checkpoint_path=checkpoint_path, device=device)

    def _build_model(self) -> None:
        plans_path = os.path.join(self.checkpoint_path, "plans.json")
        dataset_path = os.path.join(self.checkpoint_path, "dataset.json")
        ckpt_path = os.path.join(self.checkpoint_path, "fold_all", "checkpoint_best.pth")
        for p in (plans_path, dataset_path, ckpt_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"BraTS teacher expects {p} to exist (nnUNet layout).")

        self._plans = json.load(open(plans_path))
        self._dataset = json.load(open(dataset_path))
        self.model = _build_plainconvunet(self._plans, self._dataset)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        result = self.model.load_state_dict(ckpt["network_weights"], strict=True)
        assert result is None or (len(result.missing_keys) == 0 and len(result.unexpected_keys) == 0), (
            f"BraTS strict load failed: {result}"
        )
        self._n_params = sum(p.numel() for p in self.model.parameters())
        self._best_ema_dice = ckpt.get("_best_ema")
        self._current_epoch = ckpt.get("current_epoch")

    def _register_hooks(self) -> None:
        self._raw_feats = {}

        def mk_hook(name):
            def hook(_module, _inp, out):
                self._raw_feats[name] = out.detach()

            return hook

        for i, stage in enumerate(self.model.decoder.stages):
            stage.register_forward_hook(mk_hook(f"dec_stage_{i}"))
        for i, seg_layer in enumerate(self.model.decoder.seg_layers):
            seg_layer.register_forward_hook(mk_hook(f"seg_logits_stage_{i}"))

    def preprocess(self, x: torch.Tensor, meta: Optional[dict] = None) -> torch.Tensor:
        # Re-derive nnUNet's own clamp(foreground, q=0.995) + z-score(foreground)
        # FRESH from whatever input we're given -- this recomputation (not a
        # pass-through) is what restores invariance to asparagus's prior
        # normalization; see module docstring.
        #
        # Mask source (Task 3, /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/03_PREPROCESSING_REPORT.md Issue A): if an
        # explicit mask is supplied via meta["mask"] (bool/uint8 tensor,
        # same shape as x, batch dim included) -- e.g. the mask saved by
        # data/preprocess_raw.py, computed at native resolution before any
        # interpolation and therefore NOT contaminated by cubic-resample
        # boundary leakage -- it is used directly. Otherwise falls back to
        # the self-derived `xi != xi.min()` heuristic (fine on its own once
        # Issue A's background re-cleaning has already run, but an explicit
        # mask is strictly more correct and is preferred whenever available).
        explicit_mask = None
        if meta is not None and "mask" in meta and meta["mask"] is not None:
            explicit_mask = meta["mask"].to(dtype=torch.bool, device=x.device)

        b = x.shape[0]
        out = torch.empty_like(x)
        for i in range(b):
            xi = x[i]
            if explicit_mask is not None:
                mask = explicit_mask[i]
            else:
                empty_val = xi.min()
                mask = xi != empty_val
            if mask.sum() == 0:
                out[i] = xi
                continue
            fg = xi[mask]
            q995 = torch.quantile(fg.float(), 0.995)
            clamped = torch.clamp(xi, max=q995)
            fg_clamped = clamped[mask]
            mean, std = fg_clamped.mean(), fg_clamped.std().clamp_min(1e-8)
            clamped_masked = torch.where(mask, clamped, torch.zeros_like(clamped))
            out[i] = torch.where(mask, (clamped_masked - mean) / std, torch.zeros_like(clamped))
        return out

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, meta: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        self._raw_feats.clear()
        inp = self.preprocess(x, meta=meta)
        seg_outputs = self.model(inp)  # list of 5 (deep_supervision=True), highest-res first
        result = dict(self._raw_feats)
        result["final"] = seg_outputs[0]
        self._features = result
        return dict(result)

    @property
    def feature_specs(self) -> Dict[str, dict]:
        specs = {}
        for i, (c, s) in enumerate(zip(_DECODER_CHANNELS, _DECODER_SCALES)):
            specs[f"dec_stage_{i}"] = {
                "channels": c,
                "spatial_scale": s,
                "module_path": f"model.decoder.stages[{i}]",
            }
            specs[f"seg_logits_stage_{i}"] = {
                "channels": 2,  # background + whole_tumor
                "spatial_scale": s,
                "module_path": f"model.decoder.seg_layers[{i}]",
            }
        specs["final"] = {"channels": 2, "spatial_scale": 1.0, "module_path": "model.decoder.seg_layers[4] (highest-res)"}
        return specs

    @property
    def norm_type(self) -> str:
        return "percentile"  # practically-invariant, ~99.8% argmax agreement -- see docstring, not exact

    @property
    def input_requirements(self) -> dict:
        return {
            "modalities": "any single MRI-like channel (trained on pooled FLAIR/T1/T1c/T2, 1-channel input)",
            "spacing": (1.0, 1.0, 1.0),
            "patch_size": (128, 160, 112),  # nnUNet plans; verified to also run at 128^3 (divisible by required 32/16)
            "skull_stripped": "unverified",
            "in_channels": 1,
            "is_3d": True,
        }
