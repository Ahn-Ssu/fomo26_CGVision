"""VesselFM teacher wrapper.

Source verified directly (not guessed) against a real clone of
https://github.com/bwittmann/vesselFM (commit checked out at
/root/teachers/repos/vesselFM) and the real checkpoint downloaded from
https://huggingface.co/bwittmann/vesselFM:

  - Architecture: vesselfm/seg/configs/model/dyn_unet_base.yaml instantiates
    `monai.networks.nets.DynUNet` (via hydra `_target_`) with:
        in_channels=1, out_channels=1, spatial_dims=3,
        strides=[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
        kernel_size=[[3,3,3]]*6,
        upsample_kernel_size=[[2,2,2]]*5,
        filters=[32,64,128,256,320,320], res_block=True
    This is the exact config referenced by vesselfm/seg/configs/inference.yaml
    (`defaults: - model: dyn_unet_base`).

  - Weights: HuggingFace repo `bwittmann/vesselFM`, file `vesselFM_base.pt`
    (~125.7 MB), downloaded to
    /root/teachers/checkpoints/vesselfm/vesselFM_base.pt via
    `huggingface_hub.hf_hub_download`. This is the ONLY checkpoint published
    in that repo (verified via `HfApi().list_repo_files`) and is the one
    vesselfm/seg/inference.py itself falls back to downloading
    (`hf_hub_download(repo_id='bwittmann/vesselFM', filename='vesselFM_base.pt')`)
    when no local `cfg.ckpt_path` is given -- i.e. it is the generalist,
    zero-shot checkpoint (trained jointly on D_real + D_drand + D_flow per the
    HF README), not a dataset-specific finetune. No other checkpoint variants
    exist to choose between.

  - `model.load_state_dict(ckpt, strict=True)` succeeds: "<All keys matched
    successfully>". model.parameters() total = 31,418,977 (unique tensors).
    Note: the raw checkpoint dict has 180 keys summing to 62,837,921 elements
    -- roughly (not exactly) double the unique parameter count. This is NOT a
    mismatch: MONAI's DynUNet builds a recursive `skip_layers` structure
    (`DynUNetSkipLayer`) whose `.downsample` submodules are the SAME Python
    objects as `model.downsamples[i]` (assigned by reference, not copied), so
    each of those tensors is addressable under two different state_dict key
    paths (e.g. both `downsamples.0.conv1.conv.weight` and
    `skip_layers.next_layer.downsample.conv1.conv.weight`). `model.parameters()`
    deduplicates by object identity (31,418,977, matches architecture); raw
    `state_dict()`/checkpoint key enumeration does not (62,837,921).

  - No native multi-feature API: DynUNet.forward() returns only the final
    logits (vesselfm/seg/module.py: `pred_mask = self.model(image)`; monai
    dynunet.py forward(): `out = self.skip_layers(x); out = self.output_block(out); return out`).
    Forward hooks are therefore required (see _register_hooks below).

  - Decoder stages, verified empirically with a 128^3 input on GPU (see
    feature_specs / smoke test):
        model.upsamples[0] -> (1,320, 8, 8, 8)   spatial_scale=0.0625
        model.upsamples[1] -> (1,256,16,16,16)   spatial_scale=0.125
        model.upsamples[2] -> (1,128,32,32,32)   spatial_scale=0.25
        model.upsamples[3] -> (1, 64,64,64,64)   spatial_scale=0.5
        model.upsamples[4] -> (1, 32,128,128,128) spatial_scale=1.0
        model.output_block -> (1,  1,128,128,128) spatial_scale=1.0 (final logits)
    peak GPU mem for this forward (fresh process, RTX 6000 Ada, cuda:0):
    2.584 GB. Forward time (warm, 128^3): ~0.03s.

Normalization: vesselfm/seg/configs/inference.yaml `transforms_config`
(the literal config consumed by vesselfm/seg/inference.py ->
vesselfm/seg/utils/data.py::generate_transforms(), i.e. actual inference-time
preprocessing, NOT training augmentation) is exactly:
    - EnsureChannelFirst: {channel_dim: "no_channel"}
    - ScaleIntensityRangePercentiles: {lower: 1, upper: 99, b_min: 0, b_max: 1, clip: true}
    - ToTensor: {device: null}
`monai.transforms.ScaleIntensityRangePercentiles._normalize` (monai 1.3.2,
monai/transforms/intensity/array.py) computes
    a_min = percentile(img, lower); a_max = percentile(img, upper)
over the ENTIRE `img` array as given (`monai.transforms.utils_pytorch_numpy_unification.percentile`
with `dim=None`, i.e. flattened, no foreground mask, no channel_wise since
`channel_wise` defaults to False and is not set in the config) -- this is a
FULL-VOLUME percentile, NOT foreground-only. It then applies
`ScaleIntensityRange`: `img = clip((img - a_min) / (a_max - a_min), b_min, b_max)`.
This matches PRE_ANALYSIS.md's assumed shape (percentile 1-99 -> [0,1]
min-max) for the percentile bounds and output range, but the assumption of
foreground-only masking, if PRE_ANALYSIS.md made one, is NOT what the code
does -- confirmed by reading `_normalize()` directly, no foreground mask is
ever applied.

Because both this transform (percentile-clip + linear rescale) and
asparagus's `volume_wise_znorm` (clamp @ fg 99th pct -> fg z-score -> global
rescale-to-[0,1]) are monotonic non-decreasing functions of the raw
intensity, composing them (Path B: vesselfm-normalize(asparagus-normalize(raw)))
is empirically indistinguishable from Path A (vesselfm-normalize(raw)) up to
float precision. Verified on a real T1 volume
(/root/data/FOMO-MRI/fomo-60k/sub_11043/ses_1/t1.nii.gz, 240x240x155):
    preprocessed-volume max abs diff (Path A vs B, full volume): 1.79e-07
    preprocessed-volume mean abs diff:                           4.67e-09
    Pearson correlation (200k-voxel subsample):                  0.99999999999999...
    Spearman rank correlation (200k-voxel subsample):             0.9977
    cosine similarity (full flattened volumes):                   1.0
    model final-logit max abs diff (128^3 center patch, both paths through
      the real network):                                          0.00988
    model final-logit cosine similarity:                           1.0
    decoder feature cosine similarity (all 5 stages):              1.0
=> norm_type = "percentile" => Path (b) applies safely (empirically confirmed,
not just assumed): asparagus-normalized input can be fed through
VesselFMTeacher.preprocess() and gives (numerically) the same result as
feeding raw intensities through it.

preprocess() below reimplements ScaleIntensityRangePercentiles(1, 99, 0, 1,
clip=True) with `torch.quantile` per-sample (batched), matching MONAI's
`_normalize()`/`ScaleIntensityRange.__call__()` formula exactly:
    a_min, a_max = quantile(x, 0.01), quantile(x, 0.99)   # per-sample, flattened
    out = clip((x - a_min) / (a_max - a_min), 0, 1)
(MONAI's `percentile()` helper falls back to `np.percentile` instead of
`torch.quantile` for tensors with >1e6 elements purely as a torch-CPU
workaround for an old pytorch bug (torch#64947); both give the same linear-
interpolated percentile value, so using `torch.quantile` uniformly here is
numerically equivalent and keeps the whole op on-GPU and batched, following
the same convention as AnatomixTeacher.preprocess().)
"""

import os
from typing import Dict, Optional

import torch

from .base import BaseTeacher

_DEFAULT_CKPT = "/root/teachers/checkpoints/vesselfm/vesselFM_base.pt"

# filters[i] is the output channel count at each of DynUNet's 6 resolution
# levels (0=full res stem ... 5=bottleneck), per dyn_unet_base.yaml.
_FILTERS = [32, 64, 128, 256, 320, 320]

# Decoder stage i = model.upsamples[i], output channel count and spatial
# scale (relative to full input resolution) -- both empirically verified
# with a 128^3 forward pass (see module docstring and verify script).
_DEC_CHANNELS = [320, 256, 128, 64, 32]
_DEC_SCALES = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]


class VesselFMTeacher(BaseTeacher):
    """checkpoint_path: path to vesselFM_base.pt (defaults to the standard
    location this repo downloads it to)."""

    def __init__(self, checkpoint_path: str = _DEFAULT_CKPT, device: str = "cuda"):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"VesselFM weight file not found: {checkpoint_path}")
        super().__init__(checkpoint_path=checkpoint_path, device=device)

    def _build_model(self) -> None:
        from monai.networks.nets import DynUNet

        self.model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            kernel_size=[[3, 3, 3]] * 6,
            strides=[[1, 1, 1]] + [[2, 2, 2]] * 5,
            upsample_kernel_size=[[2, 2, 2]] * 5,
            filters=_FILTERS,
            res_block=True,
        )
        sd = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        result = self.model.load_state_dict(sd, strict=True)
        assert result is None or (len(result.missing_keys) == 0 and len(result.unexpected_keys) == 0), (
            f"VesselFM strict load failed: {result}"
        )
        self._n_params = sum(p.numel() for p in self.model.parameters())

    def _register_hooks(self) -> None:
        def make_hook(name):
            def hook(module, inp, out):
                self._features[name] = out
            return hook

        for i, up_block in enumerate(self.model.upsamples):
            handle = up_block.register_forward_hook(make_hook(f"dec_stage_{i}"))
            self._hook_handles.append(handle)
        handle = self.model.output_block.register_forward_hook(make_hook("final"))
        self._hook_handles.append(handle)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def preprocess(self, x: torch.Tensor, meta: Optional[dict] = None) -> torch.Tensor:
        # x is assumed already asparagus foreground-z-normed (Path B) -- see
        # module docstring for the empirical proof this is safe for VesselFM's
        # percentile-based normalization.
        # Reimplements ScaleIntensityRangePercentiles(lower=1, upper=99,
        # b_min=0, b_max=1, clip=True), per-sample over the batch.
        b = x.shape[0]
        flat = x.reshape(b, -1).float()
        a_min = torch.quantile(flat, 0.01, dim=1, keepdim=True)
        a_max = torch.quantile(flat, 0.99, dim=1, keepdim=True)
        denom = (a_max - a_min).clamp_min(1e-8)
        out = (flat - a_min) / denom
        out = out.clamp(0.0, 1.0)
        return out.view_as(x).to(x.dtype)

    @property
    def feature_specs(self) -> Dict[str, dict]:
        specs = {}
        for i, (c, s) in enumerate(zip(_DEC_CHANNELS, _DEC_SCALES)):
            specs[f"dec_stage_{i}"] = {
                "channels": c,
                "spatial_scale": s,
                "module_path": f"model.upsamples[{i}]",
            }
        specs["final"] = {"channels": 1, "spatial_scale": 1.0, "module_path": "model.output_block"}
        return specs

    @property
    def norm_type(self) -> str:
        return "percentile"

    @property
    def input_requirements(self) -> dict:
        return {
            "modalities": "any",
            "spacing": "any",
            "patch_size": (128, 128, 128),
            "skull_stripped": "any",
            "in_channels": 1,
            "is_3d": True,
        }
