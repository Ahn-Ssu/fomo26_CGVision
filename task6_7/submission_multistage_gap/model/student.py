"""Student network: a residual-encoder 3D UNet (ResEnc UNet).

Design rationale (see /root/00_PRE_ANALYSIS.md Sec 1.1 and
/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/01_ASPARAGUS_ANALYSIS.md): ResEnc UNet was chosen over Primus (3D ViT)
because all of our frozen teachers are convolutional 3D UNets, so stage-wise
feature matching is architecturally natural. This module deliberately mirrors
the exact nnU-Net-standard 6-stage plan our two nnU-Net-family teachers
(VesselFM's MONAI DynUNet, BraTS's dynamic_network_architectures PlainConvUNet)
were trained with -- features_per_stage=[32,64,128,256,320,320],
strides=[1,2,2,2,2,2] -- so that student decoder stages land on EXACTLY the
same spatial scales as those two teachers' decoder stages with no
interpolation needed (see /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/02_TEACHER_REPORT.md Sec E). Only a channel-count
projection (networks/projections.py) is needed per matched stage.

Named submodule convention (intentionally matches the teachers' own
`encoder.stages[i]` / `decoder.stages[i]` naming -- see brats_teacher.py /
vesselfm_teacher.py -- so forward-hook code and mental model transfer
directly): `student.encoder.stages[i]`, `student.decoder.stages[i]`.

This file is meant to be edited directly by whoever is iterating on the
student architecture -- it does not reach into gardening_tools/asparagus
internals, it is a from-scratch, fully-owned definition.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

FEATURES_PER_STAGE = [32, 64, 128, 256, 320, 320]
STRIDES = [1, 2, 2, 2, 2, 2]
N_CONV_PER_STAGE = 2

# Decoder stage output specs, deepest-first, matching what teachers expose
# (see /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/02_TEACHER_REPORT.md Sec C) -- for a 128^3 input.
DECODER_STAGE_CHANNELS = [320, 256, 128, 64, 32]
DECODER_STAGE_SCALES = [0.0625, 0.125, 0.25, 0.5, 1.0]


class TeacherConditionalInstanceNorm3d(nn.Module):
    """InstanceNorm3d whose affine (gamma, beta) is per-teacher instead of
    shared -- the normalization itself (per-instance mean/var) stays
    teacher-agnostic; only the post-norm affine transform is teacher-specific.
    ver2 (2026-07-20): all teachers' (weight, bias) pairs coexist as separate
    nn.Parameters at all times (same pattern as networks/projections.py's
    per-teacher heads) -- forward() just indexes into whichever teacher was
    episodically sampled this step, so only that teacher's affine params get
    gradient (relies on the existing DDP find_unused_parameters=True in
    run_pretrain.py). Initialized to gamma=1/beta=0 per teacher, matching
    nn.InstanceNorm3d(affine=True)'s own default init."""

    def __init__(self, num_features: int, teachers: List[str]):
        super().__init__()
        self.norm = nn.InstanceNorm3d(num_features, affine=False)
        self.weight = nn.ParameterDict({t: nn.Parameter(torch.ones(num_features)) for t in teachers})
        self.bias = nn.ParameterDict({t: nn.Parameter(torch.zeros(num_features)) for t in teachers})

    def forward(self, x: torch.Tensor, teacher_name: str) -> torch.Tensor:
        x = self.norm(x)
        w = self.weight[teacher_name].view(1, -1, 1, 1, 1)
        b = self.bias[teacher_name].view(1, -1, 1, 1, 1)
        return x * w + b


class Convpass3D(nn.Module):
    """Parallel bottleneck adapter (ver3 Sec B, 2026-07-24): down(1x1x1) ->
    depthwise(3x3x3) -> up(1x1x1), added to a DecoderStage's block output
    AFTER the block's own final activation (see DecoderStage.forward) --
    deliberately NOT inside ResidualBlock3D's internal residual sum (which
    is followed by LeakyReLU there), per 2026-07-24 decision: putting a
    zero-initialized-but-eventually-signed adapter output before a ReLU
    would clip it asymmetrically once trained. Living at the DecoderStage
    level as a pure parallel add-on keeps the correction bidirectional and
    keeps ResidualBlock3D itself completely unmodified.

    r fixed per stage as max(4, C_out // 16) (C_out = the block's own output
    channels, NOT the concatenated input channels -- confirmed 2026-07-24,
    gives a uniform ~32x compression ratio across all 5 decoder stages).
    alpha=16 (fixed, not learned) is the LoRA-style constant scale factor
    applied as alpha/r to the `up` output.

    Init (CHANGED from the original ver3 spec during Step 1 mock verification,
    2026-07-24 -- see NOTES.md "double-zero-init dead-gradient bug"): `up` is
    Kaiming-initialized like `down`/`dw`, NOT zero. The spec's original
    "up=0" was meant to guarantee a no-op at init, but combined with the
    per-teacher gate ALSO being 0-initialized (DecoderStage.convpass_gate),
    the two zeros multiply: `out = block_out + gate * up(...)`, and
    d(out)/d(gate) = up(...)'s current output = 0 identically whenever
    up.weight=0 (not just "small", exactly 0 by construction, verified via
    autograd: gate.grad, up/dw/down.weight.grad are all EXACTLY 0.0 every
    step, forever -- a genuine fixed point, not slow learning). The gate
    ALONE already guarantees the no-op-at-init property (out = block_out +
    0 * anything = block_out exactly, regardless of what `anything` is), so
    zeroing `up` on top of that was redundant for that goal and is what
    created the dead branch. With `up` Kaiming-initialized instead, the
    no-op-at-init property still holds exactly (still gated by gate=0), but
    now d(out)/d(gate) = up(dw(down(x))) is a genuine (generically nonzero)
    tensor, so `gate` gets real gradient from step 1 and can move; once it
    does, down/dw/up (whose own gradients are gated multiplicatively by
    `gate`) start receiving nonzero gradient too on the following step --
    mirroring how standard LoRA's B-then-A staggered wake-up works."""

    ALPHA = 16

    def __init__(self, c_in: int, c_out: int, r: int, stride: int = 1):
        super().__init__()
        # `stride` (ver4, 2026-07-27): encoder stages downsample INSIDE the
        # block (conv1 has stride=s), so the block's input and output live at
        # DIFFERENT resolutions whenever stride!=1 -- unlike decoder stages,
        # where the block itself is always stride=1 (upsample already brings
        # x to the output resolution before concat). To keep Convpass(x)
        # shape-compatible with block_out when used on an encoder stage, its
        # `down` conv (1x1x1) takes the same stride as the block's own conv1
        # -- exactly mirroring how ResidualBlock3D's own `skip` conv (also
        # 1x1x1) already handles this. Decoder call sites are unaffected
        # (stride=1 default, unchanged).
        self.down = nn.Conv3d(c_in, r, kernel_size=1, stride=stride, bias=False)
        self.dw = nn.Conv3d(r, r, kernel_size=3, padding=1, groups=r, bias=False)
        self.up = nn.Conv3d(r, c_out, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.down.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.dw.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.up.weight, nonlinearity="relu")
        self.scale = self.ALPHA / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        h = self.dw(h)
        h = self.up(h)
        return h * self.scale


class ResidualBlock3D(nn.Module):
    """Two 3x3x3 convs (InstanceNorm + LeakyReLU) with a residual skip,
    matching the norm/activation choices of our nnU-Net-family teachers
    (InstanceNorm3d, LeakyReLU) so feature statistics are in a comparable
    regime.

    `teachers=None` (default) reproduces Phase 1 exactly -- a single shared
    `nn.InstanceNorm3d(affine=True)`, same submodule/state_dict keys as
    before (so ver1 checkpoints keep loading unchanged). Passing a teacher
    name list switches to `TeacherConditionalInstanceNorm3d` (ver2's
    teacher-specific bias/gamma); `forward()`'s `teacher_name` arg is only
    consulted in that mode."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 teachers: Optional[List[str]] = None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=True)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.conditional = teachers is not None
        if self.conditional:
            self.norm1 = TeacherConditionalInstanceNorm3d(out_channels, teachers)
            self.norm2 = TeacherConditionalInstanceNorm3d(out_channels, teachers)
        else:
            self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)

        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor, teacher_name: Optional[str] = None) -> torch.Tensor:
        identity = x if self.skip is None else self.skip(x)
        h = self.conv1(x)
        h = self.norm1(h, teacher_name) if self.conditional else self.norm1(h)
        h = self.act1(h)
        h = self.conv2(h)
        h = self.norm2(h, teacher_name) if self.conditional else self.norm2(h)
        out = h + identity
        return self.act2(out)


class EncoderStage(nn.Module):
    """ver4 (2026-07-27): mirrors DecoderStage's Convpass wrapping for the
    encoder side -- `out = block_out + gate_k * Convpass_k(x)`, where `x` is
    the stage's OWN input (pre-block), not the concatenated tensor decoder
    uses (encoder has no concat). No skip_alpha here -- ver4 drops
    skip-scaling entirely (2026-07-27 decision: Convpass-only, no Part A),
    so encoder and decoder Convpass insertion are now symmetric (both just
    `block_out + gate*Convpass(block_input)`). Convpass3D's `stride` is set
    to match the block's own stride so Convpass(x)'s output shape lines up
    with block_out (see Convpass3D docstring)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 norm_teachers: Optional[List[str]] = None,
                 convpass_teachers: Optional[List[str]] = None):
        super().__init__()
        self.block = ResidualBlock3D(in_channels, out_channels, stride=stride, teachers=norm_teachers)
        self.convpass_enabled = convpass_teachers is not None
        if self.convpass_enabled:
            r = max(4, out_channels // 16)
            self.convpass_gate = nn.ParameterDict({t: nn.Parameter(torch.tensor(0.0)) for t in convpass_teachers})
            self.convpass = nn.ModuleDict({t: Convpass3D(in_channels, out_channels, r, stride=stride)
                                            for t in convpass_teachers})

    def forward(self, x: torch.Tensor, teacher_name: Optional[str] = None) -> torch.Tensor:
        out = self.block(x, teacher_name)
        if self.convpass_enabled:
            out = out + self.convpass_gate[teacher_name] * self.convpass[teacher_name](x)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels: int, features_per_stage: List[int] = FEATURES_PER_STAGE,
                 strides: List[int] = STRIDES, norm_teachers: Optional[List[str]] = None,
                 convpass_teachers: Optional[List[str]] = None):
        super().__init__()
        # Bare ResidualBlock3D (no EncoderStage wrapper) whenever
        # convpass_teachers is None -- keeps state_dict keys IDENTICAL to
        # ver1/ver2/ver3 (`encoder.stages.N.conv1...`, not `...stages.N.block.
        # conv1...`) so existing checkpoints (incl. the live ver3 run, which
        # auto-resumes from checkpoints/last.pt on every crash) keep loading
        # unchanged. Only the new ver4 encoder-Convpass path uses EncoderStage.
        self.stages = nn.ModuleList()
        prev_c = in_channels
        for c, s in zip(features_per_stage, strides):
            if convpass_teachers is not None:
                self.stages.append(EncoderStage(prev_c, c, stride=s,
                                                 norm_teachers=norm_teachers,
                                                 convpass_teachers=convpass_teachers))
            else:
                self.stages.append(ResidualBlock3D(prev_c, c, stride=s, teachers=norm_teachers))
            prev_c = c
        self.out_channels = features_per_stage

    def forward(self, x: torch.Tensor, teacher_name: Optional[str] = None) -> List[torch.Tensor]:
        skips = []
        for stage in self.stages:
            x = stage(x, teacher_name)
            skips.append(x)
        return skips  # skips[-1] is the bottleneck


class DecoderStage(nn.Module):
    """ver3 (2026-07-24): when `convpass_teachers` is set, adds:
      A. per-teacher skip scaling: `skip <- alpha_l^(k) * skip` BEFORE concat
         (skip is the encoder-side tensor being concatenated -- confirmed
         2026-07-24 this is what alpha multiplies, not the upsampled x).
         alpha init=1.0. Controlled by `skip_alpha` (default True = ver3
         behavior); ver4 (2026-07-27) sets skip_alpha=False to drop this
         mechanism entirely (Convpass-only, no Part A -- see NOTES.md/A1-A2:
         dropping it also removes the encoder/decoder Convpass asymmetry,
         since encoder has no skip tensor to scale in the first place).
      B. Convpass3D parallel adapter on the concatenated (post-skip-scale,
         if enabled) tensor, gated by a per-teacher learned scalar
         `convpass_gate` (init=0.0) added to the block's output: `out =
         block_out + gate_k * Convpass_k(cat)`. With alpha=1 (its init, if
         enabled) and gate=0 (its init), this is an EXACT no-op -- output is
         bit-identical to ver2's DecoderStage with the same teachers --
         verified in examples/verify_ver3_mock.py.

    `skip_alpha=True` (default) preserves ver3's exact structure/state_dict
    keys unchanged -- only setting it False (ver4) removes the skip_alpha
    ParameterDict."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 norm_teachers: Optional[List[str]] = None,
                 convpass_teachers: Optional[List[str]] = None, skip_alpha: bool = True):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResidualBlock3D(out_channels + skip_channels, out_channels, stride=1, teachers=norm_teachers)

        self.convpass_enabled = convpass_teachers is not None
        self.skip_alpha_enabled = self.convpass_enabled and skip_alpha
        if self.convpass_enabled:
            c_in = out_channels + skip_channels
            r = max(4, out_channels // 16)
            if self.skip_alpha_enabled:
                self.skip_alpha = nn.ParameterDict({t: nn.Parameter(torch.tensor(1.0)) for t in convpass_teachers})
            self.convpass_gate = nn.ParameterDict({t: nn.Parameter(torch.tensor(0.0)) for t in convpass_teachers})
            self.convpass = nn.ModuleDict({t: Convpass3D(c_in, out_channels, r) for t in convpass_teachers})

    def forward(self, x: torch.Tensor, skip: torch.Tensor, teacher_name: Optional[str] = None) -> torch.Tensor:
        x = self.upsample(x)
        if self.skip_alpha_enabled:
            skip = self.skip_alpha[teacher_name] * skip
        cat = torch.cat([x, skip], dim=1)
        out = self.block(cat, teacher_name)
        if self.convpass_enabled:
            out = out + self.convpass_gate[teacher_name] * self.convpass[teacher_name](cat)
        return out


class Decoder(nn.Module):
    def __init__(self, features_per_stage: List[int] = FEATURES_PER_STAGE,
                 norm_teachers: Optional[List[str]] = None,
                 convpass_teachers: Optional[List[str]] = None, skip_alpha: bool = True):
        super().__init__()
        # features_per_stage: [32,64,128,256,320,320] (encoder, shallow->deep)
        # decoder goes deep->shallow, 5 stages (mirrors n_stages-1)
        rev = list(reversed(features_per_stage))  # [320,320,256,128,64,32]
        self.stages = nn.ModuleList()
        in_c = rev[0]
        for i in range(len(rev) - 1):
            skip_c = rev[i + 1]
            out_c = rev[i + 1]
            self.stages.append(DecoderStage(in_c, skip_c, out_c, norm_teachers=norm_teachers,
                                             convpass_teachers=convpass_teachers, skip_alpha=skip_alpha))
            in_c = out_c

    def forward(self, skips: List[torch.Tensor], teacher_name: Optional[str] = None) -> List[torch.Tensor]:
        """Returns decoder stage features, DEEPEST FIRST (matching
        BraTS/VesselFM teacher hook order: dec_stage_0 = deepest/lowest-res)."""
        x = skips[-1]
        stage_outputs = []
        for i, stage in enumerate(self.stages):
            skip = skips[-(i + 2)]
            x = stage(x, skip, teacher_name)
            stage_outputs.append(x)
        return stage_outputs  # [dec_stage_0 (deepest), ..., dec_stage_4 (full-res)]


class StudentResEncUNet(nn.Module):
    """forward_with_features() is the primary training-time entry point:
    returns the 5 decoder-stage feature maps (dict, dec_stage_0..4) used as
    the student side of feature-level distillation. No segmentation head is
    included here -- pretraining never needs one; asp_finetune_* (Task 1
    finding, see /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/01_ASPARAGUS_ANALYSIS.md Sec C.4) rebuilds a fresh task head
    from config and transplants only matching encoder/decoder weights, keyed
    by a "model." prefix.
    """

    def __init__(self, in_channels: int = 1,
                 features_per_stage: List[int] = FEATURES_PER_STAGE,
                 strides: List[int] = STRIDES,
                 teachers: Optional[List[str]] = None,
                 convpass: bool = False,
                 convpass_encoder: bool = False,
                 skip_alpha: bool = True,
                 norm_conditional: bool = True):
        super().__init__()
        # ver3 (2026-07-24): Convpass decoder-only, IN teacher-conditional,
        # skip_alpha on -- convpass_encoder=False, norm_conditional=True,
        # skip_alpha=True are all defaults, so this constructor call is
        # BIT-IDENTICAL to before whenever the new args are left at default
        # (preserves the live ver3 run's checkpoint compatibility).
        # ver4 (2026-07-27): Convpass-only, no IN affine, no skip_alpha,
        # encoder+decoder both -- pass teachers=[...], convpass=True,
        # convpass_encoder=True, skip_alpha=False, norm_conditional=False.
        norm_teachers = teachers if norm_conditional else None
        encoder_convpass_teachers = teachers if (convpass and convpass_encoder) else None
        decoder_convpass_teachers = teachers if convpass else None
        self.encoder = Encoder(in_channels, features_per_stage, strides,
                                norm_teachers=norm_teachers, convpass_teachers=encoder_convpass_teachers)
        self.decoder = Decoder(features_per_stage, norm_teachers=norm_teachers,
                                convpass_teachers=decoder_convpass_teachers, skip_alpha=skip_alpha)
        self.decoder_stage_channels = list(reversed(features_per_stage))[1:]  # [320,256,128,64,32]
        self.decoder_stage_scales = DECODER_STAGE_SCALES
        self.conditional = teachers is not None

    def forward_with_features(self, x: torch.Tensor, teacher_name: Optional[str] = None,
                               include_encoder: bool = False) -> Dict[str, torch.Tensor]:
        """include_encoder (added for V-JEPA's mixed enc+dec distillation targets -- 2026-08-02):
        also includes enc_stage_0..5 in the returned dict. Default False preserves the exact
        original return shape/keys for every existing caller (encoder skips are always computed
        regardless -- this flag only controls whether they're ALSO returned, so it's free when
        unused and never changes what gets computed). For a teacher whose targets are ALL
        encoder-side (VoCo), prefer forward_encoder_only() instead -- it skips the decoder's
        compute (and backward) entirely rather than just omitting it from the return dict."""
        if self.conditional and teacher_name is None:
            raise ValueError("teacher_name is required when the student was built with teacher-conditional norm")
        skips = self.encoder(x, teacher_name)
        dec_feats = self.decoder(skips, teacher_name)
        out = {f"dec_stage_{i}": f for i, f in enumerate(dec_feats)}
        if include_encoder:
            out.update({f"enc_stage_{i}": f for i, f in enumerate(skips)})
        return out

    def forward_encoder_only(self, x: torch.Tensor, teacher_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Added for VoCo's pure-encoder distillation (2026-08-02): VoCo has no decoder counterpart
        at all, so any student decoder compute on a VoCo-sampled step is pure waste -- this skips
        self.decoder(...) entirely (no forward, no backward, no decoder-Convpass gradient), unlike
        forward_with_features(include_encoder=True) which still runs and backprops through the full
        decoder and simply doesn't return its output. Use this whenever a teacher's STAGE_MAP
        entries are ALL ("enc", idx) -- see networks/projections.py::is_encoder_only_teacher()."""
        if self.conditional and teacher_name is None:
            raise ValueError("teacher_name is required when the student was built with teacher-conditional norm")
        skips = self.encoder(x, teacher_name)
        return {f"enc_stage_{i}": f for i, f in enumerate(skips)}

    def forward(self, x: torch.Tensor, teacher_name: Optional[str] = None) -> torch.Tensor:
        # Convenience: return only the full-resolution feature (last decoder stage).
        feats = self.forward_with_features(x, teacher_name)
        return feats[f"dec_stage_{len(self.decoder.stages) - 1}"]

    def convpass_gate_parameters(self) -> List[nn.Parameter]:
        """ver3: the small set of scalar gate params (skip_alpha, convpass_gate)
        that get a 10x-backbone learning rate (ver3 spec Sec A/B) -- everything
        else (including the Convpass down/dw/up conv weights themselves) uses
        the normal backbone LR. Identified by name substring rather than a
        separate module list so this stays correct regardless of how deeply
        nested the DecoderStages are."""
        return [p for n, p in self.named_parameters() if ".skip_alpha." in n or ".convpass_gate." in n]

    def backbone_parameters(self) -> List[nn.Parameter]:
        """Everything NOT in convpass_gate_parameters() -- the normal-LR group."""
        gate_ids = {id(p) for p in self.convpass_gate_parameters()}
        return [p for p in self.parameters() if id(p) not in gate_ids]

    def convpass_path_parameters(self) -> List[nn.Parameter]:
        """The Convpass adapter path ONLY -- convpass_gate scalars + the
        down/dw/up conv weights -- used for the "Convpass 경로 gradient norm"
        health diagnostic (ver3 logging spec: "죽어있으면 lr/초기화 문제").
        Deliberately EXCLUDES skip_alpha (Sec A, a separate mechanism with
        its own always-nonzero gradient from init since alpha multiplies a
        real nonzero skip tensor) -- mixing it in would mask a genuinely
        dead Convpass branch behind skip_alpha's healthy gradient, which is
        exactly the 2026-07-24 double-zero-init bug this diagnostic exists
        to catch. skip_alpha's own values are tracked separately via
        alpha_history.csv / alpha_gate_snapshot()."""
        return [p for n, p in self.named_parameters()
                if ".convpass_gate." in n or ".convpass." in n]

    def alpha_gate_snapshot(self) -> Dict[str, Dict[str, float]]:
        """{"alpha": {"stages.0.brats": 1.0, ...}, "gate": {"dec_stage0/brats":
        ..., "enc_stage0/brats": ...}} -- current scalar values (not
        gradients) of every per-teacher skip_alpha/convpass_gate, for the
        ver3/ver4 alpha_history.csv/s_history.csv logging. `alpha` stays
        empty whenever skip_alpha is disabled (ver4). `gate` covers BOTH
        decoder and encoder stages (prefixed dec_/enc_) whenever
        convpass_encoder is on."""
        alpha, gate = {}, {}
        for i, stage in enumerate(self.decoder.stages):
            if not getattr(stage, "convpass_enabled", False):
                continue
            if getattr(stage, "skip_alpha_enabled", False):
                for t, p in stage.skip_alpha.items():
                    alpha[f"stage{i}/{t}"] = p.item()
            for t, p in stage.convpass_gate.items():
                gate[f"dec_stage{i}/{t}"] = p.item()
        for i, stage in enumerate(self.encoder.stages):
            if not getattr(stage, "convpass_enabled", False):
                continue
            for t, p in stage.convpass_gate.items():
                gate[f"enc_stage{i}/{t}"] = p.item()
        return {"alpha": alpha, "gate": gate}


def build_student(in_channels: int = 1, teachers: Optional[List[str]] = None,
                   convpass: bool = False, convpass_encoder: bool = False,
                   skip_alpha: bool = True, norm_conditional: bool = True) -> StudentResEncUNet:
    """teachers=None: Phase 1 shared-affine student (ver1). teachers=[...]:
    ver2 teacher-conditional InstanceNorm affine (gamma+beta) -- see
    TeacherConditionalInstanceNorm3d. convpass=True (requires teachers):
    ver3 defaults -- adds per-teacher skip scaling + parallel Convpass
    adapters on the decoder -- see DecoderStage.

    ver4 (2026-07-27, Convpass-only): convpass=True, convpass_encoder=True
    (also add Convpass to encoder stages -- see EncoderStage), skip_alpha=
    False (drop Part A entirely), norm_conditional=False (plain shared
    InstanceNorm affine, i.e. no ver2 teacher-conditional IN -- `teachers`
    is still required/non-None in this mode, since Convpass's per-teacher
    gate/adapter ModuleDicts still need teacher identity)."""
    return StudentResEncUNet(in_channels=in_channels, teachers=teachers, convpass=convpass,
                              convpass_encoder=convpass_encoder, skip_alpha=skip_alpha,
                              norm_conditional=norm_conditional)


if __name__ == "__main__":
    # Quick shape self-test (CPU, small patch for speed).
    model = build_student()
    x = torch.randn(1, 1, 64, 64, 64)
    feats = model.forward_with_features(x)
    for k, v in feats.items():
        print(k, tuple(v.shape))
