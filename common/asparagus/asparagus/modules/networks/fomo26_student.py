"""Asparagus model wrapper around FOMO26's own StudentResEncUNet (see
/root/FOMO26/networks/student.py), for finetuning ver3/ver4/ver5 pretrain
checkpoints with a "teacher-specific-parameter-only" scheme: the shared
backbone is frozen, only ONE teacher's Convpass adapter path (+ a newly
added task head) stays trainable.

Why this exists instead of using Asparagus's stock resenc_unet_b(_clsreg)
(the literal recipe in the FOMO26 competition doc): verified by direct
source comparison that gardening_tools.ResidualUNetEncoder (what
resenc_unet_b resolves to) does NOT share submodule names with our
StudentResEncUNet -- separate `self.stem` module (ours has none, stage 0
plays that role), `stages.<i>.blocks.<j>...` (plural + index; ours is
`stages.<i>.block...`, singular), and blocks are one level deeper
(`ConvDropoutNormNonlin`-wrapped). Asparagus's own checkpoint transplant
(`BaseModule.load_state_dict` -> `should_load_key`) requires an EXACT
dotted-name + shape match, so pointing the competition's literal
`+model=resenc_unet_b_clsreg checkpoint_path=our_ckpt.pt` at our checkpoint
would load ~0 pretrained tensors -- silently starting from random init and
defeating the point of finetuning a pretrained backbone. There is also no
Convpass concept at all in gardening_tools, so "teacher-specific param"
finetuning has no destination there.

Design: register OUR architecture as the Hydra model target instead
(`configs/model/fomo26_student_clsreg.yaml` etc.), so Asparagus's data
pipeline / Trainer loop / metrics / checkpoint saving all still run
unmodified -- only the model slot is swapped. Checkpoint loading is done
OURSELVES here at construction time (NOT via Asparagus's generic
`checkpoint_path=`/`weights=` mechanism -- leave those top-level Hydra
fields unset so `resolve_checkpoint()` returns None and
`BaseModule.__init__` skips its own transplant entirely, see
base_module.py line ~71). This sidesteps the exact-name-matching risk
completely: we know our own checkpoint's keys, so we load them onto our
own freshly-built `StudentResEncUNet` submodule directly.
"""

import re
import sys

sys.path.insert(0, "/root/FOMO26")
sys.path.insert(0, "/root/task1_head_arch")

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from gardening_tools.modules.networks.BaseNet import BaseNet
from gardening_tools.modules.networks.components.heads import ClsRegHead
from multistage_head import MultiStageGeMHead  # noqa: E402
from networks.student import StudentResEncUNet  # noqa: E402


def _load_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    # Strip the "model." prefix that run_pretrain.py's save_checkpoint() adds
    # (see run_pretrain.py module docstring, Sec on Asparagus compat) to get
    # back to StudentResEncUNet's own native key names.
    return {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}


_GATE_OR_ALPHA_RE = re.compile(r"\.(?:convpass_gate|skip_alpha)\.([^.]+)$")
_CONVPASS_MODULE_RE = re.compile(r"\.convpass\.([^.]+)\.")


def _infer_student_config(state_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], bool, bool]:
    """Reads (teacher roster, convpass_encoder, skip_alpha) back out of a
    checkpoint's own key names, so this wrapper works unmodified across
    ver3 (decoder-only convpass, skip_alpha on)/ver4/ver5 (encoder+decoder
    convpass, skip_alpha off) without hardcoding per-version flags."""
    teachers = set()
    convpass_encoder = False
    skip_alpha = False
    for k in state_dict:
        m = _GATE_OR_ALPHA_RE.search(k)
        if m:
            teachers.add(m.group(1))
        m = _CONVPASS_MODULE_RE.search(k)
        if m:
            teachers.add(m.group(1))
        if k.startswith("encoder.") and ("convpass_gate" in k or ".convpass." in k):
            convpass_encoder = True
        if "skip_alpha" in k:
            skip_alpha = True
    assert teachers, "no Convpass/skip_alpha keys found in checkpoint -- not a Convpass-enabled student checkpoint?"
    return sorted(teachers), convpass_encoder, skip_alpha


def _load_with_channel_repeat(module: nn.Module, state_dict: Dict[str, torch.Tensor], in_channels: int):
    """Like Asparagus's own `repeat_stem_weights` (base_module.py) but generic
    to any first-conv layer whose pretrained input-channel dim is 1 (our
    pretraining is single-modality) and the finetune task needs more (e.g.
    Task 1's 3-modality FLAIR+ADC+DWI stack): repeats+averages instead of
    dropping the tensor, so multi-modality tasks still get a real transplant
    for the stem AND any teacher's stage-0 Convpass `down` conv (both have a
    channel-1 second dim at construction).

    Returns `repeated_keys` (2026-08-11, added for the learnable-stem
    ablation) alongside the usual (adapted, missing, unexpected) -- the
    subset of `adapted` that took the repeat branch rather than an exact
    shape match, i.e. exactly the "stem" tensors whose channel count had to
    be invented for multi-modality input. `_load_and_configure`'s
    `unfreeze_stem` option unfreezes precisely this set."""
    own_sd = module.state_dict()
    adapted = {}
    repeated_keys = set()
    for k, v in state_dict.items():
        if k not in own_sd:
            continue
        target_shape = own_sd[k].shape
        if v.shape == target_shape:
            adapted[k] = v
        elif (
            v.ndim >= 2
            and v.shape[1] == 1
            and target_shape[1] == in_channels
            and v.shape[0] == target_shape[0]
            and v.shape[2:] == target_shape[2:]
        ):
            adapted[k] = v.repeat(1, in_channels, *([1] * (v.ndim - 2))) / in_channels
            repeated_keys.add(k)
    missing, unexpected = module.load_state_dict(adapted, strict=False)
    return adapted, missing, unexpected, repeated_keys


class FomoStudentBackboneMixin:
    """Shared checkpoint-loading/freeze logic for the cls/reg and seg
    wrappers below. Expects `self.backbone` (a StudentResEncUNet) to already
    be constructed before `_load_and_configure` is called."""

    def _load_and_configure(self, checkpoint_path: str, in_channels: int,
                             teacher_name: str, from_scratch: bool,
                             unfreeze_stem: bool = False) -> None:
        raw_sd = _load_checkpoint_state_dict(checkpoint_path)
        adapted, missing, unexpected, repeated_keys = _load_with_channel_repeat(self.backbone, raw_sd, in_channels)
        n_own = len(self.backbone.state_dict())
        print(f"[FomoStudent] checkpoint={checkpoint_path}")
        print(f"[FomoStudent] transplanted {len(adapted)}/{n_own} backbone tensors "
              f"(missing={len(missing)}, unexpected_in_ckpt={len(unexpected)}, "
              f"channel_repeated={len(repeated_keys)})")
        assert len(adapted) > n_own // 2, (
            "Fewer than half of the backbone's tensors were loaded from the checkpoint -- "
            "this almost certainly means a key-name or architecture mismatch, not a healthy "
            "partial transplant. Check _infer_student_config()'s guess against the checkpoint."
        )

        for p in self.backbone.parameters():
            p.requires_grad_(False)

        trainable_names = []
        for n, p in self.backbone.named_parameters():
            if (f".convpass.{teacher_name}." in n
                    or f".convpass_gate.{teacher_name}" in n
                    or f".skip_alpha.{teacher_name}" in n):
                p.requires_grad_(True)
                trainable_names.append(n)
        assert trainable_names, (
            f"No Convpass/skip_alpha parameters found for teacher_name={teacher_name!r} -- "
            f"check spelling against the checkpoint's own teacher roster."
        )

        stem_names = []
        if unfreeze_stem:
            # repeated_keys also contains OTHER (non-selected) teachers' stage-0 Convpass
            # `down` conv weights -- those needed channel-repeat too (c_in=in_channels for
            # every teacher's Convpass, not just the selected one) but must stay frozen,
            # matching the "only the selected teacher's Convpass is trainable" invariant.
            # Exclude anything Convpass-path (".convpass." in name) -- the real stem is just
            # the backbone's own stage-0 block conv1/skip.
            stem_keys = {k for k in repeated_keys if ".convpass." not in k}
            assert stem_keys, (
                "unfreeze_stem=True but no non-Convpass channel-repeated (stem) tensors were "
                "found -- if this model was built with in_channels=1 (e.g. the mixer variant, "
                "which keeps the pretrained single-channel stem byte-identical and never "
                "repeats it), there is no repeated stem to unfreeze; use the mixer's own "
                "learnable module instead."
            )
            for n, p in self.backbone.named_parameters():
                if n in stem_keys:
                    p.requires_grad_(True)
                    stem_names.append(n)

        print(f"[FomoStudent] teacher={teacher_name}: {len(trainable_names)} trainable Convpass tensors + "
              f"{len(stem_names)} trainable stem tensors (from_scratch={from_scratch}, "
              f"unfreeze_stem={unfreeze_stem})")

        if from_scratch:
            self._reinit_teacher_convpass(teacher_name)

    def _reinit_teacher_convpass(self, teacher_name: str) -> None:
        """Re-randomizes the chosen teacher's Convpass adapter (down/dw/up +
        gate, and skip_alpha if present) in place, matching Convpass3D's own
        __init__ scheme exactly (networks/student.py) -- used for the
        from-scratch-vs-pretrained-init ablation. Params stay trainable
        (requires_grad already set True by the caller)."""
        stages = list(self.backbone.encoder.stages) + list(self.backbone.decoder.stages)
        n_reinit = 0
        for stage in stages:
            if not getattr(stage, "convpass_enabled", False):
                continue
            if teacher_name in stage.convpass:
                cp = stage.convpass[teacher_name]
                nn.init.kaiming_normal_(cp.down.weight, nonlinearity="relu")
                nn.init.kaiming_normal_(cp.dw.weight, nonlinearity="relu")
                nn.init.kaiming_normal_(cp.up.weight, nonlinearity="relu")
                n_reinit += 1
            if teacher_name in stage.convpass_gate:
                with torch.no_grad():
                    stage.convpass_gate[teacher_name].fill_(0.0)
            if getattr(stage, "skip_alpha_enabled", False) and teacher_name in stage.skip_alpha:
                with torch.no_grad():
                    stage.skip_alpha[teacher_name].fill_(1.0)
        assert n_reinit, f"from_scratch=True but found no Convpass module for teacher_name={teacher_name!r}"
        print(f"[FomoStudent] from_scratch=True: re-initialized {n_reinit} Convpass adapters for {teacher_name!r}")


class FomoStudentClsRegNet(FomoStudentBackboneMixin, nn.Module):
    """Classification/regression head on top of the frozen (except one
    teacher's Convpass) StudentResEncUNet bottleneck. Mirrors
    gardening_tools.ClsRegHead's own pattern (global-avg-pool + Linear) so
    it's a drop-in shape-wise, reusing that class directly rather than
    re-implementing it."""

    stem_weight_name = None  # tells BaseModule not to try its own stem-repeat logic; we handle it ourselves

    # Encoder stages fed to MultiStageGeMHead -- skips enc_stage_0 (32ch,
    # stride=1, full-res low-level texture; spec "07. Task 1 Head
    # Architecture" Sec 1 explicitly excludes it as low classification value).
    MULTISTAGE_ENC_IDXS = [1, 2, 3, 4, 5]

    def __init__(self, input_channels: int, output_channels: int,
                 checkpoint_path: str, teacher_name: str, from_scratch: bool = False,
                 dropout_rate: float = 0.0, stem_mode: str = "frozen",
                 head_mode: str = "base", fusion_mode: str = "early"):
        """stem_mode (2026-08-11 multimodal-stem ablation -- see chat): how
        the pretrained single-channel stem handles multi-modality input.
          "frozen" (default, original behavior): repeat+average the
            pretrained 1-channel stem weight across input_channels, then
            freeze it like the rest of the backbone.
          "learnable": same channel-repeat init, but the repeated stem
            tensors (`_load_with_channel_repeat`'s repeated_keys) stay
            trainable alongside the teacher's Convpass path.
          "mixer": the pretrained stem is loaded and kept BYTE-IDENTICAL to
            the checkpoint (backbone built with in_channels=1, no repeat --
            `_load_with_channel_repeat`'s exact-match branch handles it,
            never touches the repeat branch -- stays frozen with the rest
            of the backbone). A new learnable 1x1x1 conv (`self.mixer`)
            maps the input_channels modalities down to 1 channel BEFORE the
            frozen stem ever sees the data, so the original pretrained stem
            weights are never modified or reinitialized at all.

        head_mode / fusion_mode (2026-08-12 head-architecture ablation, see
        task1_head_arch/ spec "07. Task 1 Head Architecture 실험 -- Multi-
        stage GeM + Late Fusion"):
          head_mode "base" (default): unchanged -- last encoder stage only
            + GAP + Linear (ClsRegHead).
          head_mode "multistage_gem": MultiStageGeMHead over
            MULTISTAGE_ENC_IDXS, each stage independently signed-GeM-pooled
            + LayerNorm'd before concat (see multistage_head.py).
          fusion_mode "early" (default): unchanged -- the full
            [B,input_channels,D,H,W] volume goes through the stem as one
            tensor.
          fusion_mode "late": each modality goes through the frozen,
            BYTE-IDENTICAL 1-channel pretrained stem separately
            ([B,M,D,H,W] -> [B*M,1,D,H,W] -> encoder -> reshape back to
            [B,M*C,d,h,w] per stage) instead of a channel-repeated or
            mixed-down stem -- requires stem_mode="frozen" since the other
            stem_mode values exist only to solve early fusion's
            multi-channel-stem problem, which late fusion doesn't have.
        """
        super().__init__()
        assert stem_mode in ("frozen", "learnable", "mixer"), f"unknown stem_mode={stem_mode!r}"
        assert head_mode in ("base", "multistage_gem"), f"unknown head_mode={head_mode!r}"
        assert fusion_mode in ("early", "late"), f"unknown fusion_mode={fusion_mode!r}"
        if fusion_mode == "late":
            assert stem_mode == "frozen", (
                "fusion_mode='late' pushes each modality through the native 1-channel "
                "pretrained stem separately, so stem_mode (which exists to handle "
                "multi-channel early-fusion input) is meaningless here -- leave it at "
                "the 'frozen' default."
            )
        self.teacher_name = teacher_name
        self.num_classes = output_channels
        self.stem_mode = stem_mode
        self.head_mode = head_mode
        self.fusion_mode = fusion_mode
        self.n_modalities = input_channels

        raw_sd = _load_checkpoint_state_dict(checkpoint_path)
        teachers, convpass_encoder, skip_alpha = _infer_student_config(raw_sd)
        assert teacher_name in teachers, f"{teacher_name!r} not in checkpoint's teacher roster {teachers}"

        backbone_in_channels = 1 if (stem_mode == "mixer" or fusion_mode == "late") else input_channels
        self.backbone = StudentResEncUNet(
            in_channels=backbone_in_channels, teachers=teachers, convpass=True,
            convpass_encoder=convpass_encoder, skip_alpha=skip_alpha, norm_conditional=False,
        )
        self._load_and_configure(checkpoint_path, backbone_in_channels, teacher_name, from_scratch,
                                  unfreeze_stem=(stem_mode == "learnable"))

        self.mixer = nn.Conv3d(input_channels, 1, kernel_size=1) if stem_mode == "mixer" else None

        fusion_factor = input_channels if fusion_mode == "late" else 1
        if head_mode == "base":
            bottleneck_channels = self.backbone.encoder.out_channels[-1] * fusion_factor
            self.head = ClsRegHead(
                input_channels=bottleneck_channels, output_channels=output_channels,
                pool_op=nn.AdaptiveAvgPool3d, dropout_rate=dropout_rate,
            )
        else:
            stage_channels = [self.backbone.encoder.out_channels[i] * fusion_factor
                               for i in self.MULTISTAGE_ENC_IDXS]
            self.head = MultiStageGeMHead(stage_channels, output_channels, dropout_rate=dropout_rate)

        trainable_names = [n for n, p in self.named_parameters() if p.requires_grad]
        print(f"[FomoStudentClsRegNet] head_mode={head_mode!r} fusion_mode={fusion_mode!r}: "
              f"{len(trainable_names)} trainable parameters total:")
        for n in trainable_names:
            print(f"  requires_grad=True: {n}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_stages = len(self.backbone.encoder.stages)
        if self.fusion_mode == "late":
            assert x.shape[1] == self.n_modalities, (
                f"expected {self.n_modalities} modalities, got {x.shape[1]}"
            )
            b, m = x.shape[0], x.shape[1]
            feats = self.backbone.forward_encoder_only(
                x.reshape(b * m, 1, *x.shape[2:]), teacher_name=self.teacher_name,
            )
            feats = {k: v.reshape(b, m * v.shape[1], *v.shape[2:]) for k, v in feats.items()}
        else:
            if self.mixer is not None:
                x = self.mixer(x)
            feats = self.backbone.forward_encoder_only(x, teacher_name=self.teacher_name)

        if self.head_mode == "base":
            return self.head([feats[f"enc_stage_{n_stages - 1}"]])
        stage_feats = [feats[f"enc_stage_{i}"] for i in self.MULTISTAGE_ENC_IDXS]
        return self.head(stage_feats)


class FomoStudentSegNet(FomoStudentBackboneMixin, BaseNet):
    """Segmentation head on top of the frozen (except one teacher's Convpass)
    StudentResEncUNet's full decoder. Unlike FomoStudentClsRegNet (encoder
    bottleneck only), this uses `forward_with_features` to run the WHOLE
    decoder and takes its last stage (`dec_stage_{N-1}`, 32 channels,
    DECODER_STAGE_SCALES[-1]=1.0 -- full input resolution, see
    networks/student.py) as the seg head's input.

    Subclasses gardening_tools.BaseNet (not bare nn.Module, unlike
    FomoStudentClsRegNet) purely to inherit `sliding_window_predict()` --
    asparagus's SegmentationModule.test_step/predict_step call
    `self.model.sliding_window_predict(...)` directly (not `forward`), and
    BaseNet's implementation is architecture-agnostic (only needs
    `self.forward(patch)` and `self.num_classes`, see BaseNet.py) so no
    override is needed here. BaseNet.load_state_dict is irrelevant to us --
    same as FomoStudentClsRegNet, checkpoint loading is done ourselves in
    `_load_and_configure`, never through Asparagus's own `weights=`
    mechanism (top-level `checkpoint_path=`/`weights=` stay unset so
    `resolve_checkpoint()` returns None and that path is never taken).

    deep_supervision is NOT supported (fixed single full-res output) --
    matches the original implementation plan's decision that rewiring
    Convpass for multi-scale deep-supervision outputs is out of scope; set
    `model.deep_supervision=false` in the Hydra config for this model.
    """

    stem_weight_name = None  # see FomoStudentClsRegNet -- tells BaseModule not to try its own stem-repeat logic

    def __init__(self, input_channels: int, output_channels: int,
                 checkpoint_path: str, teacher_name: str, from_scratch: bool = False):
        super().__init__()
        self.teacher_name = teacher_name
        self.num_classes = output_channels

        raw_sd = _load_checkpoint_state_dict(checkpoint_path)
        teachers, convpass_encoder, skip_alpha = _infer_student_config(raw_sd)
        assert teacher_name in teachers, f"{teacher_name!r} not in checkpoint's teacher roster {teachers}"

        self.backbone = StudentResEncUNet(
            in_channels=input_channels, teachers=teachers, convpass=True,
            convpass_encoder=convpass_encoder, skip_alpha=skip_alpha, norm_conditional=False,
        )
        self._load_and_configure(checkpoint_path, input_channels, teacher_name, from_scratch)

        final_channels = self.backbone.decoder_stage_channels[-1]  # 32, full-res dec_stage
        self.seg_head = nn.Conv3d(final_channels, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_with_features(x, teacher_name=self.teacher_name)
        full_res = feats[f"dec_stage_{len(self.backbone.decoder.stages) - 1}"]
        return self.seg_head(full_res)
