"""Standalone (no asparagus/lightning/hydra dependency) port of
FomoStudentSegNet -- FOMO26 Task 4 (Multiclass Tissue Segmentation)
submission containers' model class -- for finetuning ver5 pretrain
checkpoints with a "teacher-specific-parameter-only" scheme: the shared
backbone is frozen, only ONE teacher's Convpass adapter path (+ a newly
added task head) stays trainable.

Ported verbatim from common/asparagus/asparagus/modules/networks/
fomo26_student.py (the actual class used both for training, via
configs/model/fomo26_student_seg.yaml, and for this submission's own
predict.py) + student.py (verbatim, copied alongside as student.py) -- only
the dead dev-machine `sys.path.insert(...)` line and the `networks.student`
import path (flattened to a same-directory `from student import ...`, since
this file and student.py both live in submission/model/) were removed;
no architectural or numerical change.

Unlike Task1/Task5's ported `net.py`, this class's constructor still
requires a real `checkpoint_path` (the ver5 pretrain checkpoint) --
`_infer_student_config()` reads real config info that this port doesn't
hardcode, so the same "skip the transplant, hardcode the flags" trick isn't
applied here. The pretrain checkpoint must be supplied separately (not
included in this repository, see the top-level README).

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

import itertools
import re

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from gardening_tools.modules.networks.BaseNet import BaseNet
from gardening_tools.modules.networks.components.heads import ClsRegHead
from student import StudentResEncUNet  # noqa: E402


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
                             unfreeze_stem: bool = False,
                             unfreeze_encoder_convpass: bool = True) -> None:
        """unfreeze_encoder_convpass (2026-08-16, encoder/decoder arch-freeze
        search -- see task2_archsearch/SPEC.md): the original behavior always
        unfroze the selected teacher's Convpass path across the WHOLE
        backbone (encoder AND decoder together, one loop, no way to pick
        just one). Setting this False leaves the encoder's own selected-
        teacher Convpass frozen too, so the encoder does a byte-identical
        frozen forward pass (0 trainable encoder params) -- decoder-side
        Convpass unfreezing is untouched either way, since the seg
        subclass's own freeze_decoder/decoder_scheme logic already fully
        governs decoder trainability independently."""
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
            is_target_convpass = (f".convpass.{teacher_name}." in n
                    or f".convpass_gate.{teacher_name}" in n
                    or f".skip_alpha.{teacher_name}" in n)
            if not is_target_convpass:
                continue
            if n.startswith("encoder.") and not unfreeze_encoder_convpass:
                continue
            p.requires_grad_(True)
            trainable_names.append(n)
        assert trainable_names, (
            f"No Convpass/skip_alpha parameters found for teacher_name={teacher_name!r} -- "
            f"check spelling against the checkpoint's own teacher roster, or "
            f"unfreeze_encoder_convpass=False left nothing trainable if this is an "
            f"encoder-only architecture."
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

    def _reinit_teacher_convpass(self, teacher_name: str, scope: str = "both") -> None:
        """Re-randomizes the chosen teacher's Convpass adapter (down/dw/up +
        gate, and skip_alpha if present) in place, matching Convpass3D's own
        __init__ scheme exactly (networks/student.py) -- used for the
        from-scratch-vs-pretrained-init ablation. Params stay trainable
        (requires_grad already set True by the caller).

        scope (2026-08-16, encoder/decoder arch-freeze search): "both"
        (original behavior) reinits encoder+decoder Convpass together;
        "encoder"/"decoder" reinits only that half, leaving the other half's
        Convpass at its checkpoint-loaded (pretrained) init -- needed
        because the new search treats "encoder Convpass init" and "decoder
        Convpass init" as independent axes, whereas the old single from_scratch
        flag always coupled them."""
        assert scope in ("both", "encoder", "decoder"), f"unknown scope={scope!r}"
        stages = []
        if scope in ("both", "encoder"):
            stages += list(self.backbone.encoder.stages)
        if scope in ("both", "decoder"):
            stages += list(self.backbone.decoder.stages)
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
        assert n_reinit, f"from_scratch reinit (scope={scope!r}) but found no Convpass module for teacher_name={teacher_name!r}"
        print(f"[FomoStudent] re-initialized {n_reinit} Convpass adapters for {teacher_name!r} (scope={scope!r})")

    def _reinit_decoder_backbone(self) -> None:
        """Randomly reinitializes the decoder's non-Convpass backbone
        submodules (conv/norm layers) in place, via each submodule's own
        reset_parameters() -- used for decoder_scheme="scratch" (2026-08-16
        arch-freeze search). Does NOT touch Convpass/gate/skip_alpha
        submodules (any teacher's), so their checkpoint-loaded or
        from-scratch-reinit state (set separately) is preserved. Uses each
        module's own reset_parameters() rather than reimplementing an
        init scheme by hand, so Conv3d gets its normal kaiming-uniform reset
        and InstanceNorm3d gets its normal running-stats/affine reset,
        matching what a freshly-constructed decoder would have looked like."""
        n_reset = 0
        for n, m in self.backbone.decoder.named_modules():
            if ".convpass" in n or "convpass_gate" in n or "skip_alpha" in n:
                continue
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
                n_reset += 1
        assert n_reset, "decoder_scheme='scratch' but found no resettable (non-Convpass) decoder submodules"
        print(f"[FomoStudentSegNet] decoder_scheme='scratch': reset {n_reset} decoder backbone submodules to random init")


class FomoStudentClsRegNet(FomoStudentBackboneMixin, nn.Module):
    """Classification/regression head on top of the frozen (except one
    teacher's Convpass) StudentResEncUNet bottleneck. Mirrors
    gardening_tools.ClsRegHead's own pattern (global-avg-pool + Linear) so
    it's a drop-in shape-wise, reusing that class directly rather than
    re-implementing it."""

    stem_weight_name = None  # tells BaseModule not to try its own stem-repeat logic; we handle it ourselves

    def __init__(self, input_channels: int, output_channels: int,
                 checkpoint_path: str, teacher_name: str, from_scratch: bool = False,
                 dropout_rate: float = 0.0, stem_mode: str = "frozen"):
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
        """
        super().__init__()
        assert stem_mode in ("frozen", "learnable", "mixer"), f"unknown stem_mode={stem_mode!r}"
        self.teacher_name = teacher_name
        self.num_classes = output_channels
        self.stem_mode = stem_mode

        raw_sd = _load_checkpoint_state_dict(checkpoint_path)
        teachers, convpass_encoder, skip_alpha = _infer_student_config(raw_sd)
        assert teacher_name in teachers, f"{teacher_name!r} not in checkpoint's teacher roster {teachers}"

        backbone_in_channels = 1 if stem_mode == "mixer" else input_channels
        self.backbone = StudentResEncUNet(
            in_channels=backbone_in_channels, teachers=teachers, convpass=True,
            convpass_encoder=convpass_encoder, skip_alpha=skip_alpha, norm_conditional=False,
        )
        self._load_and_configure(checkpoint_path, backbone_in_channels, teacher_name, from_scratch,
                                  unfreeze_stem=(stem_mode == "learnable"))

        self.mixer = nn.Conv3d(input_channels, 1, kernel_size=1) if stem_mode == "mixer" else None

        bottleneck_channels = self.backbone.encoder.out_channels[-1]
        self.head = ClsRegHead(
            input_channels=bottleneck_channels, output_channels=output_channels,
            pool_op=nn.AdaptiveAvgPool3d, dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mixer is not None:
            x = self.mixer(x)
        feats = self.backbone.forward_encoder_only(x, teacher_name=self.teacher_name)
        bottleneck = feats[f"enc_stage_{len(self.backbone.encoder.stages) - 1}"]
        return self.head([bottleneck])


class FomoStudentSegNet(FomoStudentBackboneMixin, BaseNet):
    """Segmentation head on top of the (optionally decoder-frozen)
    StudentResEncUNet's full decoder. Unlike FomoStudentClsRegNet (encoder
    bottleneck only), this uses `forward_with_features` to run the WHOLE
    decoder and takes its last stage (`dec_stage_{N-1}`, 32 channels,
    DECODER_STAGE_SCALES[-1]=1.0 -- full input resolution, see
    networks/student.py) as the seg head's input.

    Subclasses gardening_tools.BaseNet (not bare nn.Module, unlike
    FomoStudentClsRegNet) so asparagus's SegmentationModule.test_step/
    predict_step (which call `self.model.sliding_window_predict(...)`
    directly, not `forward`) keep working -- but `sliding_window_predict`
    itself is OVERRIDDEN below (see its own docstring for why: BaseNet's
    version has no Gaussian blending and no overlap-count normalization).
    BaseNet.load_state_dict is irrelevant to us -- same as
    FomoStudentClsRegNet, checkpoint loading is done ourselves in
    `_load_and_configure`, never through Asparagus's own `weights=`
    mechanism (top-level `checkpoint_path=`/`weights=` stay unset so
    `resolve_checkpoint()` returns None and that path is never taken).

    deep_supervision is NOT supported (fixed single full-res output) --
    matches the original implementation plan's decision that rewiring
    Convpass for multi-scale deep-supervision outputs is out of scope; set
    `model.deep_supervision=false` in the Hydra config for this model.

    freeze_decoder (2026-08-11, FOMO25 Appendix B revision -- see
    segmentation/NOTES.md section 6.3): FOMO25's segmentation task showed
    full-backbone-frozen (encoder+decoder) was a common trait of every
    bottom-of-leaderboard team (TUKE 0.046, biopet 0.005, our own ashash
    0.003), while the strongest frozen-encoder result (Curia, 0.110) trained
    the decoder. Two arms:
      freeze_decoder=False ("arm A", the new default, submission-priority):
        encoder stays frozen (only its selected-teacher Convpass path is
        trainable, as before) but the decoder's own backbone weights
        (upsample/block conv+norm layers) are ALSO unfrozen -- everything in
        `self.backbone.decoder.named_parameters()` EXCEPT other-teachers'
        `.convpass.`/`.convpass_gate.`/`.skip_alpha.` submodules, which must
        stay frozen (same "only the selected teacher's adapter path trains"
        invariant `_load_and_configure` already enforces for the encoder).
      freeze_decoder=True ("arm B", research arm, the ORIGINAL spec's
        policy): decoder stays fully frozen except its own selected-teacher
        Convpass path, exactly as `_load_and_configure` already sets up by
        itself -- this branch is a no-op.

    encoder_convpass_frozen / decoder_scheme (2026-08-16, encoder/decoder
    arch-freeze search -- see task2_archsearch/SPEC.md, motivated by a
    sibling Task 1 classification finding that a "learnable stem" arm
    underperformed frozen/shared alternatives): two ADDITIVE axes, both
    default to reproducing the old armA/armB behavior exactly so no
    existing caller/config is affected.
      encoder_convpass_frozen=False (default): unchanged -- encoder's
        selected-teacher Convpass is trainable, as it always was.
      encoder_convpass_frozen=True: encoder's Convpass ALSO stays frozen
        (0 trainable encoder params -- combine with stem_mode="frozen" for
        a fully frozen encoder forward pass).
      decoder_scheme=None (default): ignored -- decoder trainability is
        governed purely by freeze_decoder/from_scratch as before.
      decoder_scheme set (overrides freeze_decoder for this instance):
        "finetune"           -> freeze_decoder=False, no reinit (= old armA)
        "convpass_pretrained"-> freeze_decoder=True,  no reinit (= old armB)
        "convpass_scratch"   -> freeze_decoder=True,  decoder-only Convpass
                                 reinit (encoder Convpass, if unfrozen,
                                 keeps its pretrained init)
        "scratch"             -> freeze_decoder=False, PLUS the decoder's
                                 non-Convpass backbone (conv/norm layers) is
                                 randomly reinitialized before being made
                                 trainable (see _reinit_decoder_backbone)
    """

    stem_weight_name = None  # see FomoStudentClsRegNet -- tells BaseModule not to try its own stem-repeat logic

    def __init__(self, input_channels: int, output_channels: int,
                 checkpoint_path: str, teacher_name: str, from_scratch: bool = False,
                 stem_mode: str = "frozen", freeze_decoder: bool = False,
                 encoder_convpass_frozen: bool = False, decoder_scheme: str = None):
        super().__init__()
        assert stem_mode in ("frozen", "learnable", "mixer"), f"unknown stem_mode={stem_mode!r}"
        assert decoder_scheme in (None, "scratch", "finetune", "convpass_pretrained", "convpass_scratch"), (
            f"unknown decoder_scheme={decoder_scheme!r}"
        )
        self.teacher_name = teacher_name
        self.num_classes = output_channels
        self.stem_mode = stem_mode
        self.encoder_convpass_frozen = encoder_convpass_frozen
        self.decoder_scheme = decoder_scheme

        raw_sd = _load_checkpoint_state_dict(checkpoint_path)
        teachers, convpass_encoder, skip_alpha = _infer_student_config(raw_sd)
        assert teacher_name in teachers, f"{teacher_name!r} not in checkpoint's teacher roster {teachers}"

        backbone_in_channels = 1 if stem_mode == "mixer" else input_channels
        self.backbone = StudentResEncUNet(
            in_channels=backbone_in_channels, teachers=teachers, convpass=True,
            convpass_encoder=convpass_encoder, skip_alpha=skip_alpha, norm_conditional=False,
        )

        if decoder_scheme is not None:
            freeze_decoder = decoder_scheme in ("convpass_pretrained", "convpass_scratch")
        self.freeze_decoder = freeze_decoder

        self._load_and_configure(
            checkpoint_path, backbone_in_channels, teacher_name,
            from_scratch=(False if decoder_scheme is not None else from_scratch),
            unfreeze_stem=(stem_mode == "learnable"),
            unfreeze_encoder_convpass=not encoder_convpass_frozen,
        )
        if decoder_scheme == "convpass_scratch":
            self._reinit_teacher_convpass(teacher_name, scope="decoder")

        self.mixer = nn.Conv3d(input_channels, 1, kernel_size=1) if stem_mode == "mixer" else None

        if not freeze_decoder:
            decoder_names = []
            for n, p in self.backbone.decoder.named_parameters():
                if ".convpass." in n or ".convpass_gate." in n or ".skip_alpha." in n:
                    continue  # non-selected teachers' adapter paths must stay frozen -- same invariant as the encoder
                p.requires_grad_(True)
                decoder_names.append(n)
            print(f"[FomoStudentSegNet] freeze_decoder=False: {len(decoder_names)} decoder backbone "
                  f"tensors unfrozen (upsample/block conv+norm; other-teacher Convpass paths stay frozen)")
            if decoder_scheme == "scratch":
                self._reinit_decoder_backbone()
        else:
            print(f"[FomoStudentSegNet] freeze_decoder=True: decoder backbone stays frozen "
                  f"(only its own selected-teacher Convpass path is trainable, via _load_and_configure"
                  f"{', reinitialized from scratch' if decoder_scheme == 'convpass_scratch' else ''})")
        print(f"[FomoStudentSegNet] encoder_convpass_frozen={encoder_convpass_frozen}, "
              f"decoder_scheme={decoder_scheme!r}")

        final_channels = self.backbone.decoder_stage_channels[-1]  # 32, full-res dec_stage
        self.seg_head = nn.Conv3d(final_channels, output_channels, kernel_size=1)

        trainable_names = [n for n, p in self.named_parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[FomoStudentSegNet] FULL trainable-parameter dump ({len(trainable_names)} tensors, "
              f"{n_trainable}/{n_total} params = {100 * n_trainable / n_total:.2f}%):")
        for n in trainable_names:
            print(f"  requires_grad=True: {n}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mixer is not None:
            x = self.mixer(x)
        feats = self.backbone.forward_with_features(x, teacher_name=self.teacher_name)
        full_res = feats[f"dec_stage_{len(self.backbone.decoder.stages) - 1}"]
        return self.seg_head(full_res)

    def sliding_window_predict(self, data: torch.Tensor, patch_size, overlap: float,
                                mirror: bool = False) -> torch.Tensor:
        """OVERRIDES gardening_tools.BaseNet.sliding_window_predict (2026-08-11,
        FOMO25 Appendix B revision -- segmentation/NOTES.md section 6.9).
        Kept as a drop-in replacement (identical call signature -- asparagus's
        SegmentationModule.test_step/predict_step call
        `self.model.sliding_window_predict(data=x, patch_size=..., overlap=0.5)`
        unmodified) because BaseNet's own implementation was found to have NO
        Gaussian blending and NO overlap-count normalization at all (window
        overlaps are simply summed via `canvas[...] += out`, never divided --
        verified by reading BaseNet.py directly), and `mirror=True` is never
        actually passed by SegmentationModule, so TTA mirroring is effectively
        dead code there too. The spec requires both Gaussian weighting and
        mirroring, so this uses `monai.inferers.sliding_window_inference`
        (mode="gaussian", already handles overlap normalization correctly)
        instead of re-deriving that math by hand.
        """
        from monai.inferers import sliding_window_inference

        predictor = self._mirror_tta_forward if mirror else self.forward
        return sliding_window_inference(
            inputs=data,
            roi_size=list(patch_size),
            sw_batch_size=1,
            predictor=predictor,
            overlap=overlap,
            mode="gaussian",
            padding_mode="constant",
            cval=0.0,
        )

    def _mirror_tta_forward(self, patch: torch.Tensor) -> torch.Tensor:
        """Averages predictions over all 2**n_spatial_dims flip combinations
        of `patch` (un-flipping each prediction before averaging) -- same set
        of flips BaseNet.sliding_window_predict's own `mirror=True` branch
        used (spatial dims 2..4 for 3D, 8 combinations total), just computed
        per-patch here so it composes with monai's windowing/blending."""
        spatial_dims = list(range(2, patch.ndim))
        preds = []
        for r in range(len(spatial_dims) + 1):
            for dims in itertools.combinations(spatial_dims, r):
                flipped = torch.flip(patch, dims) if dims else patch
                out = self.forward(flipped)
                preds.append(torch.flip(out, dims) if dims else out)
        return torch.stack(preds, dim=0).mean(dim=0)
