"""Per-(teacher, stage) 1x1x1 Conv3d projection heads for feature-level
distillation.

Student decoder stages (networks/student.py) are deliberately built on the
same nnU-Net-standard 6-stage plan as two of our three live teachers
(VesselFM, BraTS), so spatial scales already match 1:1 at every stage
(dec_stage_0..4, scales [1/16,1/8,1/4,1/2,1]) -- see /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/02_TEACHER_REPORT.md Sec E.
Only channel counts differ, so all that's needed here is a 1x1x1 conv per
(teacher, stage) pair mapping student_channels -> teacher_channels, so the
distillation loss (losses/distillation.py) can be computed directly in the
teacher's own feature space.

Anatomix is the one teacher with a different (4-stage) plan -- it is mapped
onto student stages 1..4 (skipping student's deepest stage 0), per
/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/02_TEACHER_REPORT.md Sec E's mapping table. This is encoded in STAGE_MAP below,
not guessed at call time.

V-JEPA (2026-08-02): unlike the other teachers, its 4 hierarchical layers
(2/5/8/11) are all natively the SAME spatial resolution (a ViT doesn't
downsample across depth) -- there's no single "spatial scale ladder" to
align 1:1 the way BraTS/VesselFM's own decoders do. Per external
investigation (`/root/external_teacher_probe/`, see REPORT.md and the later
chat discussion on layer/stage pairing), the chosen mapping instead aligns
"how deep into the STUDENT's own processing order" a stage sits (encoder,
pre-bottleneck -> decoder, post-bottleneck) with "how semantically deep"
each V-JEPA layer is (primitive -> abstract), deliberately skipping the 8^3
bottleneck (enc_stage_4/dec_stage_0) since so few spatial positions survive
masking there that a gradient signal from it was judged too sparse to be
useful, and because BraTS/VesselFM's own dec_stage_0 already covers it:
    layer2  (primitive)  -> student enc_stage_2 (32^3, pre-bottleneck)
    layer5                -> student enc_stage_3 (16^3, pre-bottleneck)
    layer8                -> student dec_stage_1 (16^3, post-bottleneck)
    layer11 (semantic)   -> student dec_stage_2 (32^3, post-bottleneck)
This is the first teacher whose STAGE_MAP needs to target the student's
ENCODER (not just decoder) -- see the (side, idx) key format below.

VoCo-OpenMind (2026-08-02, teacher/voco_teacher.py): unlike V-JEPA, this one
IS architecturally identical to the student's own encoder plan (channels
[32,64,128,256,320,320], strides [1,2,2,2,2,2] -- see voco_teacher.py's
docstring for the empirical verification) and has NO decoder checkpoint at
all, so the mapping is the simplest possible: student enc_stage_i <-> VoCo's
own stage i, 1:1, for i=0..4 (student's stage 5, stride 32, has no VoCo
counterpart used -- VoCo's own stride-32 stage was rank-deficient even
pooled across 30 volumes in the PCA probe, dropped there and so absent from
voco_teacher.py's feature_specs too). ALL of VoCo's targets are "enc" side --
see is_encoder_only_teacher() below, which lets the training loop skip the
student's decoder forward+backward entirely on a VoCo-sampled step (no
decoder counterpart exists for VoCo regardless, so that compute would
otherwise be pure waste).

STAGE_MAP key format: either a plain int `student_idx` (legacy shorthand,
means student DECODER stage `student_idx` -- kept byte-for-byte compatible
with existing checkpoints' projection-head naming, see _head_key) or a
`(side, student_idx)` tuple where side is "enc" or "dec", for teachers that
need to target student ENCODER stages too.

Usage:
    heads = build_projection_heads(dec_channels=[320,256,128,64,32],
                                    enc_channels=[32,64,128,256,320,320],
                                    teacher_feature_specs=teacher.feature_specs)
    projected = heads.project("vesselfm", student_feats)  # student_feats from
                                                            # forward_with_features(..., include_encoder=True)
"""

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

StageKey = Union[int, Tuple[str, int]]

# student_stage_key -> teacher_stage_idx, per teacher. student_stage_key is
# either a plain int (decoder stage, legacy) or ("enc"|"dec", idx).
# Derived from /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/02_TEACHER_REPORT.md Sec E (VesselFM/BraTS: 1:1; Anatomix: shifted by 1).
STAGE_MAP: Dict[str, Dict[StageKey, int]] = {
    "anatomix": {1: 0, 2: 1, 3: 2, 4: 3},  # student stage 0 has no Anatomix counterpart
    "anatomix+brains": {1: 0, 2: 1, 3: 2, 4: 3},
    "vesselfm": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
    "brats": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
    "vjepa": {("enc", 2): 0, ("enc", 3): 1, ("dec", 1): 2, ("dec", 2): 3},
    "voco": {("enc", 0): 0, ("enc", 1): 1, ("enc", 2): 2, ("enc", 3): 3, ("enc", 4): 4},
}


# V-JEPA has no live model in this codebase (its features come precomputed
# from /root/external_teacher_probe/vjepa_cache/, see
# data/vjepa_cache_dataset.py) so there's no `teacher.feature_specs` to read
# channel counts off of -- fixed here instead (ViT-B hidden dim, all 4
# hierarchical layers share it).
VJEPA_FEATURE_SPECS = {f"dec_stage_{i}": {"channels": 768} for i in range(4)}


def _parse_stage_key(key: StageKey) -> Tuple[str, int]:
    if isinstance(key, tuple):
        return key
    return "dec", key


def _head_key(side: str, idx: int) -> str:
    # "dec" keeps the EXACT legacy naming ("stage_{idx}") so existing
    # checkpoints' proj_heads_state_dict keys are unaffected -- only "enc"
    # (new, never existed before) gets a distinguishing prefix.
    return f"stage_{idx}" if side == "dec" else f"stage_{side}_{idx}"


def is_encoder_only_teacher(teacher_name: str) -> bool:
    """True iff EVERY STAGE_MAP entry for this teacher targets the student's encoder -- lets the
    training loop call student.forward_encoder_only() (skips the decoder's compute+backward
    entirely) instead of forward_with_features(include_encoder=True) (which still runs the full
    decoder). Derived from STAGE_MAP itself rather than a separately-maintained set, so it can't
    drift out of sync with the actual mapping."""
    smap = STAGE_MAP.get(teacher_name, {})
    return bool(smap) and all(_parse_stage_key(k)[0] == "enc" for k in smap)


def teacher_needs_encoder_features(teacher_name: str) -> bool:
    """True iff ANY STAGE_MAP entry for this teacher targets the student's encoder (covers both
    encoder-only teachers like VoCo and mixed ones like V-JEPA) -- tells the caller whether
    forward_with_features(..., include_encoder=True) is needed at all."""
    smap = STAGE_MAP.get(teacher_name, {})
    return any(_parse_stage_key(k)[0] == "enc" for k in smap)


class ProjectionHeads(nn.Module):
    """Holds one 1x1x1 Conv3d per (teacher, mapped student stage), trainable
    (these are part of the student's optimizer param group, not the frozen
    teacher)."""

    def __init__(self, dec_channels: List[int], enc_channels: List[int],
                 teacher_specs: Dict[str, Dict[str, dict]]):
        super().__init__()
        self.heads = nn.ModuleDict()
        self.stage_map = {}
        channels_by_side = {"dec": dec_channels, "enc": enc_channels}
        for teacher_name, specs in teacher_specs.items():
            if teacher_name not in STAGE_MAP:
                continue
            smap = STAGE_MAP[teacher_name]
            self.stage_map[teacher_name] = smap
            teacher_heads = nn.ModuleDict()
            for student_key, teacher_idx in smap.items():
                side, idx = _parse_stage_key(student_key)
                teacher_stage_key = f"dec_stage_{teacher_idx}"
                if teacher_stage_key not in specs:
                    continue
                out_c = specs[teacher_stage_key]["channels"]
                in_c = channels_by_side[side][idx]
                teacher_heads[_head_key(side, idx)] = nn.Conv3d(in_c, out_c, kernel_size=1)
            self.heads[teacher_name] = teacher_heads

    def project(self, teacher_name: str, student_feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Returns {teacher_stage_key: projected_student_feature} for every
        mapped stage of the given teacher. student_feats must contain
        enc_stage_* keys too (forward_with_features(..., include_encoder=True))
        whenever the teacher's STAGE_MAP references an "enc" side entry."""
        out = {}
        smap = self.stage_map[teacher_name]
        heads = self.heads[teacher_name]
        for student_key, teacher_idx in smap.items():
            side, idx = _parse_stage_key(student_key)
            key = _head_key(side, idx)
            if key not in heads:
                continue
            student_feat = student_feats[f"{side}_stage_{idx}"]
            out[f"dec_stage_{teacher_idx}"] = heads[key](student_feat)
        return out


def build_projection_heads(dec_channels: List[int], enc_channels: List[int],
                            teacher_feature_specs: Dict[str, Dict[str, dict]]) -> ProjectionHeads:
    return ProjectionHeads(dec_channels, enc_channels, teacher_feature_specs)
