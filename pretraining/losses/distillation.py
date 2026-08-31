"""Feature-level distillation loss.

Phase 1 (per user request, /root/00_PRE_ANALYSIS.md): a simple single-student
f_theta pipeline, one frozen teacher forward per step (episodic sampling,
sampler/episodic.py), no LoRA / teacher-specific adapters yet.

Loss per matched stage = (1 - cosine_similarity) + mse_weight * MSE, averaged
over matched stages for whichever teacher was sampled this step. Cosine
similarity is computed channel-wise per spatial location (dim=1), matching
common feature-distillation practice (e.g. DINO/iBOT-style patch losses --
see /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/01_ASPARAGUS_ANALYSIS.md's note on modules/losses/ibot.py as the closest
in-repo precedent) since teacher/student feature magnitudes are not expected
to match on an absolute scale.
"""

from typing import Dict

import torch
import torch.nn.functional as F


def stage_distillation_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor,
                             mse_weight: float = 0.1) -> torch.Tensor:
    if student_feat.shape[2:] != teacher_feat.shape[2:]:
        teacher_feat = F.interpolate(teacher_feat, size=student_feat.shape[2:], mode="trilinear", align_corners=False)
    cos = F.cosine_similarity(student_feat, teacher_feat.detach(), dim=1)  # (B, D, H, W)
    cos_loss = (1.0 - cos).mean()
    mse_loss = F.mse_loss(student_feat, teacher_feat.detach())
    return cos_loss + mse_weight * mse_loss


def episodic_distillation_loss(projected_student_feats: Dict[str, torch.Tensor],
                                teacher_feats: Dict[str, torch.Tensor],
                                mse_weight: float = 0.1) -> Dict[str, torch.Tensor]:
    """projected_student_feats / teacher_feats: {dec_stage_i: tensor}, already
    matched by ProjectionHeads.project(). Returns per-stage losses + 'total'."""
    losses = {}
    matched_keys = [k for k in projected_student_feats if k in teacher_feats]
    assert matched_keys, "No matched stages between projected student features and teacher features."
    total = 0.0
    for k in matched_keys:
        loss_k = stage_distillation_loss(projected_student_feats[k], teacher_feats[k], mse_weight=mse_weight)
        losses[k] = loss_k
        total = total + loss_k
    losses["total"] = total / len(matched_keys)
    return losses
