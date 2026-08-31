"""Frozen teacher wrapper interface for feature-level distillation.

Invariants every subclass must uphold (enforced/checked in tests/verify_teachers.py):
  - all parameters requires_grad=False
  - always in eval() mode
  - forward runs under torch.no_grad()
  - returns intermediate feature dict, not just final output
  - handles its own teacher-specific preprocessing internally

Input contract: the `x` tensor passed to preprocess()/extract_features() is
assumed to already be asparagus's foreground z-normalized volume (see
PRE_ANALYSIS.md Sec 3). Subclasses whose native normalization is NOT
rank-preserving (norm_type == "absolute") must instead accept raw intensities
via `meta={"raw": raw_tensor}` or document why they cannot be used at all.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn


class BaseTeacher(nn.Module, ABC):
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._features: Dict[str, torch.Tensor] = {}
        self._hook_handles = []
        self._build_model()
        self._register_hooks()
        self._freeze()
        self.to(device)

    @abstractmethod
    def _build_model(self) -> None:
        """Construct architecture and load checkpoint. Must verify strict=True
        (or explicitly document+justify why not) and raise on mismatch."""

    @abstractmethod
    def _register_hooks(self) -> None:
        """Register forward hooks that populate self._features with
        {stage_name: tensor} on every forward call."""

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):
        # Hard-block accidental .train() calls anywhere in the training loop.
        if mode:
            raise RuntimeError(f"{self.__class__.__name__} is a frozen teacher; .train() is disabled.")
        return super().train(False)

    @abstractmethod
    def preprocess(self, x: torch.Tensor, meta: Optional[dict] = None) -> torch.Tensor:
        """Teacher-specific normalization applied to an asparagus-normalized
        (or, if norm_type=='absolute', raw-via-meta) input volume."""

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, meta: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        self._features.clear()
        inp = self.preprocess(x, meta=meta)
        out = self(inp)
        if not self._features:
            # Model has no internal stage hooks (e.g. pure feature extractor);
            # subclasses may populate self._features inside forward() itself
            # instead of via hooks -- fall back to final output.
            self._features["final"] = out if not isinstance(out, (list, tuple)) else out[0]
        return dict(self._features)

    @property
    @abstractmethod
    def feature_specs(self) -> Dict[str, dict]:
        """{stage_name: {"channels": int, "spatial_scale": float, "module_path": str}}"""

    @property
    @abstractmethod
    def norm_type(self) -> str:
        """'percentile' | 'absolute' | 'none'"""

    @property
    @abstractmethod
    def input_requirements(self) -> dict:
        """{"modalities": [...] | "any", "spacing": tuple | "any",
            "patch_size": tuple, "skull_stripped": bool | "any",
            "in_channels": int, "is_3d": bool}"""
