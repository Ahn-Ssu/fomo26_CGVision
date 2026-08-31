"""Central registry mapping teacher name -> wrapper class + default checkpoint arg.

Populated incrementally as each teacher wrapper is implemented and verified.
Entries marked BLOCKED are intentionally absent from TEACHER_REGISTRY until
their blocker is resolved (see /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/02_TEACHER_REPORT.md Sec I).
"""

from .anatomix_teacher import AnatomixTeacher
from .vesselfm_teacher import VesselFMTeacher, _DEFAULT_CKPT as _VESSELFM_DEFAULT_CKPT
from .brats_teacher import BraTSTeacher, _DEFAULT_DIR as _BRATS_DEFAULT_DIR
from .voco_teacher import VoCoTeacher, _DEFAULT_CKPT as _VOCO_DEFAULT_CKPT

TEACHER_REGISTRY = {
    "anatomix": (AnatomixTeacher, "anatomix"),
    "anatomix+brains": (AnatomixTeacher, "anatomix+brains"),
    "vesselfm": (VesselFMTeacher, _VESSELFM_DEFAULT_CKPT),
    "brats": (BraTSTeacher, _BRATS_DEFAULT_DIR),
    "voco": (VoCoTeacher, _VOCO_DEFAULT_CKPT),
}


def get_teacher(name: str, device: str = "cuda"):
    if name not in TEACHER_REGISTRY:
        raise KeyError(f"Unknown teacher '{name}'. Available: {list(TEACHER_REGISTRY)}")
    cls, ckpt = TEACHER_REGISTRY[name]
    return cls(ckpt, device=device)
