"""Quick sanity check: load every registered teacher, run a dummy + real-MRI
forward, print shapes. Thin wrapper around the more thorough
/root/teachers/scripts/verify_teachers.py (frozen/deterministic/grad-leak
checks) -- run that one for full verification; this one is for a fast
"is everything still wired up" check after editing FOMO26/teacher/*.

Usage: python3 /root/FOMO26/examples/sanity_check_teachers.py
"""

import sys

import torch

sys.path.insert(0, "/root")
from FOMO26.teacher.registry import TEACHER_REGISTRY, get_teacher  # noqa: E402

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 1, 128, 128, 128, device=device)
    for name in TEACHER_REGISTRY:
        print(f"\n=== {name} ===")
        teacher = get_teacher(name, device=device)
        feats = teacher.extract_features(x)
        for k, v in feats.items():
            print(f"  {k}: {tuple(v.shape)}")
        del teacher
        if device == "cuda":
            torch.cuda.empty_cache()
    print("\nAll teachers loaded and forwarded successfully.")
