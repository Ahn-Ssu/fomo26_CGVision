"""Episodic teacher sampler: one teacher chosen per training step (episode),
per PRE_ANALYSIS.md's design ("Episode마다 다른 teacher/task를 샘플링, VTM의
episodic scheme 차용"). Phase 1: uniform random by default, with optional
fixed weights (e.g. to implement the BraTS modality-confidence weighting
noted in /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/02_TEACHER_REPORT.md/PRE_ANALYSIS.md Sec 5.1 once modality metadata is
wired through from the dataset -- not yet done in Phase 1, see
data/fomo300k_dataset.py's `modality` field which is already returned but not
yet consumed here).
"""

import random
from typing import Dict, List, Optional


class EpisodicTeacherSampler:
    def __init__(self, teacher_names: List[str], weights: Optional[Dict[str, float]] = None, seed: Optional[int] = None):
        self.teacher_names = list(teacher_names)
        if weights is None:
            weights = {name: 1.0 for name in self.teacher_names}
        missing = set(self.teacher_names) - set(weights)
        assert not missing, f"Missing sampling weight for teachers: {missing}"
        self.weights = [weights[name] for name in self.teacher_names]
        self._rng = random.Random(seed)

    def sample(self) -> str:
        return self._rng.choices(self.teacher_names, weights=self.weights, k=1)[0]
