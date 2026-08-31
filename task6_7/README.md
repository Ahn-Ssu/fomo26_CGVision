# Task 6/7 — Linear Probing / Fairness Embeddings

Frozen-embedding extraction from the `ver5` pretrained checkpoint for
downstream linear probing. See `PREREGISTRATION.md` for the full pre-
registered plan (this task's submissions are evaluated by an external
linear probe trained on the embeddings we provide, so the embedding design
itself — not any trainable head — is the whole submission).

## Neutral teacher-context

Both submissions use a "neutral" teacher-context: the reference teacher's
Convpass adapter gate is zeroed at every stage, which collapses
`out = block_out + gate * convpass_out` to exactly `out = block_out` —
verified bit-identical to a Convpass-free forward pass
(`verify_neutral_gate.py`, `verify_neutral_gate_decoder.py`). This avoids
arbitrarily picking one of the 5 pretraining teachers' adapters for a task
that has no obvious "correct" teacher to condition on.

## `submission_v1/` — deepest-stage GAP (submitted, scored)

Global-average-pools only the final (6th) encoder stage → 320-dim
embedding. fp32, single whole-volume forward pass (no sliding-window
tiling — an earlier bf16+sliding-window variant scored dramatically worse
on the real leaderboard, traced to `InstanceNorm3d` computing genuinely
different statistics over a 128³ training-patch-sized window vs. a whole,
larger volume — see this file's own docstring history for the measured
embedding-space L2 distances).

## `submission_multistage_gap/` — multistage GAP concat (submitted)

Concatenates GAP-pooled features from stages 1–5 (dropping only the
shallowest, least-semantic 32-channel stage): 64+128+256+320+320 =
**1088-dim**. Motivated by the same multistage-vs-final-stage-only finding
from Task1/Task5's own architecture sweep (multistage clearly beat
final-stage-only there, both in held-out AUROC and overfitting behavior) —
deep layers converge toward the pretraining objective's own specialization,
discarding information a *different* downstream task may still need.
GAP (not GMP) was chosen by discussion, in the absence of local ground-
truth labels for this task: the task's own "fairness" framing suggests a
diffuse/holistic downstream target (e.g. demographic or scanner-related
structural variation) rather than a localized-lesion-style signal, which
favors averaging over max-pooling.

Both variants pad the preprocessed volume to a **minimum** of 64 per axis
(not just "next multiple of 32") before the 5 stride-2 downsamples —
found necessary after `container-validator`'s own tiny (32×32×16) synthetic
test fixture collapsed to a literal 1×1×1 spatial size at the deepest
stage, which `InstanceNorm3d` (no running stats — it normalizes over the
current input's own spatial extent every call) rejects outright. Real
scans are always far larger than this floor, so it only ever activates as
a crash guard on tiny/synthetic inputs.
