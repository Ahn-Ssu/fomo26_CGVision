# Task 3 — Brain Age Regression

Preprocessing, stratified data splitting, and training/orchestration code
for brain-age regression finetuned on the `ver5` pretrained checkpoint,
following the same architecture-sweep methodology as Task1
(`training/asparagus_orchestrate_task3_head_arch.py`).

See `TASK3_HANDOFF.md` for the full handoff notes (preprocessing pipeline,
stratified split design, and the intended 20-job architecture sweep this
was prepared for), including a flagged Blackwell/PyTorch compatibility
gotcha relevant to running on newer GPU hardware.

No final submission container is included here — training/preprocessing
was completed and handed off for the architecture sweep to be run on
separate compute; see the handoff doc for status at time of handoff.
