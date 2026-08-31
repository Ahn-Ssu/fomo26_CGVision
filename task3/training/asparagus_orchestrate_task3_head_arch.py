"""Task 3 (Brain Age regression, single T1w modality) head-architecture
comparison -- same 2 head_mode arms already validated on Task 1
(base = last-stage+GAP+Linear vs multistage_gem = multi-stage GeM pooling,
see /root/task1_head_arch/), but Task 3 has only 1 modality, so `fusion_mode`
is moot (fixed at "early") and `stem_mode` is fixed at "frozen" (input_channels=1
already matches the pretrained stem's native channel count exactly -- no
channel-repeat needed, so "learnable"/"mixer" stem_mode don't apply and
"learnable" would actually hit an assertion error, see FomoStudentClsRegNet's
docstring). So the search space really is just {base, multistage_gem} x 10
folds = 20 jobs, per the user's framing ("single vs multi stage features").

**RUN THIS ON THE TARGET MACHINE (mk8, 2x RTX 5090 32GB), NOT on the machine
that authored it** -- see TASK3_HANDOFF.md for the full environment setup
this depends on (torch/CUDA version for Blackwell/sm_120, file layout,
`/root/task1_head_arch/` path requirement, etc).

**Before running the full 20-job sweep**: re-measure real VRAM/step-time on
mk8 itself (`measure_task3_vram.py`, included in this handoff, adjust
BUDGETS_MB to 32_000) -- the batch_size=2/jobs_per_gpu=2 defaults below are a
CONSERVATIVE starting point projected from a DIFFERENT GPU (RTX 6000 Ada,
49GB card, measured batch=2 -> ~12.7GB reserved for this exact model+shape),
not a real Blackwell measurement. Blackwell's bf16 tensor core / allocator
behavior may differ meaningfully from Ada's -- do not skip this step, same
principle as Step 2 of the Task 1 head-arch experiment on the source machine
overturning ITS OWN spec's back-of-envelope VRAM guess.

Recipe (LR/warmup/epochs) is Task 1's empirically-used starting point, NOT
independently validated for regression convergence -- watch val/MSE curves
on the first couple of folds before trusting all 20 jobs' results blindly.
"""
import sys

sys.path.insert(0, "/root")
from asparagus_orchestrate_common import run_orchestrator

SCRATCH = "/root/fomo26_task3_sweep_scratch"
DATA_PATH = "/root/asparagus_data/REGR002_FOMO26_BrainAge_iso1mm"
CHECKPOINT = "/root/FOMO26/expr/pretraining/ver5/checkpoints/step_280000.pt"
TEACHER = "brats"  # same default as Task 1 -- NOT specifically validated for a generic (non-lesion) regression task, worth an ablation later if GPU time allows

HEAD_MODES = ["base", "multistage_gem"]


def build_jobs():
    jobs = []
    # head_mode-major order (mirrors Task 1's arm-major convention): one
    # head_mode's 10 folds finish before the next starts, so a complete
    # pooled-10-fold MAE for "base" is usable as soon as its jobs land,
    # without waiting on all 20.
    for head_mode in HEAD_MODES:
        for fold in range(10):
            jobs.append({"name": f"{head_mode}_fold{fold}", "fold": fold, "head_mode": head_mode})
    return jobs


def build_cmd(job, gpu):
    return (
        "source /root/asparagus_env.sh && cd /root/asparagus && "
        f"CUDA_VISIBLE_DEVICES={gpu} /root/asparagus_venv/bin/asp_finetune_reg "
        "task=REGR002_FOMO26_BrainAge +model=fomo26_student_clsreg "
        f"data.data_path={DATA_PATH} "
        f"train_split_path={DATA_PATH}/split_stratified10.json "
        f"test_split_path={DATA_PATH}/TEST_stratified10_fold{job['fold']}.json "
        f"model.checkpoint_path={CHECKPOINT} "
        f"model.teacher_name={TEACHER} model.from_scratch=false "
        "model.stem_mode=frozen "
        f"model.head_mode={job['head_mode']} "
        "model.fusion_mode=early "
        "model.finetune_lr=1e-4 "
        "lightning.lightning_module=FomoRegressionModule "
        "data.train_split=split_stratified10 "
        f"data.test_split=TEST_stratified10_fold{job['fold']} "
        f"data.fold={job['fold']} "
        "hardware.num_workers=4 hardware.compile_mode=null "
        "training.batch_size=2 training.epochs=25 training.warmup_epochs=3 "
        "training.target_size=[176,256,256] "
        "transforms.normalize=false "
        "training.check_val_every_n_epoch=1 "
        "logger.wandb_logging=false "
        "root=task3_head_arch"
    )


if __name__ == "__main__":
    # num_gpus=2 (mk8's actual GPU count), jobs_per_gpu=2 as a starting point
    # (32GB budget / ~13-15GB per job projected, leaves headroom -- but see
    # the module docstring: RE-MEASURE on mk8 before trusting this).
    # min_available_mb / stall_timeout_s / launch_stagger_s copied from Task 1's
    # working values -- num_workers/concurrency ceiling is CPU/RAM-bound, not
    # GPU-bound, and mk8's core count/RAM is unknown from here, so this also
    # needs re-verification (see TASK3_HANDOFF.md Sec 4.4-equivalent).
    run_orchestrator(
        build_jobs(), build_cmd, SCRATCH,
        num_gpus=2, jobs_per_gpu=2, min_available_mb=20_000,
        stall_timeout_s=1200, launch_stagger_s=30,
    )
