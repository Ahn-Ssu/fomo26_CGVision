"""Champion recipe stabilization ablation (2026-08-17): does EMA of the
trainable weights (only the small PEFT Convpass adapters + head; backbone is
frozen) reduce the late-training plateau fluctuation seen in the arch18 v2
champion sweep (modality_specific_multistage_gem swung ~0.85-0.91 pooled
AUROC across steps 1500-2500), without needing a separate large sweep?

Per the user's framing: the champion 3-seed replicate (see
asparagus_orchestrate_task1_champion_seeds.py /
asparagus_aggregate_task1_champion_ensemble.py) showed BETWEEN-run (seed)
variance is small (0.856-0.885). The remaining, larger source of instability
is WITHIN-run checkpoint variance across the late-training plateau. EMA
should target that, not the seed variance.

Design: a single training run per fold (10 folds, 1 seed -- this is not a
seed study) with `model.ema_ablation=true`. The new EmaAblationCallback
(asparagus/modules/callbacks/ema_ablation.py) maintains 3 EMA shadow copies
of the trainable weights (beta=0.99/0.995/0.997) updated after every
optimizer step, and evaluates raw + all 3 EMA variants against the fold's
held-out subjects every 100 steps (aligned with the existing
ckpt_every_n_steps cadence), writing per-subject probabilities to
predictions/ema_ablation_by_step.csv inside each fold's run dir. Zero
additional GPU training cost -- only extra (cheap) forward passes on ~2-3
held-out subjects every 100 steps, since only ~312K trainable parameters
need EMA-tracking (backbone excluded).

Same locked training contract as the champion seed sweep: modality_specific
+ multistage + gem, frozen backbone, mild spatial aug, LR 1e-4, 10
epochs/2500 steps, warmup_ratio=0.1. Uses training.seed=100003 (a new fixed
seed, distinct from the champion seed sweep's 100001/100002 and v2's
uncontrolled seed1) so this run is itself reproducible; seed is NOT the
variable under study here.

root=arch18_champion_ema keeps this separate from arch18_v2 and
arch18_champion_seed{2,3}.
"""
import sys

sys.path.insert(0, "/root")
from asparagus_orchestrate_common import run_orchestrator

SCRATCH = "/tmp/claude-0/-root/8e19c03d-842d-49c8-ac10-4c5f17622188/scratchpad/task1_champion_ema_sweep"
DATA_PATH = "/root/asparagus_data/CLS002_FOMO26_Infarct_iso1mm"
CHECKPOINT = "/root/FOMO26/expr/pretraining/ver5/checkpoints/step_280000.pt"
TEACHER = "brats"
SEED_VALUE = 100003


def build_jobs():
    return [{"name": f"champion_ema_fold{fold}", "fold": fold} for fold in range(10)]


def build_cmd(job, gpu):
    return (
        "source /root/asparagus_env.sh && cd /root/asparagus && "
        f"CUDA_VISIBLE_DEVICES={gpu} /root/asparagus_venv/bin/asp_finetune_cls "
        "task=CLS002_FOMO26_Infarct +model=fomo26_task1_arch18 "
        f"data.data_path={DATA_PATH} "
        f"train_split_path={DATA_PATH}/split_stratified10.json "
        f"test_split_path={DATA_PATH}/TEST_stratified10_fold{job['fold']}.json "
        f"model.checkpoint_path={CHECKPOINT} model.teacher_name={TEACHER} model.from_scratch=false "
        "model.modality_arch=modality_specific model.feature_source=multistage model.pooling_mode=gem "
        "model.ckpt_every_n_steps=100 model.ema_ablation=true "
        f"training.seed={SEED_VALUE} "
        "lightning.lightning_module=FomoClassificationModule "
        "data.train_split=split_stratified10 "
        f"data.test_split=TEST_stratified10_fold{job['fold']} "
        f"data.fold={job['fold']} "
        "hardware.num_workers=4 hardware.compile_mode=null "
        "training.batch_size=2 training.epochs=10 training.warmup_ratio=0.1 "
        "training.target_size=[192,224,192] "
        "transforms.normalize=false "
        "transforms.cpu_tr_transforms=CPU_clsreg_train_transforms_mild_spatial "
        "training.check_val_every_n_epoch=1 "
        "logger.wandb_logging=false "
        "root=arch18_champion_ema"
    )


if __name__ == "__main__":
    run_orchestrator(
        build_jobs(), build_cmd, SCRATCH,
        num_gpus=4, jobs_per_gpu=1, min_available_mb=40_000,
        stall_timeout_s=480, launch_stagger_s=60,
    )
