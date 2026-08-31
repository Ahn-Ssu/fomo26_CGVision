"""affine_lv_fixedmask (final candidate) extended to 30 epochs (2026-08-30).

Follow-up to the ep20 result: 20ep clearly beat 10ep (masked-test 0.8819
vs 0.8316/0.8681), and unlike the unmasked baseline (which overfits past
10ep, 0.8299->0.8142), this masked recipe shows no sign of overfitting
yet. Testing whether it keeps improving at 30ep.

Deliberately launched with num_gpus=3 and only folds [0,1,2] in this
first call -- GPU3 is reserved for a parallel augmentation-sensitivity
TTA investigation using the already-completed ep20 5-fold models. Folds
[3,4] are queued separately once that TTA work frees GPU3 up (see
asparagus_orchestrate_task5_lv_fixedmask_ep30_remaining.py).
"""
import sys

sys.path.insert(0, "/root")
from asparagus_orchestrate_common import run_orchestrator

SCRATCH = "/tmp/claude-0/-root/8e19c03d-842d-49c8-ac10-4c5f17622188/scratchpad/task5_lv_fixedmask_ep30"
ASPARAGUS_DATA = "/root/asparagus_data"
CHECKPOINT = "/root/FOMO26/expr/pretraining/ver5/checkpoints/step_280000.pt"
TEACHER = "brats"

MODALITY_ARCH = "modality_specific"
FEATURE_SOURCE = "multistage"
POOLING_MODE = "gem"
TARGET_SIZE = [192, 224, 192]
VARIANT = "affine_iso10"
CPU_TF = "CPU_clsreg_train_transforms_lv_fixedmask_affine"
ROOT_NAME = "task5_lv_fixedmask_ep30"
FOLDS = [0, 1, 2]  # remaining [3,4] launched separately after TTA work frees gpu3


def build_jobs():
    return [{"name": f"affine_lv_fixedmask_ep30_fold{fold}", "fold": fold} for fold in FOLDS]


def build_cmd(job, gpu):
    data_path = f"{ASPARAGUS_DATA}/Task5_PMG_{VARIANT}"
    target_size_str = "[" + ",".join(str(x) for x in TARGET_SIZE) + "]"
    return (
        "source /root/asparagus_env.sh && cd /root/asparagus && "
        f"CUDA_VISIBLE_DEVICES={gpu} /root/asparagus_venv/bin/asp_finetune_cls "
        f"task=Task5_PMG_{VARIANT} +model=fomo26_task1_arch18 "
        f"data.data_path={data_path} "
        f"train_split_path={data_path}/split_stratified5.json "
        f"test_split_path={data_path}/TEST_stratified5_fold{job['fold']}.json "
        f"model.checkpoint_path={CHECKPOINT} model.teacher_name={TEACHER} model.from_scratch=false "
        f"model.modality_arch={MODALITY_ARCH} "
        f"model.feature_source={FEATURE_SOURCE} "
        f"model.pooling_mode={POOLING_MODE} "
        "model.ckpt_every_n_epoch=1 "
        "lightning.lightning_module=FomoClassificationModule "
        "data.train_split=split_stratified5 "
        f"data.test_split=TEST_stratified5_fold{job['fold']} "
        f"data.fold={job['fold']} "
        "hardware.num_workers=4 hardware.compile_mode=null "
        "training.batch_size=2 training.epochs=30 training.warmup_ratio=0.1 "
        f"training.target_size={target_size_str} "
        "transforms.normalize=false "
        f"transforms.cpu_tr_transforms={CPU_TF} "
        "training.check_val_every_n_epoch=1 "
        "logger.wandb_logging=false "
        f"root={ROOT_NAME}"
    )


if __name__ == "__main__":
    run_orchestrator(
        build_jobs(), build_cmd, SCRATCH,
        num_gpus=3, jobs_per_gpu=1, min_available_mb=40_000,
        min_gpu_free_mb=44_000,
        stall_timeout_s=1500, launch_stagger_s=30,
    )
