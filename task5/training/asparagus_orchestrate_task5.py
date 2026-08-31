"""Task5 (PMG binary classification, T1-only) sweep: 6 dataset variants
(hdbet_only/rigid/affine x 0.7mm/1.0mm) x 5 folds = 30 jobs, using the
arch18 sweep's winning architecture (modality_specific + multistage + gem --
functionally identical to shared_modalitywise at n_modalities=1, kept as
modality_specific to match the literal winning config) and the SAME
training recipe as Task1's arch18 sweep (checkpoint, teacher, LR/epochs/
batch/warmup/augmentation), per user instruction 2026-08-18. See
task5_preprocess_pt.py for how the 6 asparagus_data/Task5_PMG_{cond}_{sp}/
roots (.pt tensors, dataset.json, split_stratified5.json,
TEST_stratified5_fold{k}.json) were built, and task5_make_splits.py for the
scanner x class x in-plane-bin stratified 5-fold assignment reused
identically across all 6 variants for a paired comparison.

target_size varies per variant (each already padded to a multiple of 32,
see task5_pad_to_32.py) -- NOT a single fixed shape like Task1's arch18
sweep, so build_cmd looks it up per job.

epochs=20 (2026-08-20 rerun, root=task5_aug_ep20_sweep, NEW root -- distinct
from the 10-epoch v2 run): the user wants to see the full 6-variant
comparison at a schedule length comparable to the real Task1 submissions
(10/20 epochs) before deciding which Task5 variant to submit first --
mirrors the same reasoning behind the Task1 HD-BET+rigid experiment's
5ep->20ep rerun. min_gpu_free_mb=44_000 added since this coexists with that
Task1 20-epoch rerun (currently running on the same 4 GPUs) -- both
orchestrators now check real GPU memory before launching, avoiding the
2026-08-19 collision (that time, the OLD Task5 v2 orchestrator had no such
gate and blindly launched onto a GPU the new Task1 orchestrator was already
using).
"""
import sys

sys.path.insert(0, "/root")
from asparagus_orchestrate_common import run_orchestrator

SCRATCH = "/tmp/claude-0/-root/8e19c03d-842d-49c8-ac10-4c5f17622188/scratchpad/task5_sweep_ep20"
ASPARAGUS_DATA = "/root/asparagus_data"
CHECKPOINT = "/root/FOMO26/expr/pretraining/ver5/checkpoints/step_280000.pt"
TEACHER = "brats"

MODALITY_ARCH = "modality_specific"
FEATURE_SOURCE = "multistage"
POOLING_MODE = "gem"

VARIANTS = {
    "hdbet_only_iso07": [288, 320, 288],
    "hdbet_only_iso10": [192, 224, 224],
    "rigid_iso07": [256, 288, 256],
    "rigid_iso10": [192, 224, 192],
    "affine_iso07": [256, 288, 256],
    "affine_iso10": [192, 224, 192],
}
N_FOLDS = 5


def build_jobs():
    jobs = []
    # variant-major order: a complete pooled-5-fold result per variant is
    # usable as soon as its 5 jobs land, without waiting on all 6.
    for variant, target_size in VARIANTS.items():
        for fold in range(N_FOLDS):
            jobs.append({
                "name": f"{variant}_fold{fold}", "fold": fold,
                "variant": variant, "target_size": target_size,
            })
    return jobs


def build_cmd(job, gpu):
    data_path = f"{ASPARAGUS_DATA}/Task5_PMG_{job['variant']}"
    target_size_str = "[" + ",".join(str(x) for x in job["target_size"]) + "]"
    return (
        "source /root/asparagus_env.sh && cd /root/asparagus && "
        f"CUDA_VISIBLE_DEVICES={gpu} /root/asparagus_venv/bin/asp_finetune_cls "
        f"task=Task5_PMG_{job['variant']} +model=fomo26_task1_arch18 "
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
        "training.batch_size=2 training.epochs=20 training.warmup_ratio=0.1 "
        f"training.target_size={target_size_str} "
        "transforms.normalize=false "
        "transforms.cpu_tr_transforms=CPU_clsreg_train_transforms_mild_spatial_intensity "
        "training.check_val_every_n_epoch=1 "
        "logger.wandb_logging=false "
        "root=task5_aug_ep20_sweep"
    )


if __name__ == "__main__":
    run_orchestrator(
        build_jobs(), build_cmd, SCRATCH,
        num_gpus=4, jobs_per_gpu=1, min_available_mb=40_000,
        min_gpu_free_mb=44_000,
        stall_timeout_s=1500, launch_stagger_s=30,
    )
