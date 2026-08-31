"""Task5 LV-targeted masking augmentation experiment (2026-08-29).

Motivation: LR-asymmetry group analysis + SmoothGrad saliency both
independently converged on the same finding -- the rigid/affine Task5
classifiers' attention/discriminative signal concentrates around the
LATERAL VENTRICLE boundary, not the cortex (where the actual PMG pathology
-- a gray-matter cortical folding malformation -- lives). Real leaderboard
AUROC for rigid (0.398) and affine (0.396) is near/below random despite
locally-decent held-out AUROC (~0.86), consistent with the model latching
onto ventricle shape/size/asymmetry as an easy but non-generalizing
shortcut. Per-subject quantitative comparison (Welch/Mann-Whitney, n=24
PMG vs 24 control per variant) found NO significant PMG-vs-control
asymmetry difference either inside or outside a data-driven ventricle
mask, meaning whatever the model is keying on inside the ventricle region
is not even a robust population-level signal -- likely per-subject noise
that happens to correlate with train-fold labels but doesn't transfer.

Two candidate fixes, both gated behind a fixed per-variant LV+
periventricular bounding box (asparagus/modules/transforms/presets/train.py
Torch_LVRandomCutout / Torch_LVFixedMask, _LV_BBOX_RIGID / _LV_BBOX_AFFINE
-- derived from a 48-subject population CSF-consistency ventricle mask,
eroded to drop the thin 3rd-ventricle/interhemispheric-fissure sliver, then
expanded by a 15-voxel margin to also cover immediately surrounding tissue
per user request, "LV와 그 주변부의 일그러짐으로 학습되는 것을 방지"):

  - lv_cutout:    every sample, w.p. 0.8, 1-3 random-sized (40-90% of each
                  bbox axis -> per-box volume ~6-73% of the bbox) boxes at
                  random positions inside the bbox are corrupted with
                  locally-toned noise. Stochastic, different every epoch
                  (empirically ~15-58% of the bbox corrupted per draw).
  - lv_fixedmask: the ENTIRE bbox is corrupted every sample, every epoch --
                  deterministic full denial of the ventricle region, to
                  test whether that more aggressively forces reliance on
                  the cortex (at the risk of also discarding whatever small
                  amount of real signal might exist near the ventricle).

Only rigid/affine have a shared registration template the fixed bbox
coordinates are valid in -- hdbet_only has no common space, so it's
excluded from this experiment.

epochs=10 matches the existing hdbet_only_encoder_output comparison run
(root=task5_hdbet_only_encoder_output_ep10) and the original v2 baselines,
for a directly comparable pooled-AUROC readout without waiting on a full
20-epoch schedule.
"""
import sys

sys.path.insert(0, "/root")
from asparagus_orchestrate_common import run_orchestrator

SCRATCH = "/tmp/claude-0/-root/8e19c03d-842d-49c8-ac10-4c5f17622188/scratchpad/task5_lv_augment"
ASPARAGUS_DATA = "/root/asparagus_data"
CHECKPOINT = "/root/FOMO26/expr/pretraining/ver5/checkpoints/step_280000.pt"
TEACHER = "brats"

MODALITY_ARCH = "modality_specific"
FEATURE_SOURCE = "multistage"
POOLING_MODE = "gem"
TARGET_SIZE = [192, 224, 192]  # both rigid_iso10 and affine_iso10
N_FOLDS = 5

CONDITIONS = {
    "rigid_lv_cutout": ("rigid_iso10", "CPU_clsreg_train_transforms_lv_cutout_rigid"),
    "rigid_lv_fixedmask": ("rigid_iso10", "CPU_clsreg_train_transforms_lv_fixedmask_rigid"),
    "affine_lv_cutout": ("affine_iso10", "CPU_clsreg_train_transforms_lv_cutout_affine"),
    "affine_lv_fixedmask": ("affine_iso10", "CPU_clsreg_train_transforms_lv_fixedmask_affine"),
}


def build_jobs():
    jobs = []
    for cond_name, (variant, tf_name) in CONDITIONS.items():
        for fold in range(N_FOLDS):
            jobs.append({
                "name": f"{cond_name}_fold{fold}", "fold": fold,
                "variant": variant, "cpu_tf": tf_name, "cond": cond_name,
            })
    return jobs


def build_cmd(job, gpu):
    data_path = f"{ASPARAGUS_DATA}/Task5_PMG_{job['variant']}"
    target_size_str = "[" + ",".join(str(x) for x in TARGET_SIZE) + "]"
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
        "training.batch_size=2 training.epochs=10 training.warmup_ratio=0.1 "
        f"training.target_size={target_size_str} "
        "transforms.normalize=false "
        f"transforms.cpu_tr_transforms={job['cpu_tf']} "
        "training.check_val_every_n_epoch=1 "
        "logger.wandb_logging=false "
        "root=task5_lv_augment_ep10"
    )


if __name__ == "__main__":
    run_orchestrator(
        build_jobs(), build_cmd, SCRATCH,
        num_gpus=4, jobs_per_gpu=1, min_available_mb=40_000,
        min_gpu_free_mb=44_000,
        stall_timeout_s=1500, launch_stagger_s=30,
    )
