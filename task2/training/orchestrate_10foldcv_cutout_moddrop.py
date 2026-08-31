"""Cutout + modality-dropout ablation (user request 2026-08-21): tests
whether reducing the model's learned dependency on any single spatial
region (3D Cutout) and specifically the 3rd modality channel (swi_or_t2s,
ModalityDropout) helps generalization -- see asparagus/modules/transforms/
cutout.py and modality_dropout.py for the transform implementations, and
SEG024_LOO13_SPATIALINTENSITY_LRMIRROR_CUTOUT_MODDROP_LR1E4.yaml for the
config (identical to AUGEF/SEG021 except gpu_tr_transforms adds these two
regularizers on top of the existing mild intensity augmentation).

10 jobs, all 4 GPUs, standard 10-fold CV (seed=42 partition, 3-modality).
Seed block 291000+fold (new)."""
import sys

sys.path.insert(0, "/root/fomo26_segm_mk5")
from asparagus_orchestrate_common import run_orchestrator

SCRATCH = "/root/fomo26_segm_mk5/task2_archsearch/output/10foldcv_cutout_moddrop_orchestrator_logs"
CHECKPOINT = "/root/FOMO26/expr/pretraining/ver5/checkpoints/step_280000.pt"
SEED_BASE = 291000
GPU_IDS = [0, 1, 2, 3]
EPOCHS = 60

STEM_MODE = "learnable"
ENCODER_CONVPASS_FROZEN = True
DECODER_SCHEME = "finetune"
N_FOLDS = 10


def build_jobs():
    return [
        {"name": f"10foldcv_cutout_moddrop_fold{fold}", "fold": fold, "seed": SEED_BASE + fold}
        for fold in range(N_FOLDS)
    ]


def build_cmd(job, gpu):
    return (
        "source /root/fomo26_segm_mk5/asparagus_env.sh && cd /root/fomo26_segm_mk5 && "
        f"CUDA_VISIBLE_DEVICES={GPU_IDS[gpu]} /root/fomo26_segm_mk5/asparagus_venv/bin/asp_finetune_seg "
        "+projects/fomo26_seg=SEG024_LOO13_SPATIALINTENSITY_LRMIRROR_CUTOUT_MODDROP_LR1E4 "
        "+model=fomo26_student_seg "
        "model.finetune_lr=1e-4 "
        f"model.checkpoint_path={CHECKPOINT} model.teacher_name=brats "
        f"model.stem_mode={STEM_MODE} model.encoder_convpass_frozen={str(ENCODER_CONVPASS_FROZEN).lower()} "
        f"model.decoder_scheme={DECODER_SCHEME} "
        f"data.train_split=split_10foldcv_3mod data.fold={job['fold']} data.test_split=TEST_10foldcv_3mod_fold{job['fold']} "
        f"training.seed={job['seed']} training.epochs={EPOCHS} "
        "hardware.compile_mode=null hardware.num_workers=4 logger.wandb_logging=false"
    )


if __name__ == "__main__":
    jobs = build_jobs()
    print(f"total jobs: {len(jobs)} (10-fold CV, AUGEF recipe + Cutout3D + ModalityDropout(ch=2), GPUs {GPU_IDS})")
    run_orchestrator(
        jobs, build_cmd, SCRATCH,
        num_gpus=len(GPU_IDS), jobs_per_gpu=1, min_available_mb=20_000,
        stall_timeout_s=600, launch_stagger_s=15,
    )
