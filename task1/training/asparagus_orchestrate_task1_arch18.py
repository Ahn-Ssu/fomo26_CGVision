"""Task 1 (CLS002_FOMO26_Infarct) 18-way architecture screen -- continuing
work started in another session (Codex) at
asparagus/modules/networks/task1_arch18.py (FomoTask1Arch18Net) +
configs/model/fomo26_task1_arch18.yaml. 3 backbone/fusion arms x 2
feature_source x 3 pooling_mode = 18 configs x 10 folds = 180 jobs.

  modality_arch (backbone/fusion axis):
    early_learnable      -- single 4-channel backbone, channel-repeated +
                             learnable stem (same as the original Task 1
                             pretrained+learnable-stem baseline, 0.707 AUROC)
    shared_modalitywise   -- single 1-channel backbone, frozen native stem,
                             called once per modality (weight-shared across
                             all 4 modalities, one Convpass copy trained)
    modality_specific     -- 4 INDEPENDENT 1-channel backbones (same init,
                             frozen native stem), each with its OWN
                             independently-trained Convpass copy
  feature_source: encoder_output (last stage only) | multistage (stages 1-5)
  pooling_mode: avg | max | gem (learnable signed-GeM exponent, per-stage if multistage)

Two real bugs were found+fixed before this could run at all (2026-08-15):
  1. fomo26_task1_arch18.yaml had inert `\\${model.*}` (backslash-escaped)
     interpolations, and separately pulled in a `core/fomo26_student@`
     default whose own _cls_net block references model.stem_mode/head_mode/
     fusion_mode -- keys that don't exist in this yaml, which would fail to
     resolve. Fixed by de-escaping and making the yaml self-contained.
  2. task1_arch18.py imported `from networks.student import ...` before
     `/root/FOMO26` was ever added to sys.path (that insert only happened as
     a side effect of a LATER import in the same file) -- worked by accident
     in a standalone verification script that pre-seeded sys.path itself,
     but failed under Hydra's instantiate(). Fixed by adding the sys.path
     insert at the top of the file, unconditionally.
  3. forward()'s shared_modalitywise branch iterated `enumerate(self.backbones)`
     directly, but that list has only ONE entry for shared_modalitywise
     (reused across modalities) -- silently dropped 3 of 4 modalities and
     then crashed on a channel-count mismatch. Fixed to iterate by modality
     index and reuse backbones[0] when len==1.
  4. FomoTask1Arch18Net never set `self.num_classes`, which
     ClassificationModule/RegressionModule require (`clsreg_module.py:59`,
     `self.num_classes = model.num_classes`) -- AttributeError at lightning
     module construction. Fixed by setting it in __init__.
See /root/task1_head_arch/verify_arch18.py (config resolution + trainable-
param audit + forward/backward smoke, all 18 configs) and the real 5-epoch
GPU smoke test (root=arch18_smoketest, modality_specific/multistage/gem --
the heaviest combo, ~42.5GB/49GB at batch=2) for how these were caught.

Recipe (user-specified): LR=1e-4, 5 epochs, 250 iters/epoch (config default,
unchanged), warmup_ratio=0.1 (-> exactly 125 warmup steps), batch_size=2,
target_size=[192,224,192] (same iso1mm data as the original Task 1
experiments), mild spatial augmentation (transforms.cpu_tr_transforms=
CPU_clsreg_train_transforms_mild_spatial).

model.ckpt_every_n_epoch=1 (NOT the config default of 25) -- required for
the "epoch checkpoint / epoch OOF inference" requirement: with the default,
a 5-epoch run would never trigger the per-epoch `ModelCheckpoint(every_n_epochs=...,
save_top_k=-1, filename="epoch={epoch:02d}")` callback at all (5 < 25), only
ever producing `best.ckpt`. This was verified empirically, not assumed --
see verify step 4 in this experiment's notes.
"""
import itertools
import sys

sys.path.insert(0, "/root")
from asparagus_orchestrate_common import run_orchestrator

SCRATCH = "/tmp/claude-0/-root/8e19c03d-842d-49c8-ac10-4c5f17622188/scratchpad/task1_arch18_sweep"
DATA_PATH = "/root/asparagus_data/CLS002_FOMO26_Infarct_iso1mm"
CHECKPOINT = "/root/FOMO26/expr/pretraining/ver5/checkpoints/step_280000.pt"
TEACHER = "brats"

MODALITY_ARCHES = ["early_learnable", "shared_modalitywise", "modality_specific"]
FEATURE_SOURCES = ["encoder_output", "multistage"]
POOLING_MODES = ["avg", "max", "gem"]


def build_jobs():
    jobs = []
    # config-major order: each of the 18 configs' 10 folds finish before the
    # next config starts (same "arm-major" convention as the earlier Task 1
    # head-arch sweep) -- a complete pooled-10-fold result per config is
    # usable as soon as its jobs land, without waiting on all 18.
    for modality_arch, feature_source, pooling_mode in itertools.product(
            MODALITY_ARCHES, FEATURE_SOURCES, POOLING_MODES):
        cfg_name = f"{modality_arch}_{feature_source}_{pooling_mode}"
        for fold in range(10):
            jobs.append({
                "name": f"{cfg_name}_fold{fold}", "fold": fold, "cfg_name": cfg_name,
                "modality_arch": modality_arch, "feature_source": feature_source,
                "pooling_mode": pooling_mode,
            })
    return jobs


def build_cmd(job, gpu):
    return (
        "source /root/asparagus_env.sh && cd /root/asparagus && "
        f"CUDA_VISIBLE_DEVICES={gpu} /root/asparagus_venv/bin/asp_finetune_cls "
        "task=CLS002_FOMO26_Infarct +model=fomo26_task1_arch18 "
        f"data.data_path={DATA_PATH} "
        f"train_split_path={DATA_PATH}/split_stratified10.json "
        f"test_split_path={DATA_PATH}/TEST_stratified10_fold{job['fold']}.json "
        f"model.checkpoint_path={CHECKPOINT} model.teacher_name={TEACHER} model.from_scratch=false "
        f"model.modality_arch={job['modality_arch']} "
        f"model.feature_source={job['feature_source']} "
        f"model.pooling_mode={job['pooling_mode']} "
        "model.ckpt_every_n_epoch=1 "
        "lightning.lightning_module=FomoClassificationModule "
        "data.train_split=split_stratified10 "
        f"data.test_split=TEST_stratified10_fold{job['fold']} "
        f"data.fold={job['fold']} "
        "hardware.num_workers=4 hardware.compile_mode=null "
        "training.batch_size=2 training.epochs=5 training.warmup_ratio=0.1 "
        "training.target_size=[192,224,192] "
        "transforms.normalize=false "
        "transforms.cpu_tr_transforms=CPU_clsreg_train_transforms_mild_spatial "
        "training.check_val_every_n_epoch=1 "
        "logger.wandb_logging=false "
        "root=arch18"
    )


if __name__ == "__main__":
    # jobs_per_gpu=1: the heaviest config (modality_specific/multistage/gem)
    # measured ~42.5GB/49GB at batch=2 -- no headroom for a second concurrent
    # job on the same GPU even for the lighter configs (not re-measured
    # per-config; 1 job/GPU is the safe uniform choice, matching the earlier
    # Task 1 head-arch sweep's same reasoning).
    run_orchestrator(
        build_jobs(), build_cmd, SCRATCH,
        num_gpus=4, jobs_per_gpu=1, min_available_mb=40_000,
        stall_timeout_s=1500, launch_stagger_s=30,
    )
