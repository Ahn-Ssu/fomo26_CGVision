"""Extracts the 5-fold affine_iso10 + LV-fixedmask (30-epoch,
CPU_clsreg_train_transforms_lv_fixedmask_affine) checkpoints -- the final
candidate after the 2026-08-30 LV-masking investigation (see
/root/fomo26task5/lr_asymmetry_analysis/ for the analysis that motivated
this: LR-asymmetry + SmoothGrad saliency both showed the un-masked affine
classifier's attention concentrated on the lateral-ventricle boundary
rather than the cortex, and the ventricle region was NOT even a
statistically significant group-level discriminator -- consistent with a
non-generalizing shortcut, which tracks the real leaderboard score for
this variant being near/below random (0.396) despite a locally-strong
0.83 pooled AUROC).

Uses the FIXED epoch=29 (30/30) checkpoint, not best.ckpt -- unlike the
unmasked affine_iso10 v3 container (which had a documented, data-driven
exception for using best.ckpt), the LV-masked recipe showed NO overfitting
across 10->20->30 epochs (masked-test AUROC climbed 0.83->0.88->0.90
monotonically), so the last-epoch checkpoint is both the best-performing
AND the principled no-cherry-picking default.
"""
import glob
import os

import torch

DOWNSTREAM_ROOT = "/root/FOMO26/expr/downstream_finetune"
ASPARAGUS_DATA = "/root/asparagus_data"
CHECKPOINT = "/root/FOMO26/expr/pretraining/ver5/checkpoints/step_280000.pt"
TEACHER = "brats"
TRANSFORM_PRESET = "CPU_clsreg_train_transforms_lv_fixedmask_affine"
VARIANT = "affine_iso10"
ROOT_NAME = "task5_lv_fixedmask_ep30"
EPOCH = 29
OUT_DIR = "/root/task5_submission_affine_lvmask_final/model/fold_checkpoints"
N_FOLDS = 5


def clargs_string(fold):
    data_path = f"{ASPARAGUS_DATA}/Task5_PMG_{VARIANT}"
    return (
        f"hardware.compile_mode=null,model.checkpoint_path={CHECKPOINT},"
        f"model.ckpt_every_n_epoch=1,model.feature_source=multistage,"
        f"model.from_scratch=false,model.modality_arch=modality_specific,"
        f"model.pooling_mode=gem,model.teacher_name={TEACHER},"
        f"test_split_path={data_path}/TEST_stratified5_fold{fold}.json,"
        f"train_split_path={data_path}/split_stratified5.json,"
        f"training.warmup_ratio=0.1,"
        f"transforms.cpu_tr_transforms={TRANSFORM_PRESET}"
    )


def find_fold_run_dir(fold):
    fixed = os.path.join(
        DOWNSTREAM_ROOT, f"Task5_PMG_{VARIANT}", "fomo26_task1_arch18__3D",
        "script=finetune_cls", f"root={ROOT_NAME}__stem=None_last.ckpt",
        f"leaf=default_finetune_cls__clargs={clargs_string(fold)}",
        f"split_stratified5__fold={fold}",
    )
    pattern = os.path.join(glob.escape(fixed), "run_id=*")
    matches = [m for m in glob.glob(pattern)
               if os.path.exists(os.path.join(m, "checkpoints", f"epoch={EPOCH:02d}.ckpt"))]
    assert len(matches) == 1, (fold, matches, fixed)
    return matches[0]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for fold in range(N_FOLDS):
        run_dir = find_fold_run_dir(fold)
        ckpt_path = os.path.join(run_dir, "checkpoints", f"epoch={EPOCH:02d}.ckpt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        dst = os.path.join(OUT_DIR, f"fold{fold}.pt")
        torch.save(sd, dst)
        size_mb = os.path.getsize(dst) / 1e6
        print(f"fold {fold}: {run_dir} -> {dst} ({size_mb:.1f} MB, {len(sd)} tensors)")


if __name__ == "__main__":
    main()
