"""Candidate D: HD-BET + union-mask + ANTs rigid registration, LAST epoch
(epoch=19, since epochs=20 -> 0..19) ensemble across all 10 folds --
same "fixed end-of-budget checkpoint, never cherry-pick" principle as
candidates A/C, and matches the empirical finding that at 20 epochs
last-epoch (0.904) clearly beat best.ckpt (0.875) for this experiment
(see project_task1_hdbet_rigid_experiment.md).

root=hdbet_rigid_check_ep20 was used for BOTH the original folds 0-4 run
and the resumed folds 5-9 run (2026-08-20/21) -- glob.escape + latest-mtime
selection (same fix as the other extractor scripts) correctly picks each
fold's real completed run_id regardless of how many stall+requeue retries
created extra run_id dirs along the way.
"""
import glob
import os

import torch

DATA_PATH = "/root/asparagus_data/CLS002_FOMO26_Infarct_hdbet_rigid"
MODELS_ROOT = "/root/FOMO26/expr/downstream_finetune/CLS002_FOMO26_Infarct_hdbet_rigid/fomo26_task1_arch18__3D/script=finetune_cls"
CHECKPOINT = "/root/FOMO26/expr/pretraining/ver5/checkpoints/step_280000.pt"
TEACHER = "brats"
OUT_DIR = "/root/task1_submission_d/model/fold_checkpoints"
N_FOLDS = 10
LAST_EPOCH = 19  # epochs=20 -> 0..19


def clargs_string(fold):
    return (
        f"hardware.compile_mode=null,model.checkpoint_path={CHECKPOINT},"
        f"model.ckpt_every_n_epoch=1,model.feature_source=multistage,"
        f"model.from_scratch=false,model.modality_arch=modality_specific,"
        f"model.pooling_mode=gem,model.teacher_name={TEACHER},"
        f"test_split_path={DATA_PATH}/TEST_stratified10_fold{fold}.json,"
        f"train_split_path={DATA_PATH}/split_stratified10.json,"
        f"training.warmup_ratio=0.1,"
        f"transforms.cpu_tr_transforms=CPU_clsreg_train_transforms_mild_spatial"
    )


def find_fold_run_dir(fold):
    fixed = os.path.join(
        MODELS_ROOT, "root=hdbet_rigid_check_ep20__stem=None_last.ckpt",
        f"leaf=default_finetune_cls__clargs={clargs_string(fold)}",
        f"split_stratified10__fold={fold}",
    )
    pattern = os.path.join(glob.escape(fixed), "run_id=*")
    matches = [m for m in glob.glob(pattern)
               if os.path.exists(os.path.join(m, "checkpoints", f"epoch={LAST_EPOCH:02d}.ckpt"))]
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(os.path.join(p, "checkpoints", f"epoch={LAST_EPOCH:02d}.ckpt")))
    return matches[-1]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for fold in range(N_FOLDS):
        run_dir = find_fold_run_dir(fold)
        assert run_dir is not None, f"fold {fold}: no run dir with epoch={LAST_EPOCH:02d}.ckpt found"
        ckpt_path = os.path.join(run_dir, "checkpoints", f"epoch={LAST_EPOCH:02d}.ckpt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        # 2026-08-22: store in bf16 (halves on-disk size, 515MB->322MB
        # per fold) -- predict.py upcasts back to fp32 immediately after
        # loading, so the forward pass is unaffected; only the weight
        # values are bf16-rounded (verified negligible: ensemble prob
        # 0.9641 fp32 vs 0.9631 bf16-roundtrip on a real subject).
        sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
        dst = os.path.join(OUT_DIR, f"fold{fold}.pt")
        torch.save(sd, dst)
        size_mb = os.path.getsize(dst) / 1e6
        print(f"fold {fold}: {run_dir} -> {dst} ({size_mb:.1f} MB, {len(sd)} tensors)")


if __name__ == "__main__":
    main()
