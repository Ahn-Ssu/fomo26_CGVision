"""Task 4 (Multiclass Tissue Segmentation) final submission training --
5-fold CV x 2 pretrained teachers (brats, vesselfm) = 10 jobs, using
FomoStudentSegNet (configs/model/fomo26_student_seg.yaml) with the encoder
fully frozen (backbone + its own selected-teacher Convpass, via
encoder_convpass_frozen=true) and the decoder reinitialized + trained from
scratch (decoder_scheme=scratch) -- see
configs/projects/task4/TASK4_ROI05_PRETRAIN_ENCFROZEN_DECSCRATCH.yaml for
the full recipe (patch_size=128^3, batch_size=2, 100 epochs/250 steps-per-
epoch, warmup_epochs=10, bf16-mixed precision, full-volume validation with
8-way mirror TTA disabled during training/validation -- see
validation_mirror_tta=false -- 8-way mirror TTA is applied only at final
inference time, in submission/predict.py).

Model selection: for each of the 5 folds, whichever teacher (brats vs
vesselfm) scored the higher held-out val Dice was kept for the final
ensemble -- brats won folds 1/2/4, vesselfm won folds 0/3 (see
submission/checkpoints/manifest.json). anatomix+brains, vjepa, and voco
(the pretrain checkpoint's other 3 teacher slots) were not explored for
Task_4 given the brats/vesselfm results already available and the
challenge deadline.

What we tried and rejected (see task4/README.md "What we tried and
rejected" for the full writeup with numbers): a targeted augmentation-
strengthening retrain (boosting rotation/scale/gamma exposure probability
and widening the gamma-dark range specifically) measurably improved
robustness on those 3 targeted stress-test perturbations, but showed
~zero net average-Dice change and WORSE HD95/false-negative-collapse
behavior on several non-targeted perturbations (sharpening, bias field) --
not adopted for the final submission. A fixed gamma=0.5 brightening
preprocessing step (baked into training, not just TTA) was also tried and
found to give no measurable benefit. Both experiments used this same
orchestration pattern with a different `+projects/task4=` config target.
"""
import sys

sys.path.insert(0, "/root")
from asparagus_orchestrate_common import run_orchestrator

SCRATCH = "/tmp/task4_5fold_sweep"
CHECKPOINT = "/root/FOMO26/expr/pretraining/ver5/checkpoints/step_280000.pt"
TEACHERS = ["brats", "vesselfm"]
N_FOLDS = 5


def build_jobs():
    return [
        {"name": f"task4_{teacher}_fold{fold}", "teacher": teacher, "fold": fold}
        for teacher in TEACHERS
        for fold in range(N_FOLDS)
    ]


def build_cmd(job, gpu):
    return (
        "source /root/asparagus_env.sh && cd /root/asparagus && "
        f"CUDA_VISIBLE_DEVICES={gpu} /root/asparagus_venv/bin/asp_finetune_seg "
        "+projects/task4=TASK4_ROI05_PRETRAIN_ENCFROZEN_DECSCRATCH "
        f"model.checkpoint_path={CHECKPOINT} model.teacher_name={job['teacher']} "
        f"data.fold={job['fold']} data.test_split=TEST_task4_5fold_fold{job['fold']} "
        "training.resume_from_last=false "
        "logger.wandb_logging=false"
    )


if __name__ == "__main__":
    run_orchestrator(
        build_jobs(), build_cmd, SCRATCH,
        num_gpus=4, jobs_per_gpu=1, min_available_mb=40_000,
        stall_timeout_s=900, launch_stagger_s=30,
    )
