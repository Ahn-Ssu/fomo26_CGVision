"""FOMO26 multi-teacher episodic distillation pretraining -- Phase 1.

Phase 1 scope (per user request 2026-07-15): a single student f_theta,
trained via feature-level distillation against ONE frozen teacher sampled
per step (episodic sampling), no LoRA / teacher-specific adapters yet (that
is planned as a Phase 2 upgrade on top of this).

Framework choice: `accelerate`, not asparagus. Reasoning (see
/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/02_TEACHER_REPORT.md and /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/01_ASPARAGUS_ANALYSIS.md for the underlying
analysis this decision is based on):
  - asparagus's own pretraining LightningModule (SelfSupervisedModule) is a
    single-model MSE-reconstruction trainer with no teacher concept; the
    closest precedent (DINOv2Module) is tightly coupled to a specific
    DINO/iBOT loss and a single EMA teacher-student pair, not a generic
    multi-heterogeneous-frozen-teacher pattern.
  - Our 3 live teachers (Anatomix, VesselFM, BraTS) are each defined in their
    own external repo/package with their own preprocessing -- wiring all of
    that through asparagus's Hydra config composition would be substantial
    extra engineering for Phase 1 with little payoff, versus a direct,
    fully-inspectable PyTorch + accelerate script.
  - asparagus is still the RIGHT tool for finetuning/eval (Task 1 finding)
    and that boundary is preserved here: this script's checkpoints are saved
    with "model."-prefixed keys inside a {"state_dict": ...} dict, exactly
    the contract asp_finetune_seg/cls/reg already expect
    (/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/01_ASPARAGUS_ANALYSIS.md Sec C.4) -- no shim needed to hand off a
    checkpoint from this script to `asp_finetune_*`.

Data source (updated 2026-07-16): defaults to FOMO300KPreprocessedDataset,
reading the raw-preserving .npz cache at /root/data/FOMO300K_preprocessed/
(full corpus preprocessed 2026-07-15, 165,790 scans, 1.27TB -- see
/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/03_PREPROCESSING_REPORT.md). Pass --data_source zip to fall back to the
original zip-reading FOMO300KDataset (~4x slower per sample, no preprocessing
step required -- useful for data not yet run through preprocess_raw.py).
The dataset now also returns an explicit foreground `mask` (Issue A-correct
for the preprocessed source), passed to every teacher via `meta={"mask":...}`
each step -- only BraTS's preprocess() currently uses it (z-score needs a
foreground mask), the others ignore it safely (BaseTeacher's meta contract).

Usage (4 GPUs):
    source /root/teachers/venv/bin/activate
    accelerate launch --multi_gpu --num_processes 4 /root/FOMO26/run_pretrain.py \\
        --run_name my_run --steps 10000 --batch_size 2 --patch_size 128

Single-GPU / CPU smoke test:
    python3 /root/FOMO26/run_pretrain.py --run_name smoke --steps 5 --limit 50
"""

import argparse
import csv
import json
import os
import sys
import time

import torch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from torch.utils.data import DataLoader

sys.path.insert(0, "/root")
sys.path.insert(0, "/root/FOMO26")

from FOMO26.data.fomo300k_dataset import (  # noqa: E402
    FOMO300KDataset,
    FOMO300KPreprocessedDataset,
    load_or_build_index,
    load_preprocessed_index,
    split_train_eval,
)
from FOMO26.data.vjepa_cache_dataset import VJEPACachedDataset  # noqa: E402
from FOMO26.losses.distillation import episodic_distillation_loss  # noqa: E402
from FOMO26.networks.projections import (  # noqa: E402
    VJEPA_FEATURE_SPECS, build_projection_heads, is_encoder_only_teacher, teacher_needs_encoder_features)
from FOMO26.networks.student import build_student  # noqa: E402
from FOMO26.sampler.episodic import EpisodicTeacherSampler  # noqa: E402
from FOMO26.teacher.registry import TEACHER_REGISTRY, get_teacher  # noqa: E402

EXPR_ROOT = "/root/FOMO26/expr"

# FastSurfer is intentionally excluded from live per-step teachers -- see
# FOMO26/teacher/FASTSURFER_EXCLUDED.md. All remaining registry entries are
# live 3D feature-distillation teachers.
#
# "anatomix+brains" (not plain "anatomix"), per 2026-07-17 request: Anatomix
# ships two checkpoints -- anatomix.pth (general, trained purely on synthetic
# label-derived volumes) and anatomix+brains.pth (general+brain, additionally
# synthesizes training volumes using real brain label maps -- see
# /root/anatomix/README.md). Since FOMO300K is all-brain MRI, the brain-aware
# variant is the better prior for this project. Both are registered in
# TEACHER_REGISTRY under separate keys ("anatomix" / "anatomix+brains") and
# share the same architecture/feature_specs/STAGE_MAP entry -- only the
# weights differ.
DEFAULT_TEACHERS = ["anatomix+brains", "vesselfm", "brats"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--teachers", type=str, nargs="+", default=DEFAULT_TEACHERS,
                    choices=list(TEACHER_REGISTRY.keys()))
    p.add_argument("--teacher_weights", type=str, default=None,
                    help='JSON dict of {teacher_name: weight} for episodic sampling; default uniform')
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=2, help="per-process batch size")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-2)
    p.add_argument("--mse_weight", type=float, default=0.1)
    p.add_argument("--mse_weight_overrides", type=str, default=None,
                    help="2026-08-02: JSON dict {teacher_name: mse_weight} overriding --mse_weight "
                         "for specific teachers. Motivation: the cosine term in "
                         "losses/distillation.py is scale-invariant by design (its own docstring: "
                         "'teacher/student feature magnitudes are not expected to match on an "
                         "absolute scale'), but the MSE term is NOT -- measured real per-voxel L2 "
                         "feature norms: anatomix~2.8-9.4, vesselfm~4.0-8.0, brats~1.4-3.6, "
                         "voco~0.7-2.6 (a pre-existing ~2-6x spread the cosine-dominant design "
                         "already tolerates), but vjepa~25-31 -- 5-40x larger than any CNN teacher, "
                         "enough that its raw MSE term could dominate the total loss on "
                         "vjepa-sampled steps until its projection head learns to scale up. Default "
                         "(no override) reproduces the exact prior single-mse_weight behavior for "
                         "every teacher -- this is purely an opt-in escape hatch, e.g. "
                         '\'{"vjepa": 0.02}\' to shrink just that term.')
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=500,
                    help="run evaluate() every N steps against the fixed monitoring holdout "
                         "(0 disables eval entirely)")
    p.add_argument("--eval_batch_size", type=int, default=4)
    # eval_modulus is intentionally NOT exposed as a CLI flag -- see
    # data/fomo300k_dataset.py::EVAL_HASH_MODULUS. Changing it would silently
    # break cross-version comparability (the whole point of this holdout),
    # so it is fixed in code rather than a footgun-able runtime argument.
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--grad_accum_steps", type=int, default=1,
                    help="gradient accumulation steps (wired into Accelerator; previously accepted "
                         "but silently a no-op before 2026-07-16 -- see module docstring)")
    p.add_argument("--data_source", type=str, default="preprocessed", choices=["preprocessed", "zip"],
                    help="'preprocessed' reads /root/data/FOMO300K_preprocessed/*.npz (default, fast); "
                         "'zip' reads FOMO300K's raw zip archives directly (~4x slower/sample, no "
                         "preprocessing step required)")
    p.add_argument("--limit", type=int, default=None,
                    help="debug: only index this many entries (preprocessed source) or zips (zip source)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume_from", type=str, default=None,
                    help="explicit checkpoint path to resume from (model+proj_heads+optimizer+"
                         "scheduler+step). Overrides auto-resume.")
    p.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoints/logs under this run_dir and start at "
                         "step 0 (default behavior auto-resumes from checkpoints/last.pt if it "
                         "exists -- see module docstring 'Resuming')")
    p.add_argument("--teacher_conditional_norm", action="store_true",
                    help="ver2: give each teacher its own InstanceNorm affine (gamma+beta) in the "
                         "student trunk instead of one shared affine -- see "
                         "networks/student.py::TeacherConditionalInstanceNorm3d. Conv weights (the "
                         "bulk of the params) stay shared/teacher-agnostic. Default off (ver1 "
                         "behavior, unchanged).")
    p.add_argument("--convpass_peft", action="store_true",
                    help="ver3 default (decoder-only, on top of --teacher_conditional_norm): adds "
                         "per-teacher skip scaling + parallel Convpass adapters on the decoder -- see "
                         "networks/student.py::DecoderStage/Convpass3D. The scalar gates (skip_alpha, "
                         "convpass_gate) get a 10x LR (--convpass_gate_lr_mult); Convpass conv weights "
                         "use the normal backbone LR. Does NOT require --teacher_conditional_norm -- "
                         "combine with --convpass_encoder/--convpass_no_skip_alpha for ver4 "
                         "(Convpass-only, encoder+decoder, no IN affine, no skip_alpha).")
    p.add_argument("--convpass_encoder", action="store_true",
                    help="ver4: also add Convpass parallel adapters to the encoder stages (default: "
                         "decoder-only, ver3 behavior) -- see networks/student.py::EncoderStage. "
                         "Requires --convpass_peft.")
    p.add_argument("--convpass_no_skip_alpha", action="store_true",
                    help="ver4: disable the decoder skip_alpha scaling (ver3 spec Sec A) -- Convpass-"
                         "only mode, removes the encoder/decoder Convpass asymmetry since encoder has "
                         "no skip tensor to scale in the first place. Requires --convpass_peft.")
    p.add_argument("--convpass_gate_lr_mult", type=float, default=10.0,
                    help="LR multiplier for skip_alpha/convpass_gate scalars only (ver3 spec Sec A/B)")
    p.add_argument("--peft_log_every", type=int, default=None,
                    help="ver3: log alpha/gate value snapshots + Convpass/backbone grad norms every N "
                         "steps (default: same as --eval_every, standing in for the spec's 'epoch마다' "
                         "-- this training loop has no epoch concept, see NOTES.md)")
    p.add_argument("--vjepa_cache_dir", type=str, default=None,
                    help="2026-08-02: if given, adds 'vjepa' as an episodic teacher backed by "
                         "PRECOMPUTED features (see /root/external_teacher_probe/extract_vjepa_cache.py) "
                         "instead of a live model -- no V-JEPA forward pass ever runs during training. "
                         "On a step where episodic sampling picks 'vjepa', both the student's input "
                         "image AND the teacher target come from this cache (a fixed, pre-cropped "
                         "30K-sample pool), not the normal dataloader. Requires --convpass_peft "
                         "--convpass_encoder (vjepa's targets include student ENCODER stages, so "
                         "encoder-side per-teacher Convpass capacity must exist) and "
                         "--teacher_conditional_norm/--convpass_peft implied teacher-identity "
                         "handling to know about 'vjepa' -- pass it in --teachers' effective roster "
                         "automatically, no separate flag needed there.")
    return p.parse_args()


def save_checkpoint(accelerator: Accelerator, student, proj_heads, optimizer, scheduler,
                     run_dir: str, step: int):
    """Saves EVERYTHING needed to both (a) hand off to asp_finetune_seg/cls/reg
    (the 'state_dict': {'model.<key>': tensor} contract, BaseModule.load_state_dict
    strict=False -- unaffected by the extra keys below, which asparagus simply
    ignores) AND (b) resume this exact training run (added 2026-07-17 per user
    request): proj_heads/optimizer/scheduler state + the step counter, so a
    second invocation with the same --run_name can continue exactly where this
    one left off rather than restarting cosine LR annealing / Adam
    momentum-variance from scratch."""
    unwrapped_student = accelerator.unwrap_model(student)
    unwrapped_proj = accelerator.unwrap_model(proj_heads)
    raw_sd = unwrapped_student.state_dict()
    prefixed_sd = {f"model.{k}": v for k, v in raw_sd.items()}
    ckpt = {
        "state_dict": prefixed_sd,
        "proj_heads_state_dict": unwrapped_proj.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "run_name": os.path.basename(run_dir),
    }
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step}.pt")
    torch.save(ckpt, path)
    torch.save(ckpt, os.path.join(ckpt_dir, "last.pt"))
    return path


def load_checkpoint_for_resume(path: str, student, proj_heads, optimizer, scheduler,
                                target_steps: int, num_processes: int) -> int:
    """Loads model/proj_heads/optimizer/scheduler state in-place from a
    checkpoint saved by save_checkpoint(). Must be called on EVERY rank (not
    just main) -- every DDP process needs the same starting weights, unlike
    saving which is main-process-only. Returns the step to resume from.

    Strips the "model." prefix back off state_dict keys (the prefix exists
    for asp_finetune_* compatibility, see save_checkpoint(), but student's
    own state_dict() doesn't have it).

    Scheduler resume (verified empirically 2026-07-17, see chat history --
    this took two attempts):

    Attempt 1 was `scheduler.load_state_dict(ckpt["scheduler_state_dict"])`
    followed by re-pinning `scheduler.T_max = target_steps`. This LOOKED
    correct in isolation (a plain repro script confirmed T_max/last_epoch
    end up right) but broke in the real training loop: LR got permanently
    stuck at 0 after resume. Root cause: `CosineAnnealingLR.get_lr()` is
    RECURSIVE, not closed-form, on every `.step()` call except the very
    first one after construction (`_step_count == 1`) -- its general branch
    computes `next_lr = f(last_epoch, T_max) * (CURRENT group['lr'] - eta_min)
    + eta_min`. `load_state_dict()` restores `_step_count` to whatever the
    checkpoint had (NOT 1, since the original run had already stepped
    several times), so the closed-form branch never fires again, and since
    `group['lr']` was checkpointed at exactly 0 (the tail of the old cosine
    curve), the recursive formula multiplies by zero forever -- 0 is a fixed
    point of that formula regardless of T_max.

    Fix: don't load the scheduler's state at all. Instead replicate what
    PyTorch's own `LRScheduler.__init__(..., last_epoch=N)` does internally
    (`_initial_step`: sets `_step_count = 0` then calls `.step()` once) --
    that forces the CLOSED-FORM branch (`_step_count == 1 and last_epoch >
    0`), which recomputes LR purely from `last_epoch`/`T_max`/`base_lrs`,
    independent of whatever `group['lr']` currently holds. `base_lrs` itself
    is untouched (still whatever main() set at construction) and doesn't
    need restoring.

    num_processes scaling (found 2026-07-20, see main()'s scheduler
    construction comment): this function runs on the RAW scheduler, before
    accelerator.prepare() wraps it in AcceleratedScheduler -- but that raw
    scheduler's T_max was already built as `args.steps * num_processes`, so
    `last_epoch` must be set in the same raw-step units (`start_step *
    num_processes`), not loop-step units, or the resumed LR would land on
    the wrong point of the (rescaled) cosine curve."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    student_sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    student.load_state_dict(student_sd)
    if "proj_heads_state_dict" in ckpt:
        # strict=False (2026-08-02): lets the teacher roster change across a resume (e.g. resuming
        # a run that didn't have --vjepa_cache_dir into one that now does) -- newly-added teachers'
        # heads just keep their fresh init instead of hard-failing the whole resume.
        proj_heads.load_state_dict(ckpt["proj_heads_state_dict"], strict=False)
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = ckpt.get("step", 0)
    scheduler.T_max = target_steps * num_processes
    scheduler.last_epoch = start_step * num_processes
    scheduler._step_count = 0
    scheduler.step()
    return start_step


# Tidy/long-format schema (one row per metric observation), NOT wide
# (one-column-per-teacher) -- deliberately, per 2026-07-17 request: future
# versions (e.g. ver2 with per-teacher LoRA/bias) may add/drop teachers or
# metrics, and a long format stays valid and directly concatenable across
# run_names without a schema migration. Both train_log.csv and eval_log.csv
# share this exact schema so tooling (see examples/compare_runs.py) can treat
# them uniformly.
_LOG_FIELDS = ["run_name", "step", "phase", "metric", "teacher", "value"]


def _open_log(run_dir: str, filename: str):
    path = os.path.join(run_dir, filename)
    is_new = not os.path.isfile(path)
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
    if is_new:
        writer.writeheader()
    return f, writer


def _log_row(writer, run_name: str, step: int, phase: str, metric: str, teacher: str, value: float):
    writer.writerow({"run_name": run_name, "step": step, "phase": phase,
                      "metric": metric, "teacher": teacher, "value": value})


@torch.no_grad()
def evaluate(student, proj_heads, teachers: dict, eval_loader, mse_weight: float, device,
             mse_weight_overrides: dict = None):
    """Runs the CURRENT student against EVERY teacher (not just whichever one
    episodic sampling would have picked) on the fixed monitoring holdout --
    see data/fomo300k_dataset.py::split_train_eval. Unlike training's
    per-step loss (which only ever reflects one randomly-sampled teacher),
    this gives a complete, directly-comparable per-teacher picture at every
    eval checkpoint, and does so identically regardless of what parameters
    the student/proj_heads happen to contain (shared-only today; ver2's
    LoRA/per-teacher-bias params need no changes here since this just calls
    forward_with_features()).

    Also tracks student feature-norm-per-stage as a cheap feature-collapse
    early-warning signal (a well-known failure mode in feature distillation:
    if a stage's norm drifts to ~0 or explodes, something is wrong long
    before the loss curve makes it obvious).

    ver2 note (2026-07-20): with `--teacher_conditional_norm`, the student's
    own output now DEPENDS on which teacher's affine params are active, so
    `forward_with_features()` must be called once PER TEACHER inside the
    loop below (not once per batch, shared across teachers, as ver1's single
    shared-affine student allowed) -- feat_norms is therefore keyed
    `"<teacher>/<stage>"` instead of bare `"<stage>"`.

    Returns a dict of {metric_key: value} plus per-teacher loss dict; caller
    logs it to eval_log.csv. Runs on a SINGLE process only (see call site --
    same pattern as save_checkpoint) since the holdout is small (~230
    entries by default) and doesn't need distributing.
    """
    was_training = student.training
    student.eval()
    proj_heads.eval()

    per_teacher_loss_sum = {name: 0.0 for name in teachers}
    per_teacher_batches = {name: 0 for name in teachers}
    feat_norm_sum = {}
    n_batches = 0

    for batch in eval_loader:
        # eval_loader is a plain (non-accelerator.prepare'd) DataLoader --
        # see call site: eval only ever runs on the main process, so no DDP
        # device placement happens automatically; move tensors explicitly.
        images = batch["image"].to(device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device)
        raw = batch.get("raw")  # VoCo's preprocess() needs this -- see main loop / voco_teacher.py
        if raw is not None:
            raw = raw.to(device)

        for name, teacher in teachers.items():
            if is_encoder_only_teacher(name):
                student_feats = student.forward_encoder_only(images, name)
            else:
                student_feats = student.forward_with_features(
                    images, name, include_encoder=teacher_needs_encoder_features(name))
            for k, v in student_feats.items():
                key = f"{name}/{k}"
                feat_norm_sum[key] = feat_norm_sum.get(key, 0.0) + v.norm(dim=1).mean().item()

            teacher_feats = teacher.extract_features(images, meta={"mask": mask, "raw": raw})
            projected = proj_heads.project(name, student_feats)
            effective_mse_weight = (mse_weight_overrides or {}).get(name, mse_weight)
            losses = episodic_distillation_loss(projected, teacher_feats, mse_weight=effective_mse_weight)
            per_teacher_loss_sum[name] += losses["total"].item()
            per_teacher_batches[name] += 1
        n_batches += 1

    if was_training:
        student.train()
        proj_heads.train()

    if n_batches == 0:
        return None

    return {
        "n_batches": n_batches,
        "per_teacher_loss": {name: per_teacher_loss_sum[name] / max(1, per_teacher_batches[name])
                              for name in teachers},
        "feat_norms": {k: v / n_batches for k, v in feat_norm_sum.items()},
    }


def main():
    args = parse_args()
    if (args.convpass_encoder or args.convpass_no_skip_alpha) and not args.convpass_peft:
        raise SystemExit("--convpass_encoder/--convpass_no_skip_alpha require --convpass_peft")
    if args.vjepa_cache_dir and not (args.convpass_peft and args.convpass_encoder):
        raise SystemExit("--vjepa_cache_dir requires --convpass_peft --convpass_encoder (vjepa's "
                          "targets include student ENCODER stages enc_stage_2/3, which only exist "
                          "as per-teacher Convpass capacity when --convpass_encoder is on -- see "
                          "networks/projections.py STAGE_MAP['vjepa'])")
    if args.peft_log_every is None:
        args.peft_log_every = args.eval_every
    run_dir = os.path.join(EXPR_ROOT, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # find_unused_parameters=True is required: only the sampled teacher's
    # projection-head parameters get gradients on any given step (episodic
    # sampling), so most steps leave most other teachers' heads unused --
    # standard DDP would otherwise error/hang on this. NOTE: this also means
    # DDP's `static_graph` optimization must NOT be enabled -- the set of
    # parameters participating in backward genuinely changes step to step.
    # rng_types=[] (2026-08-07): Accelerator's default rng_types=["generator"] makes every
    # accelerate-prepared DataLoader broadcast its sampler's generator state (torch.distributed.
    # broadcast, rank 0 -> others) on EVERY fresh iter(dataloader) call -- i.e. every epoch
    # restart, see accelerate/data_loader.py DataLoaderShard.__iter__. This is the confirmed root
    # cause of the recurring "Watchdog caught collective operation timeout ... Invalid mt19937
    # state" crashes seen on ver1/ver2 and much more frequently on ver5 (which restarts TWO
    # prepared dataloaders -- the main corpus loader and the vjepa cache loader -- instead of
    # one): if any rank calls this broadcast without all others calling it too (or noticeably
    # later than the others), the mismatched ranks hang until NCCL's 600s watchdog fires. We don't
    # need this synchronization -- each rank is meant to see a different data shard, and our
    # crash-resume behavior already doesn't depend on exact cross-rank RNG continuity (observed:
    # post-resume loss/eval values are close but not bit-identical to pre-crash, confirming
    # nothing here relies on it). Disabling it removes the crash trigger entirely rather than
    # just tolerating it via run_train.sh's auto-restart wrapper.
    accelerator = Accelerator(mixed_precision=args.mixed_precision, kwargs_handlers=[ddp_kwargs],
                               gradient_accumulation_steps=args.grad_accum_steps, rng_types=[])

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Fixed monitoring holdout (data/fomo300k_dataset.py::split_train_eval):
    # deterministic hash-based split, IDENTICAL across every run regardless
    # of --seed/run_name, so eval_log.csv rows are directly comparable across
    # future versions (ver2 with LoRA, etc.) at matching steps -- see module
    # docstring / evaluate(). Per 2026-07-17 request: everything NOT in this
    # small holdout (~0.075% of the corpus) still goes to training, matching
    # the FOMO25 precedent of using the full pretrain corpus for gradient
    # updates (this holdout exists purely for monitoring, not for early
    # stopping / model selection).
    if args.data_source == "preprocessed":
        full_index = load_preprocessed_index(limit=args.limit)
        train_index, eval_index = split_train_eval(full_index)
        dataset = FOMO300KPreprocessedDataset(patch_size=args.patch_size, index=train_index, seed=args.seed)
        eval_dataset = FOMO300KPreprocessedDataset(patch_size=args.patch_size, index=eval_index, seed=0) \
            if eval_index else None
    else:
        full_index = load_or_build_index(limit_zips=args.limit)
        train_index, eval_index = split_train_eval(full_index)
        dataset = FOMO300KDataset(patch_size=args.patch_size, index=train_index, seed=args.seed)
        eval_dataset = FOMO300KDataset(patch_size=args.patch_size, index=eval_index, seed=0) \
            if eval_index else None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, drop_last=True,
                             persistent_workers=args.num_workers > 0)
    eval_loader = None
    if eval_dataset is not None and args.eval_every > 0:
        # num_workers matches training's (not 0) -- eval __getitem__ cost is
        # dominated by CPU-bound npz decompression + on-the-fly znorm (same
        # as training samples), and with num_workers=0 a ~230-entry holdout
        # took ~135s serially in testing (2026-07-17). Parallelizing this is
        # pure I/O/CPU-bound speedup, no correctness concern (eval_dataset
        # has no shuffling/randomness to worry about across workers).
        eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                  num_workers=min(args.num_workers, 16))
    elif args.eval_every > 0:
        print(f"WARNING: monitoring holdout is empty (likely --limit={args.limit} is too small "
              f"for the hash-based split to select any entries) -- eval disabled for this run.")

    # Full teacher-identity roster, including "vjepa" if --vjepa_cache_dir is set -- used for
    # student per-teacher module identity (Convpass/conditional-norm) and episodic sampling.
    # "vjepa" is deliberately EXCLUDED from `teachers` below (no live model -- see get_teacher()),
    # so it must be added here rather than flow through --teachers/TEACHER_REGISTRY.
    all_teacher_names = list(args.teachers)
    if args.vjepa_cache_dir:
        all_teacher_names.append("vjepa")

    # teacher identity is needed whenever EITHER the IN affine is
    # teacher-conditional (--teacher_conditional_norm) OR Convpass is on
    # (--convpass_peft, which indexes its own per-teacher gate/adapter
    # ModuleDicts regardless of whether norm is conditional -- ver4 uses
    # --convpass_peft WITHOUT --teacher_conditional_norm for exactly this).
    needs_teacher_identity = args.teacher_conditional_norm or args.convpass_peft
    student = build_student(
        in_channels=1,
        teachers=all_teacher_names if needs_teacher_identity else None,
        convpass=args.convpass_peft,
        convpass_encoder=args.convpass_encoder,
        skip_alpha=not args.convpass_no_skip_alpha,
        norm_conditional=args.teacher_conditional_norm,
    )

    # Build one instance of each LIVE teacher PER PROCESS, on this process's own
    # device -- teachers are frozen/inference-only, no DDP wrapping needed.
    # "vjepa" (if present) never gets a live model -- its features are precomputed
    # (see FOMO26/data/vjepa_cache_dataset.py) -- only a fixed feature_specs entry.
    teachers = {name: get_teacher(name, device=str(accelerator.device)) for name in args.teachers}
    teacher_specs = {name: t.feature_specs for name, t in teachers.items()}
    if args.vjepa_cache_dir:
        teacher_specs["vjepa"] = VJEPA_FEATURE_SPECS

    proj_heads = build_projection_heads(student.decoder_stage_channels, student.encoder.out_channels,
                                         teacher_specs)

    vjepa_dataset = VJEPACachedDataset(args.vjepa_cache_dir) if args.vjepa_cache_dir else None
    vjepa_loader = None
    if vjepa_dataset is not None:
        if accelerator.is_main_process:
            print(f"vjepa cache: {len(vjepa_dataset)} samples from {args.vjepa_cache_dir}")
        vjepa_loader = DataLoader(vjepa_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=min(4, args.num_workers), drop_last=True,
                                   persistent_workers=min(4, args.num_workers) > 0)

    # ver3: skip_alpha/convpass_gate scalars get a 10x LR (spec Sec A/B);
    # everything else (backbone convs/IN affines, Convpass down/dw/up conv
    # weights themselves, proj_heads) uses the normal --lr. With
    # --convpass_peft off, convpass_gate_parameters() is empty and this
    # reduces to the ver1/ver2 single-group behavior.
    gate_params = student.convpass_gate_parameters() if args.convpass_peft else []
    gate_ids = {id(p) for p in gate_params}
    backbone_params = [p for p in student.parameters() if id(p) not in gate_ids] + list(proj_heads.parameters())
    if gate_params:
        param_groups = [
            {"params": backbone_params, "lr": args.lr},
            {"params": gate_params, "lr": args.lr * args.convpass_gate_lr_mult},
        ]
        print(f"ver3 param groups: backbone={sum(p.numel() for p in backbone_params):,} params @ lr={args.lr:.1e}, "
              f"convpass gates={sum(p.numel() for p in gate_params):,} params "
              f"@ lr={args.lr * args.convpass_gate_lr_mult:.1e}")
    else:
        param_groups = backbone_params
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    # accelerate's AcceleratedScheduler.step() (assigned by accelerator.prepare()
    # below) calls the wrapped scheduler's .step() `num_processes` times per
    # call, not once -- see accelerate/scheduler.py -- so a bare T_max=args.steps
    # completes a full cosine cycle every args.steps/num_processes loop
    # iterations, not args.steps (verified empirically 2026-07-20 on ver1's
    # actual 4-GPU run: lr hit exactly 0 at loop step 50000 = 200000/4, then
    # climbed back to peak by step 200000 -- 2 full down-up cycles instead of
    # one monotonic decay). Pre-multiplying by num_processes here cancels that
    # scaling so `--steps N` means N *loop* iterations, matching ckpt_every/
    # eval_every/log_every's units and the wall-clock estimates in run_train.sh.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps * accelerator.num_processes)

    # Resume (added 2026-07-17): default behavior auto-resumes from
    # checkpoints/last.pt if it exists in this run_dir -- re-running the same
    # --run_name command continues training rather than silently restarting
    # from step 0. --resume_from overrides with an explicit path (e.g. to
    # roll back to an earlier step_N.pt, or seed a new run from another run's
    # weights). --fresh forces starting at step 0 regardless.
    #
    # Loaded BEFORE accelerator.prepare() -- state loaded into the raw
    # (unwrapped) objects here is preserved through prepare()'s DDP/mixed-
    # precision wrapping. Must run on EVERY rank (unlike saving, which is
    # main-process-only): every DDP process needs identical starting weights.
    start_step = 0
    resume_path = args.resume_from
    if resume_path is None and not args.fresh:
        auto_path = os.path.join(run_dir, "checkpoints", "last.pt")
        if os.path.isfile(auto_path):
            resume_path = auto_path
    if resume_path is not None:
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {resume_path}")
        start_step = load_checkpoint_for_resume(resume_path, student, proj_heads, optimizer, scheduler,
                                                 args.steps, accelerator.num_processes)
        if accelerator.is_main_process:
            print(f"  resumed at step {start_step}, lr={scheduler.get_last_lr()[0]:.2e}")
    elif args.fresh and accelerator.is_main_process and os.path.isdir(os.path.join(run_dir, "checkpoints")):
        print(f"--fresh given: ignoring existing checkpoints under {run_dir}/checkpoints/ "
              f"(train_log.csv/eval_log.csv, if present, will still be APPENDED to, not overwritten)")

    if vjepa_loader is not None:
        student, proj_heads, optimizer, dataloader, scheduler, vjepa_loader = accelerator.prepare(
            student, proj_heads, optimizer, dataloader, scheduler, vjepa_loader
        )
    else:
        student, proj_heads, optimizer, dataloader, scheduler = accelerator.prepare(
            student, proj_heads, optimizer, dataloader, scheduler
        )

    teacher_weights = json.loads(args.teacher_weights) if args.teacher_weights else None
    mse_weight_overrides = json.loads(args.mse_weight_overrides) if args.mse_weight_overrides else {}
    # Same seed on every rank -> every rank samples the SAME teacher per step
    # (required for the DDP gradient-sync reasoning above to stay simple).
    sampler = EpisodicTeacherSampler(all_teacher_names, weights=teacher_weights, seed=args.seed)

    student.train()
    proj_heads.train()

    train_log_file, train_log_writer = (None, None)
    eval_log_file, eval_log_writer = (None, None)
    alpha_log_file, alpha_log_writer = (None, None)
    gate_log_file, gate_log_writer = (None, None)
    grad_norm_log_file, grad_norm_log_writer = (None, None)
    if accelerator.is_main_process:
        train_log_file, train_log_writer = _open_log(run_dir, "train_log.csv")
        eval_log_file, eval_log_writer = _open_log(run_dir, "eval_log.csv")
        if args.convpass_peft:
            # ver3 diagnostics (spec: "CSV로도 따로 남길 것") -- separate files
            # under logs/, same tidy schema as train/eval_log.csv so they can
            # be loaded with the same csv.DictReader pattern.
            os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
            alpha_log_file, alpha_log_writer = _open_log(run_dir, "logs/alpha_history.csv")
            gate_log_file, gate_log_writer = _open_log(run_dir, "logs/s_history.csv")
            grad_norm_log_file, grad_norm_log_writer = _open_log(run_dir, "logs/grad_norms.csv")

    data_iter = iter(dataloader)
    vjepa_iter = iter(vjepa_loader) if vjepa_loader is not None else None
    step = start_step
    t_last_log = time.time()
    loss_accum = 0.0

    if step >= args.steps and accelerator.is_main_process:
        print(f"Resumed step ({step}) already >= --steps ({args.steps}) -- nothing to do. "
              f"Pass a larger --steps to continue training this run.")

    while step < args.steps:
        # Teacher is sampled BEFORE pulling a batch (unlike the pre-vjepa version, which always
        # pulled from `dataloader` first) -- vjepa steps pull from a completely different loader
        # (fixed cached pool, no live teacher forward) instead of the normal corpus.
        teacher_name = sampler.sample()

        if teacher_name == "vjepa":
            try:
                vbatch = next(vjepa_iter)
            except StopIteration:
                vjepa_iter = iter(vjepa_loader)
                vbatch = next(vjepa_iter)
            images = vbatch["image"]
            teacher_feats = vbatch["teacher_feats"]  # precomputed -- {dec_stage_i: tensor}, no live forward
        else:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            images = batch["image"]  # accelerator.prepare'd dataloader moves batch tensors to device
            mask = batch.get("mask")
            raw = batch.get("raw")  # pre-asparagus-znorm crop -- only VoCo's preprocess() uses this
            teacher = teachers[teacher_name]

        with accelerator.accumulate(student):
            unwrapped = student.module if hasattr(student, "module") else student
            if is_encoder_only_teacher(teacher_name):
                # VoCo (currently the only one): no decoder counterpart exists at all, so running
                # the decoder here would be pure wasted compute+backward -- skip it entirely.
                student_feats = unwrapped.forward_encoder_only(images, teacher_name)
            else:
                # include_encoder=True only costs anything when the teacher's STAGE_MAP actually
                # references an "enc" side entry (currently just vjepa's mixed enc+dec targets) --
                # the encoder skips are always computed regardless, this flag only controls
                # whether they're ALSO returned.
                student_feats = unwrapped.forward_with_features(
                    images, teacher_name, include_encoder=teacher_needs_encoder_features(teacher_name))
            if teacher_name != "vjepa":
                # frozen, torch.no_grad() internally; mask only used by BraTS's preprocess() (z-score
                # needs a foreground mask), raw only used by VoCo's preprocess() (its own whole-volume
                # z-score can't safely compose with asparagus's already-clamped-and-normed `images`,
                # see teacher/voco_teacher.py) -- every other teacher ignores unrecognized meta keys.
                teacher_feats = teacher.extract_features(images, meta={"mask": mask, "raw": raw})

            proj_module = proj_heads.module if hasattr(proj_heads, "module") else proj_heads
            projected = proj_module.project(teacher_name, student_feats)

            effective_mse_weight = mse_weight_overrides.get(teacher_name, args.mse_weight)
            losses = episodic_distillation_loss(projected, teacher_feats, mse_weight=effective_mse_weight)
            loss = losses["total"]

            optimizer.zero_grad()
            accelerator.backward(loss)

            # ver3 diagnostics (spec: "Convpass 경로 gradient norm"/"백본
            # gradient norm", 매 로깅 스텝) -- computed on the raw gradients
            # right after backward(), before optimizer.step() touches them.
            # Cheap (just summing squares of already-computed .grad tensors)
            # so gating it behind the log_every cadence is about log-file
            # size, not runtime cost.
            if args.convpass_peft and (step + 1) % args.log_every == 0 and accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(student)
                convpass_params = unwrapped.convpass_path_parameters()
                convpass_ids = {id(p) for p in convpass_params}
                convpass_grad_sq = sum(p.grad.pow(2).sum().item() for p in convpass_params if p.grad is not None)
                backbone_grad_sq = sum(p.grad.pow(2).sum().item() for p in unwrapped.parameters()
                                       if p.grad is not None and id(p) not in convpass_ids)
                _log_row(grad_norm_log_writer, args.run_name, step + 1, "ver3", "grad_norm",
                         "backbone", backbone_grad_sq ** 0.5)
                _log_row(grad_norm_log_writer, args.run_name, step + 1, "ver3", "grad_norm",
                         "convpass", convpass_grad_sq ** 0.5)
                grad_norm_log_file.flush()

            optimizer.step()
            scheduler.step()

        loss_accum += loss.item()
        step += 1

        if step == 100 and accelerator.is_main_process:
            # ver3 spec: "GPU mem, step time -- 시작 후 100 step -- 오버헤드 확인"
            if torch.cuda.is_available():
                mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"  [overhead check @ step 100] peak GPU mem so far: {mem_gb:.2f} GB")

        if args.convpass_peft and step % args.peft_log_every == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(student)
            snapshot = unwrapped.alpha_gate_snapshot()
            for key, val in snapshot["alpha"].items():
                _log_row(alpha_log_writer, args.run_name, step, "ver3", "alpha", key, val)
            for key, val in snapshot["gate"].items():
                _log_row(gate_log_writer, args.run_name, step, "ver3", "gate", key, val)
            alpha_log_file.flush()
            gate_log_file.flush()

        if step % args.log_every == 0 and accelerator.is_main_process:
            dt = time.time() - t_last_log
            avg_loss = loss_accum / args.log_every
            print(f"step {step}/{args.steps} teacher={teacher_name} loss={avg_loss:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} ({dt/args.log_every*1000:.0f}ms/step)")
            _log_row(train_log_writer, args.run_name, step, "train", "distill_loss", teacher_name, avg_loss)
            train_log_file.flush()
            loss_accum = 0.0
            t_last_log = time.time()

        if eval_loader is not None and step % args.eval_every == 0 and accelerator.is_main_process:
            t_eval_start = time.time()
            unwrapped_student = accelerator.unwrap_model(student)
            unwrapped_proj = accelerator.unwrap_model(proj_heads)
            result = evaluate(unwrapped_student, unwrapped_proj, teachers, eval_loader,
                               args.mse_weight, accelerator.device,
                               mse_weight_overrides=mse_weight_overrides)
            eval_dt = time.time() - t_eval_start
            if result is not None:
                for name, loss_val in result["per_teacher_loss"].items():
                    _log_row(eval_log_writer, args.run_name, step, "eval", "distill_loss", name, loss_val)
                for stage, norm_val in result["feat_norms"].items():
                    _log_row(eval_log_writer, args.run_name, step, "eval", "feat_norm", stage, norm_val)
                eval_log_file.flush()
                loss_str = " ".join(f"{n}={v:.4f}" for n, v in result["per_teacher_loss"].items())
                print(f"  [eval step {step}] n_batches={result['n_batches']} {loss_str} ({eval_dt:.1f}s)")
            # Excluded from the training-throughput (ms/step) window above --
            # otherwise eval cost gets misattributed to the next logged
            # training step's speed, making it look like training itself
            # stalled (observed in testing: a step falsely showing
            # "136114ms/step" that was actually ~135s of eval).
            t_last_log = time.time()

        if step % args.ckpt_every == 0 and accelerator.is_main_process:
            path = save_checkpoint(accelerator, student, proj_heads, optimizer, scheduler, run_dir, step)
            print(f"saved checkpoint: {path}")

    if accelerator.is_main_process:
        save_checkpoint(accelerator, student, proj_heads, optimizer, scheduler, run_dir, step)
        print(f"training complete, final checkpoint saved under {run_dir}/checkpoints/")
        if train_log_file is not None:
            train_log_file.close()
        if eval_log_file is not None:
            eval_log_file.close()
        if alpha_log_file is not None:
            alpha_log_file.close()
        if gate_log_file is not None:
            gate_log_file.close()
        if grad_norm_log_file is not None:
            grad_norm_log_file.close()

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
