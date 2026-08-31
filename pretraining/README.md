# FOMO26 — Multi-Teacher Episodic Distillation Pretraining

Phase 1 (`ver1`): single student `f_theta`, one frozen teacher sampled per step
(episodic), feature-level distillation only, fully shared trunk weights.

Phase 2 (`ver2`, added 2026-07-20): `--teacher_conditional_norm` gives each teacher
its own InstanceNorm affine (gamma+beta) in the student trunk (BitFit-style
bias-tuning, adapted to CNNs via conditional/domain-specific instance norm rather
than linear-layer bias) -- conv weights (the bulk of the params) stay shared
across teachers. See "ver2: teacher-conditional norm" below.

## Directory layout

```
FOMO26/
├── CLAUDE_CODE_RUN_REPORTS/   # All analysis/report deliverables, numbered by request order
│   ├── 01_ASPARAGUS_ANALYSIS.md    # Task 1: asparagus codebase reuse-vs-build analysis
│   ├── 02_TEACHER_REPORT.md        # Task 2: teacher acquisition/verification/feature-hook report
│   └── 03_PREPROCESSING_REPORT.md  # Task 3: raw-preserving preprocessing + normalization report
├── teacher/           # Frozen teacher wrappers (moved from /root/teachers/wrappers/,
│                       # 2026-07-15). Large checkpoints/repos/venv stay under
│                       # /root/teachers/ -- this dir only holds the editable code.
│   ├── base.py           # BaseTeacher abstract interface
│   ├── anatomix_teacher.py
│   ├── vesselfm_teacher.py
│   ├── brats_teacher.py
│   ├── registry.py       # TEACHER_REGISTRY, get_teacher()
│   ├── FASTSURFER_EXCLUDED.md  # why FastSurfer isn't in the registry
│   └── BRATS_BLOCKED.md        # historical -- now resolved, kept for context
├── networks/
│   ├── student.py         # StudentResEncUNet -- EDIT THIS to change student architecture
│   └── projections.py     # per-(teacher,stage) 1x1x1 conv projection heads
├── data/
│   ├── fomo300k_dataset.py  # FOMO300KPreprocessedDataset (default, reads the .npz cache) +
│   │                        # FOMO300KDataset (zip fallback, for data not yet preprocessed)
│   ├── normalize.py         # asparagus_volume_wise_znorm -- single shared implementation
│   └── preprocess_raw.py    # one-time batch job: FOMO300K zips -> raw-preserving .npz cache
│                             # (already run for the full corpus 2026-07-15/16, see
│                             # /root/FOMO26/CLAUDE_CODE_RUN_REPORTS/03_PREPROCESSING_REPORT.md -- 306,090 scans, 2.5TB at
│                             # /root/data/FOMO300K_preprocessed/, split across two log files:
│                             # preprocess_log.csv + preprocess_log_openneuro.csv, both read
│                             # by default)
├── losses/
│   └── distillation.py    # cosine + MSE feature loss
├── sampler/
│   └── episodic.py        # picks one teacher per training step
├── expr/                  # training run outputs land here: expr/<run_name>/{config.json,checkpoints/}
├── examples/               # fast standalone sanity checks, see below
└── run_pretrain.py         # main entry point (accelerate, multi-GPU)
```

## Why `accelerate`, not `asparagus`

asparagus's own pretraining `LightningModule` (`SelfSupervisedModule`) is a single-model
MSE-reconstruction trainer with no teacher concept; the closest precedent
(`DINOv2Module`) is tightly coupled to a specific DINO/iBOT loss and a single EMA
teacher-student pair. Wiring 3 heterogeneous external teacher repos (each with its own
preprocessing) through asparagus's Hydra config composition would be substantial extra
engineering for limited payoff at this stage. asparagus remains the right tool for
**finetuning/eval** (see `/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/01_ASPARAGUS_ANALYSIS.md`), and that boundary is preserved:
every checkpoint `run_pretrain.py` saves is `{"state_dict": {"model.<key>": tensor, ...}}`
— the exact contract `asp_finetune_seg/cls/reg` already expects, no shim needed.

## Running

```bash
source /root/teachers/venv/bin/activate

# Fast sanity checks (run these first after any edit):
python3 FOMO26/examples/sanity_check_teachers.py
python3 FOMO26/examples/sanity_check_dataset.py --limit 100 --n_samples 5
python3 FOMO26/examples/sanity_check_train_step.py --steps 5 --limit 50

# Single-GPU / CPU smoke test of the real training script:
python3 FOMO26/run_pretrain.py --run_name smoke --steps 5 --limit 50

# Full run, all 4 GPUs (accelerate config already set up: multi-GPU, 4 processes, bf16):
accelerate launch --multi_gpu --num_processes 4 FOMO26/run_pretrain.py \
    --run_name my_run --steps 100000 --batch_size 2 --patch_size 128 --grad_accum_steps 1

# ...or the launcher script (same thing, with tunable defaults + a required run_name):
./FOMO26/run_train.sh ver1
STEPS=100000 BATCH_SIZE=4 ./FOMO26/run_train.sh ver2 --teachers anatomix+brains vesselfm
```

`run_train.sh <run_name> [extra args]` wraps the `accelerate launch` invocation above.
Hyperparameters are overridable via environment variables (`NUM_GPUS`, `STEPS`,
`BATCH_SIZE`, `PATCH_SIZE`, `NUM_WORKERS`, `LR`, `LOG_EVERY`, `CKPT_EVERY`,
`EVAL_EVERY`, `MIXED_PRECISION`, `SEED`, `MAX_RETRIES`, `RETRY_DELAY` -- see the script for
defaults); anything else can be passed as extra CLI args, which are forwarded to
`run_pretrain.py` and override the corresponding flag (argparse: last value wins). `run_name`
has no default on purpose -- every run should be explicitly named (`ver1`, `ver2`, ...) since
it's the axis `examples/compare_runs.py`/`examples/plot_curves.py` compare across.

`run_train.sh` auto-restarts on crash (added 2026-07-24): the launch runs in a retry loop
(`MAX_RETRIES`, default 20; `RETRY_DELAY` seconds between attempts, default 30) -- any
non-zero exit (e.g. the NCCL watchdog crashes seen 2026-07-17/2026-07-23, root cause not fully
pinned down -- see "Resuming a run" below) triggers a wait-then-relaunch of the exact same
command, which auto-resumes from `checkpoints/last.pt` with no `--fresh` needed. Set
`MAX_RETRIES=0` for the old single-shot behavior.

`--data_source zip` falls back to reading FOMO300K's raw zip archives directly (no
preprocessing required, ~4x slower per sample) — useful only for data not yet run through
`data/preprocess_raw.py`.

Outputs land in `FOMO26/expr/<run_name>/` (`config.json` snapshot +
`checkpoints/step_N.pt` + `checkpoints/last.pt`).

## Monitoring (added 2026-07-17)

Training uses the FULL corpus for gradient updates (no train/val split, matching the
FOMO25 precedent of training on the entire pretrain set) -- but every `--eval_every`
steps (default 500), the CURRENT student is run against **every** teacher (not just
whichever one episodic sampling picked that step) on a small **fixed monitoring
holdout** (`data/fomo300k_dataset.py::split_train_eval`, ~0.075% of the corpus,
deterministic crc32-hash split -- the SAME holdout on every run, forever, regardless of
`--seed`/`--run_name`). This is what makes `ver1` vs `ver2` (e.g. ver2 adding
teacher-specific LoRA/bias params) comparable at matching steps: both see identical
eval data. The eval-set modulus is intentionally NOT a CLI flag -- exposing it would let
someone accidentally break comparability between runs.

Two log files per run, both in the same tidy/long schema
(`run_name,step,phase,metric,teacher,value`) so they concatenate cleanly across runs:
- `checkpoints/../train_log.csv` -- per-step training loss (one row per `--log_every`)
- `checkpoints/../eval_log.csv` -- per-teacher eval loss + per-stage student feature
  norm (a cheap feature-collapse early-warning signal) at every eval checkpoint

Compare runs:
```bash
python3 FOMO26/examples/compare_runs.py ver1 ver2 [ver3 ...]
python3 FOMO26/examples/compare_runs.py ver1 ver2 --metric feat_norm
```

Plot learning curves (added 2026-07-23): one subplot per teacher, runs overlaid as different
colors, for both train and eval `distill_loss`. Handles resume-induced duplicate step ranges
(keeps the later/authoritative row) and clips the y-axis view to a percentile (rare extreme-loss
outlier steps -- e.g. one bad batch -- would otherwise crush the whole informative range to a
flat line; the underlying CSV is untouched, only the plot view is clipped):
```bash
python3 FOMO26/examples/plot_curves.py ver1 ver2
python3 FOMO26/examples/plot_curves.py ver1 ver2 --out FOMO26/expr/plots --smooth 10
```
Saves `train_loss_<runs>.png` / `eval_loss_<runs>.png` under `--out` (default `FOMO26/expr/plots`).

## LR schedule and multi-GPU (bug found + fixed 2026-07-20)

`accelerate`'s `AcceleratedScheduler.step()` (installed by `accelerator.prepare()`)
calls the wrapped scheduler's `.step()` **`num_processes` times per call**, not once,
whenever `split_batches=False` (the default -- and correct for us, since our
per-process `--batch_size` means each of the `num_processes` GPUs pulls its own
batch rather than splitting one global batch). `run_pretrain.py` builds
`CosineAnnealingLR` with `T_max = args.steps * accelerator.num_processes` to cancel
this out, so `--steps N` means N *loop* iterations (matching `ckpt_every`/
`eval_every`/`log_every`'s units), not N/num_processes.

**`ver1`'s full 200k-step run (2026-07-17 to 2026-07-19) was trained WITHOUT this
fix** -- `T_max` was `args.steps` unscaled, so on 4 GPUs the LR actually completed
**2 full down-up cosine cycles** instead of one monotonic decay to 0 (confirmed
from `train_log.csv`: `lr` hits exactly `0.00e+00` at loop step 50000 = 200000/4,
then climbs back to the starting `1e-4` by step 200000). Training loss still
converged and looked stable throughout (~0.10-0.12 by the end), so the run is
likely still usable, but it did NOT get the intended "settle at a low LR" tail
that motivated picking a 200k-step budget in the first place -- keep this in mind
if comparing `ver1` against `ver2`+ (which will have the corrected schedule).

## ver2: teacher-conditional norm (added 2026-07-20)

```bash
./FOMO26/run_train.sh ver2 --teacher_conditional_norm
```

Each `ResidualBlock3D`'s two `InstanceNorm3d` layers get a separate (gamma, beta)
pair PER TEACHER (`networks/student.py::TeacherConditionalInstanceNorm3d`) instead
of one shared affine -- everything else (conv weights, the per-teacher projection
heads that already existed in Phase 1) is unchanged. All teachers' affine params
coexist as `nn.ParameterDict`s at all times (same pattern as the projection
heads) -- `forward_with_features(x, teacher_name)` just indexes into whichever
teacher was episodically sampled that step, so only that teacher's affine
receives gradient (relies on the existing DDP `find_unused_parameters=True`).

`--teacher_conditional_norm` defaults off, so `ver1`'s exact architecture (bare
`nn.InstanceNorm3d(affine=True)`, same state_dict keys) is still reachable from
the same codebase -- useful for `ver1`-vs-`ver2` ablations. Because the conv-layer
state_dict keys are identical in both modes (only the norm submodule's internals
differ), warm-starting `ver2` from `ver1`'s trained weights is possible via
`--resume_from .../ver1/checkpoints/step_200000.pt`, but note that
`load_checkpoint_for_resume()` currently does a strict `load_state_dict()` --
loading `ver1` weights into a `ver2` student needs `strict=False` (the missing
per-teacher affine keys just fall back to their gamma=1/beta=0 init). Not yet
wired up as a CLI option since it hasn't been asked for -- flag if wanted.

`evaluate()`'s per-stage feature norms are now logged as `"<teacher>/<stage>"`
(e.g. `brats/dec_stage_0`) instead of bare `"<stage>"`, since the student's own
output differs per teacher once norms are teacher-conditional -- `compare_runs.py`
needs no changes (it treats the `teacher` column as an opaque grouping key).

## Resuming a run (added 2026-07-17)

Every run **auto-resumes** by default: on start, `run_pretrain.py` checks for
`expr/<run_name>/checkpoints/last.pt` and, if present, loads model + proj_heads +
optimizer + scheduler state and continues from the saved `step` -- so re-running
the exact same command (e.g. after a crash, preemption, or just to extend a
finished run with a larger `--steps`) just works:

```bash
./FOMO26/run_train.sh ver1                    # first launch
./FOMO26/run_train.sh ver1                    # re-run later -- auto-resumes from last.pt
STEPS=300000 ./FOMO26/run_train.sh ver1       # extend a completed 200k run to 300k
```

- `--resume_from <path>` overrides auto-detection with an explicit checkpoint path
  (e.g. resuming `ver2` from `ver1`'s weights).
- `--fresh` ignores any existing checkpoints and starts over at step 0.
- `last.pt` is overwritten on every `--ckpt_every` save (alongside a numbered
  `step_N.pt`), so resume always continues from the most recent save.
- `run_train.sh` now wraps this in an automatic retry loop on crash -- see
  `MAX_RETRIES`/`RETRY_DELAY` above -- so you don't need to manually re-invoke this after
  every crash; it happens on its own.

**Recurring NCCL watchdog crash (seen 2026-07-17 @ ver1 step ~12500, 2026-07-23 @ ver2 step
~193750, both "[rank3] ... Watchdog caught collective operation timeout ... StopIteration ...
synchronize_rng_states ... Invalid mt19937 state")**: root cause not fully pinned down. Checked
and ruled out on the host (both exact crash timestamps): no OOM-kill, no GPU Xid errors, nothing
in dmesg at all -- not a hardware/memory issue. Leading hypothesis is a per-rank DataLoader
epoch-boundary desync (one rank's shard exhausts at a different loop-iteration count than the
others, so its `synchronize_rng_states()` broadcast on re-iterating finds no partner), but this
isn't proven. Given it's rare and the resume mechanism is solid, `run_train.sh`'s auto-restart
loop (above) is the practical mitigation in place of chasing the root cause further.

**LR scheduler gotcha (worth knowing if you touch this code):** `CosineAnnealingLR`
cannot be resumed via a plain `load_state_dict()` when extending `--steps` past a
run that already reached (or neared) LR 0 -- `get_lr()` is recursive on the
*current* `param_group['lr']` for every call after the first, so once that value
hits 0 it's a fixed point and the LR stays at 0 forever regardless of the new
`T_max`. `load_checkpoint_for_resume()` in `run_pretrain.py` works around this by
re-deriving the scheduler purely from `last_epoch`/`T_max` (mirroring what
PyTorch's own `last_epoch=` constructor argument does internally) instead of
trusting the checkpointed scheduler state -- see the docstring there for the full
story if this needs touching again.

## Known limitations / next steps (not yet done)

- ~~FOMO300K is read directly from its 81,190 zip archives~~ → **done (2026-07-16)**.
  `run_pretrain.py` now defaults to `FOMO300KPreprocessedDataset`, reading the
  raw-preserving `.npz` cache built by `data/preprocess_raw.py` (full corpus preprocessed
  2026-07-15 — see `/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/03_PREPROCESSING_REPORT.md`). ~4x faster per sample than the zip
  path (no zip extraction or SimpleITK resampling at train time — the cache is already
  1mm-isotropic with an explicit foreground mask). The zip-based `FOMO300KDataset` is kept
  as `--data_source zip` for any future data not yet preprocessed.
- **Teacher roster is 3 (anatomix, vesselfm, brats)**; FastSurfer is excluded from live
  per-step use (2D architecture + absolute-scale normalization — see
  `teacher/FASTSURFER_EXCLUDED.md`) but could be added later as an **offline
  precomputed output-level pseudo-label source** (a different code path, not the live
  `BaseTeacher.extract_features()` pattern) if wanted.
- **Student ↔ teacher stage mapping** (`networks/projections.py::STAGE_MAP`) is drawn
  from `/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/02_TEACHER_REPORT.md` Sec E and has not been tuned/validated for actual
  distillation quality yet — it's a structurally-sound starting point (spatial scales
  match exactly for VesselFM/BraTS since the student mirrors their nnU-Net 6-stage plan;
  Anatomix is shifted by one stage since it's a 4-stage network).
- **Episodic sampler is uniform random by default.** PRE_ANALYSIS.md Sec 5.1 found
  BraTS teacher confidence varies strongly by modality (FLAIR Dice 0.926 vs T1 0.856) —
  hooking that up as a sampling weight (using the `modality` field the Dataset already
  returns but the sampler doesn't yet consume) is a natural next step.
- ~~Phase 2 (not started): teacher-specific student parameters~~ → **partially done
  (2026-07-20)**: teacher-conditional InstanceNorm affine (bias+gamma), see "ver2:
  teacher-conditional norm" above. LoRA on the conv weights themselves is still
  not implemented, if wanted beyond bias-tuning.
