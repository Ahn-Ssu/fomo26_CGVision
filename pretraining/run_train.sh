#!/usr/bin/env bash
# FOMO26 training launcher.
#
# Usage:
#   ./run_train.sh <run_name> [extra run_pretrain.py args...]
#
# Examples:
#   ./run_train.sh ver1
#   ./run_train.sh ver1 --steps 50000 --batch_size 4
#   NUM_GPUS=2 ./run_train.sh smoke --steps 10 --limit 500   # quick sanity run
#
# <run_name> is required and not defaulted on purpose -- every training run
# should be explicitly named (ver1, ver2, ...) since run_name is the axis
# examples/compare_runs.py compares across (see README.md "Monitoring").
#
# Hyperparameters below are tunable via environment variables (so repeated
# runs -- ver1, ver2 with LoRA, etc. -- can override just what changed
# without editing this file); anything not covered by an env var can be
# passed as extra CLI args, which are forwarded as-is to run_pretrain.py
# and override the corresponding flag below (argparse: last value wins).
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run_name> [extra run_pretrain.py args...]" >&2
  exit 1
fi

RUN_NAME="$1"
shift

NUM_GPUS="${NUM_GPUS:-4}"
# STEPS=200000 @ ~820ms/step (4 GPU, batch_size=4, global batch 16) ~= 2 days
# wall-clock, ~10.5 "epoch"-equivalents over the ~305,860-scan training corpus.
# Chosen 2026-07-17 after comparing against DINO/iBOT/MAE/DINOv2 published
# recipes (100-500+ epochs on their own datasets) -- NOT directly comparable
# since those bootstrap representations from scratch while we distill from an
# already-competent FROZEN teacher (structurally faster to converge) -- see
# CLAUDE_CODE_RUN_REPORTS or chat history for the full reasoning. Real
# stopping criterion is watching eval_log.csv for a plateau, not this number
# in isolation; --resume_from / auto-resume (see run_pretrain.py) makes it
# cheap to extend a run instead of overcommitting to one fixed budget upfront.
STEPS="${STEPS:-200000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PATCH_SIZE="${PATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-1e-4}"
LOG_EVERY="${LOG_EVERY:-250}"
CKPT_EVERY="${CKPT_EVERY:-10000}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SEED="${SEED:-0}"

# Auto-restart-on-crash (added 2026-07-24, after two separate NCCL watchdog
# crashes -- ver1 @ step ~12500, ver2 @ step ~193750 -- both "[rank3] ...
# Watchdog caught collective operation timeout ... StopIteration ...
# synchronize_rng_states ... Invalid mt19937 state", i.e. a rank-desync at a
# dataloader epoch boundary. dmesg was checked at both exact crash timestamps
# on the host: no OOM-kill, no Xid, nothing -- rules out hardware/memory as
# the cause. Root cause not fully pinned down, so this wraps the launch in a
# retry loop instead: on any non-zero exit, wait RETRY_DELAY seconds and
# re-run the SAME command. No --fresh is passed, so run_pretrain.py's
# existing auto-resume (checkpoints/last.pt, verified working -- see chat
# history) picks up where it left off automatically. Set MAX_RETRIES=0 to
# disable and get the old single-shot exec behavior back.
MAX_RETRIES="${MAX_RETRIES:-20}"
RETRY_DELAY="${RETRY_DELAY:-30}"

VENV_ACTIVATE="/root/teachers/venv/bin/activate"
REPO_ROOT="/root"

# shellcheck disable=SC1090
source "$VENV_ACTIVATE"
cd "$REPO_ROOT"

echo "=== FOMO26 training: run_name=$RUN_NAME ==="
echo "GPUs=$NUM_GPUS steps=$STEPS batch_size=$BATCH_SIZE patch_size=$PATCH_SIZE lr=$LR seed=$SEED"
echo "Extra args: $*"
echo "Auto-restart: max_retries=$MAX_RETRIES retry_delay=${RETRY_DELAY}s"
echo "Logs: FOMO26/expr/$RUN_NAME/{train_log.csv,eval_log.csv}"
echo

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "=== launch attempt $attempt (run_name=$RUN_NAME) ==="

  set +e
  accelerate launch --multi_gpu --num_processes "$NUM_GPUS" FOMO26/run_pretrain.py \
    --run_name "$RUN_NAME" \
    --steps "$STEPS" \
    --batch_size "$BATCH_SIZE" \
    --patch_size "$PATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --lr "$LR" \
    --log_every "$LOG_EVERY" \
    --ckpt_every "$CKPT_EVERY" \
    --eval_every "$EVAL_EVERY" \
    --mixed_precision "$MIXED_PRECISION" \
    --seed "$SEED" \
    "$@"
  exit_code=$?
  set -e

  if [ "$exit_code" -eq 0 ]; then
    echo "=== training finished successfully (run_name=$RUN_NAME, attempt $attempt) ==="
    exit 0
  fi

  if [ "$MAX_RETRIES" -le 0 ] || [ "$attempt" -ge "$MAX_RETRIES" ]; then
    echo "=== giving up after $attempt attempt(s), exit code $exit_code (run_name=$RUN_NAME) ===" >&2
    exit "$exit_code"
  fi

  echo "=== crashed (exit code $exit_code) -- retrying in ${RETRY_DELAY}s, will auto-resume from" \
       "checkpoints/last.pt (attempt $((attempt + 1))/$MAX_RETRIES) ==="
  sleep "$RETRY_DELAY"
done


# nohup ./FOMO26/run_train.sh ver1 > /root/FOMO26/expr/ver1_launch.log 2>&1 &
# 내부적으로:


# accelerate launch --multi_gpu --num_processes 4 FOMO26/run_pretrain.py \
#   --run_name ver1 --steps 200000 --batch_size 4 --patch_size 128 \
#   --num_workers 8 --lr 1e-4 --log_every 250 --ckpt_every 25000 \
#   --eval_every 5000 --mixed_precision bf16 --seed 0