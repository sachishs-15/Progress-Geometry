#!/bin/bash
# Run precompute_features.py sharded across the 4 local GPUs, then merge the shards.
#
# No taskset CPU pinning (unlike a naive parallel wrapper might do) — under this
# cluster's PBS cgroups, taskset's assumed 0..ncpus-1 CPU numbering doesn't match the
# actual allowed CPU set, and a failed taskset call silently drops that worker's whole
# shard with no error propagated (see LiFT/precompute_features_parallel.sh, which hit
# exactly this and had it fixed). CUDA_VISIBLE_DEVICES alone is enough to isolate each
# worker's GPU; the PBS cgroup already caps total CPU usage.
#
# Also no `conda activate` — it depends on ~/.bashrc's conda-init hook, which
# non-interactive PBS shells don't source. Calling the env's python by full path
# sidesteps that.
#
# Usage: ./precompute_features_parallel.sh <csv> <out-dir> [model] [n-frames] [pooling] [encoder-kwargs] [patch-tokens]
#   encoder-kwargs: semicolon-separated key=value, e.g. "capture_attention=False;per_frame_batch_size=16"
#   patch-tokens: any non-empty value enables --patch-tokens (raw per-patch grid instead
#     of one pooled vector/frame -- requires a model with supports_patch_tokens=True,
#     e.g. dinov2_noreg). NUM_GPUS env var overrides the default of 4 (e.g. NUM_GPUS=7 to
#     use all free GPUs on a node where something else already holds 1).
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON=/home/sachish/.conda/envs/frozen-encoders/bin/python

CSV=${1:?Usage: $0 <csv> <out-dir> [model] [n-frames] [pooling] [encoder-kwargs] [patch-tokens]}
OUT_DIR=${2:?Usage: $0 <csv> <out-dir> [model] [n-frames] [pooling] [encoder-kwargs] [patch-tokens]}
MODEL=${3:-dinov2}
N_FRAMES=${4:-16}
POOLING=${5:-mean_patch}
ENCODER_KWARGS=${6:-}
PATCH_TOKENS=${7:-}

NUM_GPUS=${NUM_GPUS:-4}
CSV_STEM=$(basename "$CSV" .csv)
# precompute_features.py itself nests its output under <out-dir>/<model>/ (so re-running
# the same csv through a different --model never collides) — mirror that here so the
# merge step below looks in the same place, and so logs are model-distinguished too.
MODEL_OUT_DIR="${OUT_DIR}/${MODEL}"

mkdir -p "logs/${CSV_STEM}/${MODEL}"

EXTRA_ARGS=()
if [ -n "$ENCODER_KWARGS" ]; then
  EXTRA_ARGS+=(--encoder-kwargs "$ENCODER_KWARGS")
fi
if [ -n "$PATCH_TOKENS" ]; then
  EXTRA_ARGS+=(--patch-tokens)
fi

# Bash's bare `wait` (no args) always returns 0 no matter what happened to any
# backgrounded child -- so even with `set -eu` above, a shard dying mid-run (e.g. an OOM
# kill during precompute_features.py's own finalization step) would silently NOT trip this
# script's error handling, and it would sail straight into merge_features.py below
# regardless of which shards actually finished. Track each shard's own PID and wait on it
# individually instead, so a failure is caught and the (incomplete, silently wrong) merge
# never runs.
pids=()
for gpu in $(seq 0 $((NUM_GPUS-1))); do
  echo "[gpu ${gpu}] starting shard ${gpu}/${NUM_GPUS}"
  CUDA_VISIBLE_DEVICES=${gpu} \
    "$PYTHON" precompute_features.py --csv "$CSV" --out-dir "$OUT_DIR" \
      --model "$MODEL" --n-frames "$N_FRAMES" --pooling "$POOLING" --device cuda:0 \
      --shard-id ${gpu} --num-shards ${NUM_GPUS} "${EXTRA_ARGS[@]}" \
      > "logs/${CSV_STEM}/${MODEL}/precompute_gpu${gpu}.log" 2>&1 &
  pids+=($!)
done

failed=0
for gpu in "${!pids[@]}"; do
  if wait "${pids[$gpu]}"; then
    :
  else
    ec=$?
    echo "[gpu ${gpu}] FAILED (exit ${ec}) -- see logs/${CSV_STEM}/${MODEL}/precompute_gpu${gpu}.log" >&2
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "One or more shards failed -- NOT merging (would silently produce an incomplete file)." >&2
  echo "Fix the failure and re-run this exact command: precompute_features.py resumes from" >&2
  echo "each shard's existing chunks, so no completed work is lost." >&2
  exit 1
fi
echo "All ${NUM_GPUS} shards done."

"$PYTHON" merge_features.py --shards "${MODEL_OUT_DIR}/${CSV_STEM}.pt.shard*.pt" --out "${MODEL_OUT_DIR}/${CSV_STEM}.pt"
rm -f "${MODEL_OUT_DIR}/${CSV_STEM}".pt.shard*.pt
echo "Done -> ${MODEL_OUT_DIR}/${CSV_STEM}.pt"
