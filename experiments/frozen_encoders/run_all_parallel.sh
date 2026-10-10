#!/bin/bash
# Run all (model, video) pairs across the 4 local GPUs in parallel — one worker
# process per GPU, each restricted to its own GPU + CPU slice per
# commands4uday.sh's convention, each chewing through its shard of the job list.
#
# Usage: ./run_all_parallel.sh <videos.csv>
#   videos.csv: one video path per line. Blank lines and lines starting with
#   '#' are skipped, so a "video_path" header row is fine too. To run a
#   different set of videos, just point this at a different CSV — no need to
#   edit this script.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
source activate frozen-encoders

VIDEO_LIST=${1:?Usage: $0 <videos.csv> (one video path per line)}
MODELS=(dinov2 dinov3 vjepa2 clip siglip)
STRIDE=4
MAX_FRAMES=512

NUM_GPUS=4
CPUS_PER_GPU=16   # 64 cpus / 4 gpus

mkdir -p logs

mapfile -t VIDEOS < <(grep -v '^#' "$VIDEO_LIST" | grep -v '^\s*$' | grep -v '^video_path$')

# Build the full (model, video) job list.
JOBS=()
for model in "${MODELS[@]}"; do
  for video in "${VIDEOS[@]}"; do
    JOBS+=("${model}|${video}")
  done
done
echo "Total jobs: ${#JOBS[@]} (${#VIDEOS[@]} videos x ${#MODELS[@]} models)"

run_worker () {
  local gpu_id=$1
  local cpu_start=$((gpu_id * CPUS_PER_GPU))
  local cpu_end=$((cpu_start + CPUS_PER_GPU - 1))
  local i=0
  for job in "${JOBS[@]}"; do
    # Static round-robin sharding: this worker only handles every NUM_GPUS-th job.
    if (( i % NUM_GPUS == gpu_id )); then
      IFS='|' read -r model video <<< "$job"
      echo "[gpu ${gpu_id}] ${model} on $(basename "$video")"
      CUDA_VISIBLE_DEVICES=${gpu_id} taskset -c ${cpu_start}-${cpu_end} \
        python run.py --video "$video" --models "$model" \
          --stride ${STRIDE} --max-frames ${MAX_FRAMES} --pooling mean_patch --method tsne \
          >> logs/gpu${gpu_id}.log 2>&1
      status=$?
      if [ $status -ne 0 ]; then
        echo "[gpu ${gpu_id}] FAILED: ${model} on $(basename "$video") (see logs/gpu${gpu_id}.log)"
      fi
    fi
    i=$((i+1))
  done
  echo "[gpu ${gpu_id}] worker done"
}

for gpu in $(seq 0 $((NUM_GPUS-1))); do
  run_worker "$gpu" &
done
wait
echo "All ${#JOBS[@]} jobs done."
