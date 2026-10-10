#!/bin/bash
# Train ProceduralWorldModel (train.py) given a config yaml, on the cluster.
#
# No `conda activate` — non-interactive PBS shells don't source ~/.bashrc's conda-init
# hook. Calling the env's python by full path sidesteps that. (Same convention as
# external/chirality_in_action/LiFT/train.sh.)
#
# No checkpoint-dir resolution or logging setup here: train.py resolves its own
# checkpoint-dir (resolve_checkpoint_dir, auto-suffixing _1/_2/... past a previous run's
# leftovers) and writes <checkpoint-dir>/train.log itself (see train.py's _Tee) -- this
# script used to duplicate that resolution just to know where to point an external `tee`
# before train.py even started, which is exactly what let the two independently-timed
# resolutions disagree. Nothing here needs to know the checkpoint-dir at all now.
#
# Usage: ./train.sh <config.yaml> [extra train.py args...]
#   ./train.sh configs/cls_mlp_k400.yaml
#   ./train.sh configs/cls_mlp_k400.yaml --checkpoint-dir checkpoints/run2 --epochs 200
#
# Multi-GPU: NUM_GPUS=N (default 1) launches N data-parallel processes with torchrun, one per
# GPU on this node. --batch-size is then PER GPU. Request the GPUs from PBS to match, e.g.
#   export NUM_GPUS=4; qsub -v CONFIG,EXTRA_ARGS,NUM_GPUS -l select=1:ncpus=32:mem=300gb:ngpus=4 train.pbs
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON=/home/sachish/.conda/envs/lift/bin/python

CONFIG=${1:?Usage: $0 <config.yaml> [extra train.py args...]}
shift

NUM_GPUS=${NUM_GPUS:-1}
if [ "$NUM_GPUS" -gt 1 ]; then
    exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
        train.py --config "$CONFIG" "$@"
fi
exec "$PYTHON" train.py --config "$CONFIG" "$@"
