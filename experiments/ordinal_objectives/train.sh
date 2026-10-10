#!/bin/bash
# Train one objective (train.py) given a config yaml, on the cluster.
#
# No `conda activate` -- non-interactive PBS shells don't source ~/.bashrc's conda-init
# hook, so the env's python is called by full path instead. Same convention as
# ../predictive_world_model/train.sh and LiFT's own train.sh.
#
# No logging setup here either: train.py resolves its own checkpoint-dir (auto-suffixing
# _1/_2/... past a previous run's leftovers) and writes <checkpoint-dir>/train.log itself,
# so nothing outside it needs to know where that directory ended up.
#
# Usage: ./train.sh <config.yaml> [extra train.py args...]
#   ./train.sh configs/rank_n_contrast.yaml
#   ./train.sh configs/tcc.yaml --checkpoint-dir checkpoints/tcc_t0.1 \
#       --objective-kwargs '{"temperature": 0.1}'
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON=/home/sachish/.conda/envs/lift/bin/python

CONFIG=${1:?Usage: $0 <config.yaml> [extra train.py args...]}
shift

exec "$PYTHON" train.py --config "$CONFIG" "$@"
