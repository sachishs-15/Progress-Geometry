#!/usr/bin/env bash
#SBATCH --job-name=proc-planning
#SBATCH --output=experiments/%x_%j/slurm.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT="$HOME/proc-planning"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

mkdir -p "experiments/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

source activate proc-planning
python -m proc_planning.training.train --config configs/example.yaml "$@"
