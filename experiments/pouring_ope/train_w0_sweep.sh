#!/bin/bash
# window0 pouring: straighten_weight {0.1, 0} x seeds {0,1,2}; (0.1, seed 0) is
# pwm_pouring_patch_channel_paper_window0_vicreg_bs4, trained earlier from the same config.
PWM=/home/sachish/proc-planning/experiments/predictive_world_model
CFG=$PWM/configs/patch_channel_pouring_window0_vicreg.yaml
cd $PWM
for run in "0.1 1" "0.1 2" "0 0" "0 1" "0 2"; do
  set -- $run
  qsub -N w0_sw$1_s$2 -l select=1:ncpus=8:mem=32gb:ngpus=1 -l walltime=06:00:00 \
    -v CONFIG=$CFG,EXTRA_ARGS="--straighten-weight $1 --seed $2 --checkpoint-dir checkpoints/pwm_pouring_w0_sw$1_seed$2" train.pbs
done
