#!/bin/bash
#SBATCH -J sample_ar                  # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

checkpoint_path=/share/kuleshov/ssahoo/flow-ode/6eTwW0-distil-kl-bwd-32/checkpoints/0-1000.ckpt
steps=32

export HYDRA_FULL_ERROR=1

srun python -u -m main \
  mode=sample_eval \
  loader.batch_size=2 \
  loader.eval_batch_size=64 \
  data=lm1b-wrap \
  algo=duo_base \
  model=small \
  model.length=128 \
  eval.checkpoint_path=/share/kuleshov/ssahoo/flow-ode/6eTwW0-distil7-kl-bwd/checkpoints/last.ckpt \
  sampling.num_sample_batches=15 \
  sampling.steps=$steps \
  +wandb.offline=true \
  sampling.noise_removal=greedy
