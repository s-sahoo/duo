#!/bin/bash
#SBATCH -J sample_ar                  # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=16000                   # server memory requested (per node)
#SBATCH -t 24:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov,gpu          # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/text-diffusion-exp-v4-AgBZrc-small-ar-param-ar_data-openwebtext-split_seqlen-1024_maxs-1300001_bs-512/checkpoints
seed=1

export HYDRA_FULL_ERROR=1

srun python -u -m main \
  mode=sample_eval \
  seed=$seed \
  loader.batch_size=2 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  algo=ar \
  model=small \
  eval.checkpoint_path=$checkpoint_path/last.ckpt \
  sampling.num_sample_batches=100 \
  +wandb.offline=true \
  eval.generated_samples_path=$checkpoint_path/$seed-ckpt-last.json