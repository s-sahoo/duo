#!/bin/bash
#SBATCH -J posterior                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov          # Request partition
#SBATCH --constraint="[a5000|a6000|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                 # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

export HYDRA_FULL_ERROR=1
finetune_path=/path/to/duo.ckpt

srun python -u -m main \
  mode=train \
  loader.batch_size=2 \
  loader.eval_batch_size=2 \
  data=openwebtext-split \
  model=small \
  algo=distillation \
  training.finetune_path=$finetune_path \
  sampling.num_sample_batches=10 \
  sampling.steps=32 \
  eval.compute_generative_perplexity=True \
  algo.T=512 \
  lr_scheduler.num_warmup_steps=500 \
  trainer.val_check_interval=1000 \
  trainer.max_steps=50000 \
  loader.global_batch_size=128 \
  training.ema=0.999 \
  algo.update_teacher_every=10000 \
  optim.lr=6e-5 \
  trainer.limit_val_batches=8 \
  algo.teacher_ema=False \
  algo.linear_growth_dt=false \
  +wandb.offline=true
