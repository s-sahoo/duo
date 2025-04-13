#!/bin/bash
#SBATCH -J distil                     # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                  # server memory requested (per node)
#SBATCH -t 96:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov          # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

export HYDRA_FULL_ERROR=1
distil=kl-bwd

srun python -u -m main \
  mode=train \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  data=lm1b-wrap \
  model=small \
  model.length=128 \
  algo=distillation \
  training.finetune_path=/share/kuleshov/ssahoo/flow-ode/flow-ode-6eTwW0-small-udlm-lm1b-wrap/checkpoints/72-1000000.ckpt \
  sampling.num_sample_batches=10 \
  hydra.run.dir=/share/kuleshov/ssahoo/flow-ode/6eTwW0-distil7-$distil \
  sampling.steps=32 \
  eval.compute_generative_perplexity=True \
  noise=gaussian-linear  \
  algo.loss_type=$distil \
  algo.T=32 \
  lr_scheduler.num_warmup_steps=500 \
  trainer.val_check_interval=1000 \
  trainer.max_steps=35000 \
  training.ema=0.99 \
  algo.grow_dt_every=5000 \
  optim.lr=1e-4 \
  training.loss_precision='float64' \
  wandb.name=6eTwW0-distil7-lr-$distil