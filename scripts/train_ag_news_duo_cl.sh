#!/bin/bash
#SBATCH -J duo-ag-news               # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=sun               # Request partition
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1 # Single A6000 GPU
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# DUO on ag_news, single A6000
python -u -m main \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  data=ag_news \
  wandb.name=duo-ag-news-cl \
  model=small \
  algo=duo \
  model.length=128 \
  algo.curriculum.mode=simple \
  algo.curriculum.gumbel_tau_log10_start=-3.0 \
  algo.curriculum.gumbel_tau_log10_end=-3.0 \
  algo.curriculum.gamma_min=-3.5 \
  algo.curriculum.gamma_max=-1.75 \
  algo.curriculum.start=0 \
  algo.curriculum.end=500000