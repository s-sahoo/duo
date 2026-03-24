#!/bin/bash
#SBATCH -J duo-wikitext2               # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=sun          # Request partition
#SBATCH --nodelist=sun-compute-03
#SBATCH --constraint="a6000"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Single A6000 GPU
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# To enable preemption re-loading, set `hydra.run.dir` or
# `checkpointing.save_dir` explicitly.
# Single A6000: batch_size=64 (seq_len=1024, GPT-2 vocab)
# global_batch_size=512 with accumulate_grad_batches=8
# Default model.length=1024 , but requires too many GPUs
python -u -m main \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  data=wikitext2 \
  wandb.name=duo-wikitext2-cl \
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
