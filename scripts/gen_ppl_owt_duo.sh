#!/bin/bash
#SBATCH -J an_owt_duo                    # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=16000                   # server memory requested (per node)
#SBATCH -t 24:00:00                   # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov,gpu      # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

export HYDRA_FULL_ERROR=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --steps) steps="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        --ckpt) ckpt="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

checkpoint_path=/share/kuleshov/ssahoo/flow-ode/distil-distil-vjrpZb-distillation-OWT/checkpoints
ckpt=0-50000

steps=${steps:-32}
seed=${seed:-1}
echo "  Steps: $steps"
echo "  Seed: $seed"
echo "  ckpt: $ckpt"

srun python -u -m main \
  mode=sample_eval \
  seed=$seed \
  loader.batch_size=2 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  algo=duo_base \
  model=small \
  eval.checkpoint_path=$checkpoint_path/$ckpt.ckpt \
  sampling.num_sample_batches=100 \
  sampling.steps=$steps \
  +wandb.offline=true \
  eval.generated_samples_path=$checkpoint_path/samples_ancestral/$seed-$steps-ckpt-$ckpt.json