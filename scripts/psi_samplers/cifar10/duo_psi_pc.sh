# DUO psi-sampler with constant kappa in pc phase
# Kappa controls the posterior/PC mix: 1 = pure posterior, 0 = pure PC
#   t in [1.0, 0.5] -> pure-posterior
#   t in [0.5, 0.1] -> constant-kappa
#   t in [0.1, 0.0] -> pure-posterior

NUM_STEPS=256
KAPPA=0.95
NOISE=cosine
HIGH_FRAC=0.5
MIDDLE_FRAC=0.4
CHECKPOINT_PATH=<PATH-TO-DUO-CHECKPOINT>
DATA_CACHE_DIR=<YOUR-CACHE-PATH>
EVAL_BATCH_SIZE=500

python -u -m main \
    mode=fid_eval \
    data=cifar10 \
    data.cache_dir=$DATA_CACHE_DIR \
    model=unet \
    algo=duo_base \
    algo.backbone=unet \
    noise=$NOISE \
    sampling.predictor=psi \
    sampling.steps=$NUM_STEPS \
    sampling.guid_weight=1.0 \
    eval.checkpoint_path=$CHECKPOINT_PATH \
    loader.eval_batch_size=$EVAL_BATCH_SIZE \
    sampling.psi.time_profile=linear \
    sampling.psi.high_mode=pure-posterior \
    sampling.psi.middle_mode=constant-$KAPPA \
    sampling.psi.low_mode=pure-posterior \
    sampling.psi.high_frac=$HIGH_FRAC \
    sampling.psi.middle_frac=$MIDDLE_FRAC
