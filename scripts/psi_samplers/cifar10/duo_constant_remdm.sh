# DUO psi-sampler with constant-remdm-eta mode (ReMDM loop)

NUM_STEPS=256
ETA=0.01
NOISE=cosine
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
    sampling.psi.time_profile=linear-constant-linear-0.9-inv \
    sampling.psi.high_mode=pure-posterior \
    sampling.psi.middle_mode=constant-remdm-$ETA \
    sampling.psi.low_mode=pure-posterior \
    sampling.psi.high_frac=0.45 \
    sampling.psi.middle_frac=0.5
