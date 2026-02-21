# DUO psi-sampler with max-capped-eta mode (ReMDM cap)

NUM_STEPS=256
ETA=0.005
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
    sampling.psi.time_profile=linear \
    sampling.psi.high_mode=max-capped-$ETA \
    sampling.psi.middle_mode=max-capped-$ETA \
    sampling.psi.low_mode=max-capped-$ETA \
    sampling.psi.high_frac=0.0 \
    sampling.psi.middle_frac=0.0
