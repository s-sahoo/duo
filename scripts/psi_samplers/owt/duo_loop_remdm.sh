# DUO psi-sampler with constant-remdm-eta mode (ReMDM loop)

NUM_STEPS=256
ETA=0.01
NUCLEUS_P=0.95
NOISE=log-linear
CHECKPOINT_PATH=???
DATA_CACHE_DIR=???
EVAL_BATCH_SIZE=16
NUM_SAMPLE_BATCHES=32

python -u -m main \
    mode=sample_eval \
    data=openwebtext-split \
    data.cache_dir=$DATA_CACHE_DIR \
    model=small \
    algo=duo_base \
    noise=$NOISE \
    sampling.predictor=psi \
    sampling.steps=$NUM_STEPS \
    sampling.p_nucleus=$NUCLEUS_P \
    sampling.num_sample_batches=$NUM_SAMPLE_BATCHES \
    eval.checkpoint_path=$CHECKPOINT_PATH \
    loader.eval_batch_size=$EVAL_BATCH_SIZE \
    sampling.psi.time_profile=linear-constant-linear-0.9-inv \
    sampling.psi.high_mode=pure-posterior \
    sampling.psi.middle_mode=constant-remdm-$ETA \
    sampling.psi.low_mode=pure-posterior \
    sampling.psi.high_frac=0.45 \
    sampling.psi.middle_frac=0.5
