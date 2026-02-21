# MDLM psi-sampler with max-rescale-eta mode (ReMDM rescale)

NUM_STEPS=256
ETA=0.05
NUCLEUS_P=0.9
NOISE=log-linear
CHECKPOINT_PATH=/claire-rcp-scratch/home/deschena/baselines/mdlm/62-1000000.ckpt
DATA_CACHE_DIR=/claire-rcp-scratch/home/deschena/discrete_diff_base/data_cache
EVAL_BATCH_SIZE=16
NUM_SAMPLE_BATCHES=32

python -u -m main \
    mode=sample_eval \
    data=openwebtext-split \
    data.cache_dir=$DATA_CACHE_DIR \
    model=small \
    algo=mdlm \
    noise=$NOISE \
    sampling.predictor=psi \
    sampling.steps=$NUM_STEPS \
    sampling.p_nucleus=$NUCLEUS_P \
    sampling.num_sample_batches=$NUM_SAMPLE_BATCHES \
    eval.checkpoint_path=$CHECKPOINT_PATH \
    loader.eval_batch_size=$EVAL_BATCH_SIZE \
    sampling.psi.time_profile=linear \
    sampling.psi.high_mode=max-rescale-$ETA \
    sampling.psi.middle_mode=max-rescale-$ETA \
    sampling.psi.low_mode=max-rescale-$ETA \
    sampling.psi.high_frac=0.0 \
    sampling.psi.middle_frac=0.0
