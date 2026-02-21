python -u -m main \
    mode=fid_eval \
    sampling.steps=64 \
    sampling.guid_weight=1.0 \
    sampling.predictor=ancestral_cache \
    data=cifar10 \
    data.cache_dir=<YOUR-CACHE-PATH> \
    model=unet \
    noise=cosine \
    algo=mdlm \
    algo.backbone=unet \
    loader.eval_batch_size=500 \
    eval.checkpoint_path=<PATH-TO-THE-MDLM-CHECKPOINT>
