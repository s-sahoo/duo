export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -u -m main \
    mode=fid_eval \
    sampling.steps=64 \
    sampling.guid_weight=1.0 \
    data=cifar10 \
    data.cache_dir=<YOUR-CACHE-PATH> \
    model=unet \
    noise=cosine \
    algo=duo_base \
    algo.backbone=unet \
    trainer.num_nodes=1 \
    loader.eval_batch_size=500 \
    eval.checkpoint_path=<PATH-TO-THE-MDLM-CHECKPOINT>
