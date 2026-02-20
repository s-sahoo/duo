python -u -m main \
    data=cifar10 \
    data.cache_dir=<YOUR-CACHE-PATH> \
    model=unet \
    algo=duo_base \
    algo.backbone=unet \
    noise=cosine \
    loader.global_batch_size=128 \
    loader.batch_size=128 \
    loader.eval_batch_size=128 \
    loader.num_workers=8 \
    trainer.val_check_interval=2500 \
    trainer.max_steps=1_500_000 \
    lr_scheduler.num_warmup_steps=5000 \
    eval.generate_samples=False \
    optim.lr=2e-4 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
    wandb.name=duo_base_1_5M_d3pm_like_cosine \
    hydra.run.dir=./outputs/cifar10/duo_base_1_5M_d3pm_like_cosine \

