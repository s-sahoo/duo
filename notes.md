# Eval PPL
  python main.py mode=ppl_eval eval.checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/text-diff-mulan-v2-scalar-owt-not-tZycWP-small-param-subs_data-openwebtext-split_seqlen-1024_maxs-1300001_bs-512/checkpoints/last.ckpt data=openwebtext-split parameterization=subs   model.length=1024  sampling.predictor=ddpm_cache   trainer.val_check_interval=1000 time_conditioning=False sampling.steps=1000 loader.eval_batch_size=4 sampling.num_sample_batches=1 sampling.semi_ar=True sampling.stride_length=512 sampling.num_strides=0 wandb=null

# Eval samples
  python main.py mode=sample_eval eval.checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/text-diff-mulan-v2-scalar-owt-not-tZycWP-small-param-subs_data-openwebtext-split_seqlen-1024_maxs-1300001_bs-512/checkpoints/last.ckpt data=openwebtext-split parameterization=subs   model.length=1024  sampling.predictor=ddpm_cache   trainer.val_check_interval=1000 time_conditioning=False sampling.steps=1000 loader.eval_batch_size=1 sampling.num_sample_batches=10 

# Eval semi-ar samples
  python main.py mode=sample_eval eval.checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/text-diff-mulan-v2-scalar-owt-not-tZycWP-small-param-subs_data-openwebtext-split_seqlen-1024_maxs-1300001_bs-512/checkpoints/last.ckpt data=openwebtext-split parameterization=subs   model.length=1024  sampling.predictor=ddpm_cache   trainer.val_check_interval=1000 time_conditioning=False sampling.steps=1000 loader.eval_batch_size=1 sampling.num_sample_batches=10 sampling.semi_ar=True sampling.stride_length=512 sampling.num_strides=2 wandb=null

  ## lm1b ablations
    * [subs cont] python main.py mode=ppl_eval eval.checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/text-diff-abl-subs-IYzSHq-small-param-subs_data-lm1b_seqlen-128_maxs-5000000_bs-512/checkpoints/4-200000.ckpt data=lm1b parameterization=subs   sampling.predictor=subs training.importance_sampling=False training.antithetic_sampling=True time_conditioning=False wandb=null model.length=128 model.n_blocks=7 seed=0
    * [subs T=1000] python main.py mode=ppl_eval eval.checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/text-diff-abl-subs-IYzSHq-small-param-subs_data-lm1b_seqlen-128_maxs-5000000_bs-512/checkpoints/4-200000.ckpt data=lm1b parameterization=subs   sampling.predictor=subs training.importance_sampling=False training.antithetic_sampling=True time_conditioning=False eval.generate_samples=False wandb.name=eval_lm1b_abl_subs_3 model.length=128 model.n_blocks=7 T=1000 seed=0
    * [d3pm] python main.py mode=ppl_eval eval.checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/text-diff-abl-d3pm-ZO5WVr-small-param-d3pm_data-lm1b_seqlen-128_maxs-5000000_bs-512/checkpoints/3-200000.ckpt data=lm1b parameterization=d3pm   sampling.predictor=subs training.importance_sampling=False eval.generate_samples=False training.antithetic_sampling=True time_conditioning=False wandb.name=eval_lm1b_abl_d3pm_with_copy model.length=128 model.n_blocks=7 T=1000 seed=0 
    * [d3pm l_rec] python main.py mode=ppl_eval eval.checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/text-diff-abl-d3pm-ldQnAo-small-param-d3pm_data-lm1b_seqlen-128_maxs-5000000_bs-512/checkpoints/3-200000.ckpt data=lm1b parameterization=d3pm   sampling.predictor=subs training.importance_sampling=False eval.generate_samples=False training.antithetic_sampling=True time_conditioning=False wandb.name=eval_lm1b_abl_d3pm_with_copy model.length=128 model.n_blocks=7 T=1000 seed=0 


# branch: mulan
  commit: var minimization
  notes: variance minimization by minimizing the L^2 loss.

# branch: mulan-and-other-schedules
  current working branch.


