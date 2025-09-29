# [The Diffusion Duality (ICML 2025)](https://arxiv.org/abs/2506.10892)
By [Subham Sekhar Sahoo](https://s-sahoo.github.io), [Justin Deschenaux](https://jdeschena.github.io), [Aaron Gokaslan](https://skylion007.github.io),
[Guanghan Wang](https://tech.cornell.edu/people/guanghan-wang/), [Justin Chiu](https://justinchiu.netlify.app), [Volodymyr Kuleshov](https://www.cs.cornell.edu/~kuleshov/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Sf7R-dqdR6gq-H8nyZ9E3ZkyvqMTqcwq?usp=sharing)
[![deploy](https://img.shields.io/badge/Blog%20%20-8A2BE2)](http://s-sahoo.github.io/duo)
[![arXiv](https://img.shields.io/badge/arXiv-2406.07524-red.svg)](https://arxiv.org/abs/2506.10892v1)
[![deploy](https://img.shields.io/badge/🤗-Huggingface-blue)](https://huggingface.co/collections/s-sahoo/duo-67f9ff8fde919224e5fbd875)

**We unlock few-step generation in discrete diffusion language models via the underlying Guassian diffusion.**

<div align="center">
  <img src="https://github.com/s-sahoo/duo/blob/gh-pages/static/images/duo_schematic.png" width="60%">
</div>

In this repo, we release:
* **The DUO framework**
  1. Curriculum learning strategy to speed up training. [[Example]](#training)
  2. Discrete Consistency Distillation pipeline. [[Example]](#distillation)
  3. Greedy-tail sampler. [[Example]](#sampling)
* **Baseline implementations** [[Examples]](#baselines):
  1. Autoregressive Model.
  2. [MDLM](https://arxiv.org/abs/2406.07524): Sahoo et al., "Simple and Effective Masked Diffusion Language Model", NeurIPS 2024.
  3. [SEDD (absorb)](https://arxiv.org/abs/2310.16834): Lou et al., "Score Entropy Based Discrete Diffusion", ICML 2024.
  4. [D3PM (absorb)](https://arxiv.org/abs/2107.03006) Austin et al., "Structured Denoising Diffusion Models in Discrete State-Spaces", NeurIPS 2021.

<!-- <a name="code-organization"></a>
## Code Organization
1. ```main.py```: The main entry point for training / eval.
2. ```trainer_base.py```: Boiler plate trainer using pytorch lightning.
3. ```algo.py```: Algorithms such as DUO, MDLM, AR, SEDD, D3PM.
4. ```dataloader.py```: Dataloaders.
5. ```utils.py```: LR scheduler, logging, `fsspec` handling.
6. ```models/```: Denoising network architectures. Supports [DiT](https://arxiv.org/abs/2212.09748) and AR transformer.
7. ```configs/```: Config files for datasets/denoising networks/noise schedules/LR schedules.
8. ```scripts/```: Shell scripts for training/evaluation. -->


<a name="getting_started"></a>

# Getting Started

To get started, create a conda environment containing the required dependencies.

```bash
conda create -n duo python=3.12
conda activate duo
conda install nvidia/label/cuda-12.4.0::cuda-toolkit
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1
```

### :izakaya_lantern: Integral Cache [Important]
Curriculum Learning (`Sec. 4.1`) and Discrete Consistency Distillation (`Sec. 4.2`) require mapping Gaussian to discrete diffusion parameters via the Diffusion Transformation operator (`Sec. 3`), which involves computing an integral (dependent only on the tokenizer’s vocabulary size). To avoid slowing down training, we pre-compute and cache this integral. Cached operators for `bert-base-uncased` (LM1B) and `gpt2` (OWT) are in [`integral/`](integral). For other tokenizers, run: 
```
python utils.py --vocab_size=N
```
where `N` is the vocabulary size of the tokenizer.

### Checkpoints

The checkpoints for the DUO models (distilled/undistilled) trained on OpenWebText for 1M training steps are available on:
* [Huggingface](https://huggingface.co/collections/s-sahoo/duo-67f9ff8fde919224e5fbd875)🤗.
* [Google Drive folder](https://drive.google.com/drive/folders/1JpqFM8XRvifwIkjWPfMyuDvu41r1yk0t?usp=share_link) as the HF checkpoints can't be finetuned.

### Slurm scripts
Run `mkdir watch_folder` to create a directory to store slurm logs
and then run any script in [`scripts/`](scripts) as a slurm job:
```bash
sbatch scripts/ABC_XYZ.sh
```

# Training
<a name="training"></a>

To train DUO use the following scripts:
* LM1B
  * w/ sentencepacking (same as in D3PM)
    * Training script: [`scripts/train_lm1b_duo_sentencepacking.sh`](./scripts/train_lm1b_duo_sentencepacking.sh)
    * [Wandb run](https://api.wandb.ai/links/kuleshov-group/huwt0ek3) 
  * w/o sentencepacking (same as in MDLM, SEDD)
    * Training script: [`scripts/train_lm1b_duo.sh`](./scripts/train_lm1b_duo.sh)
    * [Wandb run](https://api.wandb.ai/links/sahoo-diffusion/lkv5z3tm)
    
* OWT: [`scripts/train_owt_duo.sh`](./scripts/train_owt_duo.sh).


**Curriculum Learning increases memory consumption.** For faster training on OWT, one may consider a two-stage approach:
* `Stage 1`: Curriculum Learning for `500K` steps
  * Use [`scripts/train_owt_duo.sh`](./scripts/train_owt_duo.sh) with the following modifications:
    * Reduced batch size (`loader.batch_size=32` on an `80 GB` GPU)
    * `trainer.max_steps=500000` 
* `Stage 2`: Finetuning the checkpoint from `stage 1` for `500K` more steps
  * Training script: [`scripts/train_owt_duo_finetune.sh`](scripts/train_owt_duo_finetune.sh)
  * Features a larger batch size (`loader.batch_size=64` on an `80 GB`) than `stage 1`.
  * [Wandb run](https://api.wandb.ai/links/kuleshov-group/h74aekb3): This run resumes training a `stage 1` checkpoint. Although trained for `1M` steps, the results reported in the paper correspond to the checkpoint at `500K` steps.

Control the batch size per GPU using the argument `loader.batch_size`. If `loader.batch_size * num_gpus < loader.global_batch_size`, PyTorch Lightning resorts to gradient accumulation. 

# Distillation
<a name="distillation"></a>

To distil a model using the Discrete Consisitency Distillation (`Alg. 1` in the paper), use [`scripts/distil_owt.sh`](scripts/distil_owt.sh)



# Sampling & Eval
<a name="sampling"></a>
To compute test perplexity on the validtion set of OWT use [`scripts/eval_owt_duo.sh`](scripts/eval_owt_duo.sh) and for zero shot perplexities use [`scripts/zero_shot_duo.sh`](scripts/zero_shot_duo.sh).


To generate samples from a pre-trained model use one of the following command.
Set 
* `sampling.noise_removal=greedy` to use the "Greedy-tail sampler" (equivalent to nucleus sampling in AR models; see `Sec. 4.2` in the paper).
* `sampling.noise_removal=ancestral` for the standard ancestral sampling. This produces more diverse samples (higher entropy) but with worse generative perplexity.

We have realease the distilled model `s-sahoo/duo-distilled` and the un-distilled model `s-sahoo/duo` on [Huggingface](https://huggingface.co/collections/s-sahoo/duo-67f9ff8fde919224e5fbd875)🤗. To sample from a HF model, run the following command:
```bash
python main.py \
  mode=sample_eval \
  loader.batch_size=2 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  algo=duo_base \
  algo.backbone=hf_dit \
  eval.checkpoint_path=s-sahoo/duo-distilled \
  sampling.steps=8 \
  sampling.num_sample_batches=1 \
  sampling.noise_removal=greedy \
  +wandb.offline=true 
```
We’ve also released checkpoints for the distilled `duo-distilled.ckpt` and the un-distilled model `duo.ckpt` trained on OWT in this [Google Drive folder](https://drive.google.com/drive/folders/1JpqFM8XRvifwIkjWPfMyuDvu41r1yk0t?usp=share_link). Download them and use the command in [`scripts/gen_ppl_owt_duo.sh`](scripts/gen_ppl_owt_duo.sh) while specifying the paths correctly.


# Baselines
<a name="baselines"></a>
We release the checkpoints for the baselines: SEDD, MDLM and AR trained on OpenWebText in this [Google Drive folder](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau?usp=sharing). Download the checkpoints: `ar.ckpt`, `mdlm.ckpt`, `sedd.ckpt` and specify the paths appropriately in the respective shell scripts:
* [`scripts/eval_owt_*.sh`](scripts/) for computing validation perplexity on OWT.
* [`scripts/gen_ppl_*.sh`](scripts/) for generating text samples and evaluating them.
* [`scripts/zero_shot_*.sh`](scripts/) for computing zero shot perplexities.
* [`scripts/train_*.sh`](scripts/) for training the models.

# Acknowledgements & Citation
This repository was built off of [MDLM's Github repository](https://github.com/kuleshov-group/mdlm). Cite our paper using:
```
@inproceedings{
sahoo2025the,
title={The Diffusion Duality},
author={Subham Sekhar Sahoo and Justin Deschenaux and Aaron Gokaslan and Guanghan Wang and Justin T Chiu and Volodymyr Kuleshov},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=9P9Y8FOSOk}
}
```
