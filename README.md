# The Diffusion Duality Series

This repository contains the code for the two papers in the Diffusion Duality series.

## Table of Contents
- [Chapter I (ICML 2025)](#chapter-i-icml-2025)
- [Chapter II (ICLR 2026)](#chapter-ii-psi-samplers-and-efficient-curriculum-iclr-2026)
- [What's Included](#whats-included)
- [Getting Started](#getting-started)
- [Checkpoints](#checkpoints)
- [Training](#training)
- [Distillation](#distillation)
- [Sampling & Eval](#sampling--eval)
- [Baselines](#baselines)
- [Acknowledgements & Citation](#acknowledgements--citation)

## [Chapter I (ICML 2025)](https://arxiv.org/abs/2506.10892)

By [Subham Sekhar Sahoo](https://s-sahoo.github.io), [Justin Deschenaux](https://jdeschena.com), [Aaron Gokaslan](https://skylion007.github.io),
[Guanghan Wang](https://tech.cornell.edu/people/guanghan-wang/), [Justin Chiu](https://justinchiu.netlify.app), [Volodymyr Kuleshov](https://www.cs.cornell.edu/~kuleshov/)

[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github&logoColor=white)](https://github.com/s-sahoo/duo/tree/ch-1)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Sf7R-dqdR6gq-H8nyZ9E3ZkyvqMTqcwq?usp=sharing)
[![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?logo=YouTube&logoColor=white)](https://youtu.be/FCO-nnqHOqQ?si=4eGnj5zbRgyCYWwI)
[![deploy](https://img.shields.io/badge/Blog%20%20-8A2BE2)](http://s-sahoo.github.io/duo)
[![arXiv](https://img.shields.io/badge/arXiv-2406.07524-red.svg)](https://arxiv.org/abs/2506.10892)
[![deploy](https://img.shields.io/badge/🤗-Huggingface-blue)](https://huggingface.co/collections/s-sahoo/duo-67f9ff8fde919224e5fbd875)

**Unlocks few-step generation in discrete diffusion-LLMs via the underlying Gaussian diffusion.**

<div align="center">
  <img src="https://github.com/s-sahoo/duo/blob/gh-pages/static/images/duo_schematic.png" width="60%">
</div>

## [Chapter II: $\Psi$-Samplers and Efficient Curriculum (ICLR 2026)](https://openreview.net/forum?id=RSIoYWIzaP)
By  [Justin Deschenaux](https://jdeschena.com), [Caglar Gulcehre](https://www.caglar.ai),
[Subham Sekhar Sahoo](https://s-sahoo.github.io)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uFSzrfG0KXhGcohRIfWIM2Y7V9Q7cQNA?usp=sharing)
[![deploy](https://img.shields.io/badge/Blog%20%20-8A2BE2)](http://s-sahoo.github.io/duo-ch2)
[![arXiv](https://img.shields.io/badge/arXiv-2406.07524-red.svg)](https://openreview.net/forum?id=RSIoYWIzaP)
<!-- [![deploy](https://img.shields.io/badge/🤗-Huggingface-blue)](https://huggingface.co/collections/s-sahoo/duo-67f9ff8fde919224e5fbd875) -->

**Uniform-state beats Masked diffusion on text and image generation!**

# What's Included

This repo contains:
* **The Duo / $\text{Duo}^\text{++}$ framework**
  1. Sampling with ancestral, ReMDM and $\Psi$-samplers [Example](#sampling) 
  2. Curriculum learning strategy to speed up training. [[Example]](#training)
  3. Discrete Consistency Distillation pipeline. [[Example]](#distillation)
  4. Greedy-tail sampler. [[Example]](#sampling)
* **Baseline implementations** [[Examples]](#baselines):
  1. Autoregressive Model.
  2. [MDLM](https://arxiv.org/abs/2406.07524): Sahoo et al., "Simple and Effective Masked Diffusion Language Model", NeurIPS 2024.
  3. [SEDD (absorb)](https://arxiv.org/abs/2310.16834): Lou et al., "Score Entropy Based Discrete Diffusion", ICML 2024.
  4. [D3PM (absorb)](https://arxiv.org/abs/2107.03006): Austin et al., "Structured Denoising Diffusion Models in Discrete State-Spaces", NeurIPS 2021.

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

# Getting Started
<a name="getting_started"></a>

To get started, create a conda environment containing the required dependencies.

```bash
conda create -n duo python=3.12
conda activate duo
conda install nvidia/label/cuda-12.4.0::cuda-toolkit
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1
```

# Checkpoints
<a name="checkpoints"></a>

* **Duo** (Language Modeling): Trained on OpenWebText for `1M` training steps (distilled / base):
  * [Huggingface](https://huggingface.co/collections/s-sahoo/duo-67f9ff8fde919224e5fbd875)🤗.
  * [Google Drive folder](https://drive.google.com/drive/folders/1JpqFM8XRvifwIkjWPfMyuDvu41r1yk0t?usp=share_link) as the HF checkpoints can't be finetuned.
* **Duo** (Image Modeling): Trained on CIFAR-10
  * [Huggingface (contains the raw checkpoints)](https://huggingface.co/jdeschena/duo2-cifar10)
* **Baselines** (SEDD, MDLM, AR): Trained on OpenWebText
  * [Google Drive folder](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau?usp=sharing) — download `ar.ckpt`, `mdlm.ckpt`, `sedd.ckpt`.

# Training
<a name="training"></a>

This repo implements the original Duo curriculum, as well as the fast $\text{Duo}^\text{++}$ curriculum. By default, the training scripts use the original curriculum. To enable the efficient curriculum, simply replace `algo.curriculum.mode=simple` by `algo.curriculum.mode=poly9` (see comments in each training script).

To train $\text{Duo}^\text{++}$, use the following scripts:
* LM1B
  * w/ sentencepacking (same as in D3PM)
    * Training script: [`scripts/train_lm1b_duo_sentencepacking.sh`](./scripts/train_lm1b_duo_sentencepacking.sh)
    * [Wandb run](https://api.wandb.ai/links/kuleshov-group/huwt0ek3) 
  * w/o sentencepacking (same as in MDLM, SEDD)
    * Training script: [`scripts/train_lm1b_duo.sh`](./scripts/train_lm1b_duo.sh)
    * [Wandb run](https://api.wandb.ai/links/sahoo-diffusion/lkv5z3tm)
    
* OWT: [`scripts/train_owt_duo.sh`](./scripts/train_owt_duo.sh).
* CIFAR-10:
  * Duo: [`scripts/train_cifar10_duo_cosine.sh`](./scripts/train_cifar10_duo_cosine.sh)
  * MDLM: [`scripts/train_cifar10_mdlm_cosine.sh`](./scripts/train_cifar10_mdlm_cosine.sh)
  * Both scripts default to a cosine noise schedule. To use log-linear instead, set `noise=log-linear`.

**Notes:**
* Run `mkdir watch_folder` to create a directory to store slurm logs,
  and then run any script in [`scripts/`](scripts) as a slurm job: `sbatch scripts/ABC_XYZ.sh`
* Control the batch size per GPU using the argument `loader.batch_size`. If `loader.batch_size * num_gpus < loader.global_batch_size`, PyTorch Lightning resorts to gradient accumulation. 

# Distillation
<a name="distillation"></a>

To distill a model using the Discrete Consistency Distillation (`Alg. 1` in the Duo paper), use [`scripts/distil_owt.sh`](scripts/distil_owt.sh).


# Sampling & Eval
<a name="sampling"></a>

## Likelihood
To compute test perplexity on the validation set of OWT use [`scripts/eval_owt_duo.sh`](scripts/eval_owt_duo.sh) and for zero shot perplexities use [`scripts/zero_shot_duo.sh`](scripts/zero_shot_duo.sh).

## Sampling
You can sample with ancestral sampling using the scripts in [`scripts/gen_ppl_*.sh`](scripts/). To sample with the PC samplers such as ReMDM and our $\Psi$-samplers, use the scripts in [`scripts/psi_samplers`](scripts/psi_samplers). This directory contains examples for sampling text and images.

To use the "Greedy-tail sampler" (equivalent to nucleus sampling in AR models; see `Sec. 4.2` in the paper), set `sampling.noise_removal=greedy`. Using the default `sampling.noise_removal=ancestral` will produce more diverse samples (higher entropy) but with worse generative perplexity.

To sample from a HuggingFace model, run the following command:
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

To use the example scripts with raw checkpoints (see [Checkpoints](#checkpoints)), download them and set the checkpoint path in the script.


# Baselines
<a name="baselines"></a>
Download the baseline checkpoints (see [Checkpoints](#checkpoints)) and specify the paths appropriately in the respective shell scripts:
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

@inproceedings{
    deschenaux2026the,
    title={The Diffusion Duality, Chapter {II}: \${\textbackslash}Psi\$-Samplers and Efficient Curriculum},
    author={Justin Deschenaux and Caglar Gulcehre and Subham Sekhar Sahoo},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=RSIoYWIzaP}
}
```
