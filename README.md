# [The Diffusion Duality](http://arxiv.org/abs/2406.07524)
By [Subham Sekhar Sahoo](https://s-sahoo.github.io), [Justin Deschenaux](https://mariannearriola.github.io), [Aaron Gokaslan](https://skylion007.github.io),
[Guanghan Wang](https://tech.cornell.edu/people/guanghan-wang/), [Justin Chiu](https://justinchiu.netlify.app), [Volodymyr Kuleshov](https://www.cs.cornell.edu/~kuleshov/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Sf7R-dqdR6gq-H8nyZ9E3ZkyvqMTqcwq?usp=sharing)
[![deploy](https://img.shields.io/badge/Blog%20%20-8A2BE2)](http://s-sahoo.github.io/duo)
[![arXiv](https://img.shields.io/badge/arXiv-2406.07524-red.svg)](https://openreview.net/forum?id=CB0Ub2yXjC)
[![deploy](https://img.shields.io/badge/🤗-Huggingface-blue)](https://huggingface.co/collections/s-sahoo/duo-67f9ff8fde919224e5fbd875)

![graphical_abstract_updated_2](https://github.com/s-sahoo/duo/blob/gh-pages/static/images/duo_schematic.png)

In this repo, we release:
* **The DUO framework.**
  1. Curriculum learning strategy to speed up training.
  2. Discrete Consistency Distillation pipeline.
  3. Greedy-tail sampler.
* **Baseline implementations** [[Examples]](#baselines):
  1. Autoregressive Model.
  2. [MDLM](https://arxiv.org/abs/2406.07524): Sahoo et al., "Simple and Effective Masked Diffusion Language Model", NeurIPS 2025.
  3. [SEDD (absorb)](https://arxiv.org/abs/2310.16834): Lou et al., "Score Entropy Based Discrete Diffusion", ICML 2025.
  4. [D3PM (absorb)](https://arxiv.org/abs/2107.03006) Austin et al., "Structured Denoising Diffusion Models in Discrete State-Spaces", NeurIPS 2021.

<a name="code-organization"></a>
## Code Organization
1. ```main.py```: The main entry point for training / eval.
2. ```trainer_base.py```: Boiler plate trainer using pytorch lightning.
3. ```algo.py```: Algorithms such as DUO, MDLM, AR, SEDD, D3PM.
4. ```dataloader.py```: Dataloaders.
5. ```utils.py```: LR scheduler, logging, `fsspec` handling.
6. ```models/```: Denoising network architectures. Supports [DiT](https://arxiv.org/abs/2212.09748) and AR transformer.
7. ```configs/```: Config files for datasets/denoising networks/noise schedules/LR schedules.
8. ```scripts/```: Shell scripts for training/evaluation.


<a name="getting_started"></a>

## Getting started in this repository

To get started, create a conda environment containing the required dependencies.

```bash
conda env create -n duo
conda activate duo
conda install nvidia/label/cuda-12.4.0::cuda-toolkit
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1
```

Create the following directory to store saved models and slurm logs:
```bash
mkdir watch_folder
```
and run the training as a batch job:
```bash
sbatch scripts/train_owt_duo.sh
```

### Checkpoints

We have uploaded the DUO models (distilled/undistilled) trained on OpenWebText for 1M training steps to the Huggingface hub 🤗:
[kuleshov-group/mdlm-owt](https://huggingface.co/subbham/duo)
Finetuning from these checkpoints, from the HF checkpoints might not be possible; hence, the same checkpoints have been released on [Google Drive folder](https://drive.google.com/drive/folders/1JpqFM8XRvifwIkjWPfMyuDvu41r1yk0t?usp=share_link).

## Training
```
TODO: how to create the integral cache
```

Use the below command to train the DUO model from scratch on OpenWebText.
We also provide sample `slurm` scripts for DUO with curriculum learning on LM1B [`scripts/train_lm1b_duo.sh`](./scripts/train_lm1b_duo.sh) and OWT [`scripts/train_owt_duo.sh`](./scripts/train_owt_duo.sh).


```
python main.py \
  trainer.max_steps=1000000 \
  model=small \
  data=openwebtext-split \
  wandb.name=mdlm-owt \
  parameterization=subs \
  model.length=1024 \
  sampling.steps=1000
```
The arguments `loader.batch_size` and `loader.eval_batch_size` allow you to control the batch size per GPU. If `loader.batch_size * num_gpus` is less than the global_batch_size, PyTorch Lightning will resort to gradient accumulation. You can also launch a training job on Slurm using the command: `sbatch scripts/train_owt_mdlm.sh`. The slurm scripts to train the AR, MDLM, SEDD-absorb baselines can be found in the [`scripts`](scripts/) directory.

TODO: NB Curriculum Learning often increases the memory consumption

## Eval 
To compute test perplexity, use `mode=ppl_eval`. Example scripts provided in `scripts/eval_owt_*.sh`. An example command for perplexity evaluation on OpenWebText is:
```
python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  model=small \
  parameterization=subs \
  backbone=dit \
  model.length=1024 \
  eval.checkpoint_path=/path/to/checkpoint/mdlm.ckpt \
  +wandb.offline=true
```


## Generate Samples
<a name="sample-gen"></a>

To generate samples from a pre-trained model use one of the following command.
Set 
* `sampling.noise_removal=greedy` to use the Greedy-tail sampler (equivalent to nucleus sampling in AR models; see `Sec. 4.2` in the paper).
* `sampling.noise_removal=ancestral` for the standard ancestral sampling. This produces samples with worse generative perplexity but higher entropy.

#### Huggingface model
We have realease the distilled model `s-sahoo/duo-distilled` and the un-distilled model `s-sahoo/duo` on [Huggingface🤗](https://huggingface.co/collections/s-sahoo/duo-67f9ff8fde919224e5fbd875).
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
#### Local checkpoint
We’ve released checkpoints for the distilled `duo-distilled.ckpt` and the un-distilled model `duo.ckpt` trained on OWT here: [Google Drive folder](https://drive.google.com/drive/folders/1JpqFM8XRvifwIkjWPfMyuDvu41r1yk0t?usp=share_link). Download them and use the command below to generate samples.
```bash
python -u -m main \
  mode=sample_eval \
  loader.batch_size=2 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  model.length=1024  \
  algo=duo_base \
  model=small \
  eval.checkpoint_path=/path/to/duo-distilled.ckpt \
  sampling.num_sample_batches=1 \
  sampling.steps=8 \
  sampling.noise_removal=greedy \
  +wandb.offline=true 
```


## Baselines
<a name="baselines"></a>
We release the checkpoints for the baselines: SEDD, MDLM and AR trained on OpenWebText in this [Google Drive folder](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau?usp=sharing). Download the checkpoints: `ar.ckpt`, `mdlm.ckpt`, `sedd.ckpt` and specify the paths appropriately in [scripts/](scripts/) for computing eval ppl (`scripts/eval_owt_*.sh`), text samples (`scripts/gen_ppl_*.sh`), zero shot ppl (`scripts/zero_shot_*.sh`).

### Acknowledgements
This repository was built off of [MDLM](https://github.com/kuleshov-group/mdlm).


## Citation
```
@inproceedings{
sahoo2025the,
title={The Diffusion Duality},
author={Subham Sekhar Sahoo and Justin Deschenaux and Aaron Gokaslan and Guanghan Wang and Justin T Chiu and Volodymyr Kuleshov},
booktitle={ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy},
year={2025},
url={https://openreview.net/forum?id=CB0Ub2yXjC}
}
```
