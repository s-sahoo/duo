import torch
from omegaconf import OmegaConf

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import algo
import dataloader

"""
# Instructions on running the eval

## What is the script doing?
The script evaluates (or approximates) the log-likelihood of
prefix + suffix. The script will load a checkpoint, and 
depending on the config, use the corresponding class 
(e.g. mdlm, duo, ar, etc).

- For MCQ, the log-likelihood of all continuations given the 
  same prefix is evaluated. The most likely continuation is 
  selected as the "correct" answer according to the model.

- For lambada_openai, the model is correct if the true 
  continuation is generated as the argmax. For diffusion, we 
  inject noise in the continuation, and check whether the true 
  answer computed in a single forward pass is the most likely. 
  This is naturally favoring AR models, as they run one forward 
  pass per token, while diffusion use a single pass for all 
  tokens for simplicity. Therefore, I usually only compare on 
  MCQ since it is more fair.

To run the script, you need to install the lm-eval-harness package:
```
pip install git+https://github.com/EleutherAI/lm-evaluation-harness
```

## Tasks that we tested with
**MCQ**: arc_easy, arc_challenge, hellaswag, winogrande, 
         boolq, openbookqa, race, social_iqa, mathqa, piqa
**Likelihood based**: lambada_openai

## Batch size that fits at the small scale
    - boolq -> 64
    - openbookqa -> 256
    - race -> 32
    - social_iqa -> 512
    - winogrande -> 64
    - mathqa -> 64
    - lambada_openai -> 64
    - arc_easy -> 256
    - arc_challenge -> 256
    - hellaswag -> 64
    - piqa -> 64

## Important flags
  --trust_remote_code -> some datasets execute code when loading 
                         from huggingface. Without this flag, 
                         the script might crash.
  --batch_size        -> max. num elements to use in parallel 
                         to eval the likelihood (for diffusion). 
                         For simplicity, inputs are NOT padded 
                         and batched for AR, though it should 
                         be fairly easy to add.
  --tasks             -> one or multiple comma-separated tasks 
                         to evaluate on.
  --model_args        -> string (without spaces) that contains 
                         the arguments to pass to the evaluator 
                         (stuff like checkpoints path, number 
                         of MC samples to evaluate the 
                         likelihood, path to the sentencepiece 
                         tokenizer, etc).
  --output_path       -> path to a json file where the evaluation 
                         results will be saved, instead of 
                         only being printed in the terminal 
                         (they will always be printed).
  --limit             -> debugging flag; limit the number of 
                         examples to a fixed amount, instead 
                         of using the whole dataset


## Example commands


### Run a single task with an AR model
python discrete_diffusion_harness.py \
    --tasks arc_easy \
    --batch_size 256 \
    --model dlm \
    --model_args checkpoint_path=/home/username/baselines/ar/1M.ckpt \
    --output_path ./harness_results/ar/1M/arc_easy.ckpt

### Run a single task with an MDLM model, and 2048 MC samples to approximate the likelihood
python discrete_diffusion_harness.py \
    --tasks arc_easy \
    --model dlm \
    --batch_size 256 \
    --model_args checkpoint_path=/home/username/baselines/mdlm/1M.ckpt,num_mc_samples=2048 \
    --output_path ./harness_results/mdlm/1M/arc_easy.ckpt

### Debug with 20 examples only
python discrete_diffusion_harness.py \
    --tasks arc_easy \
    --model dlm \
    --batch_size 256 \
    --model_args checkpoint_path=/home/username/baselines/mdlm/1M.ckpt,num_mc_samples=2048 \
    --limit 20 \
    --output_path ./harness_results/mdlm/1M/arc_easy.ckpt


### Run the benchmarks from "The Diffusion Duality (Chapter 2)":
    (Arc-e, Arc-c, HSwag, WinoG, PIQA, MathQA, OQA)
-> just change the checkpoint to evaluate mdlm, ar, or duo

for task_config in "arc_easy 256" "arc_challenge 256" "hellaswag 64" "winogrande 64" "piqa 64" "mathqa 64" "openbookqa 256"; do
  task=$(echo $task_config | cut -d' ' -f1)
  batch_size=$(echo $task_config | cut -d' ' -f2)
  python discrete_diffusion_harness.py \
    --batch_size $batch_size \
    --tasks $task \
    --model dlm \
    --model_args checkpoint_path=/path/to/checkpoint.ckpt \
    --output_path ./harness_results/duo/$task.json
done

"""


def requests_to_dataset(config, requests, tokenizer, num_proc):
  def _tokenize(e):
    eos_idx = tokenizer.eos_token_id
    bos_idx = tokenizer.bos_token_id
    prefix_tokens = tokenizer(e['prefix'], 
                              return_attention_mask=False, 
                              add_special_tokens=False
                              )['input_ids']
    target_tokens = tokenizer(e['target'], 
                              return_attention_mask=False, 
                              add_special_tokens=False
                              )['input_ids']
    prefix_tokens = [bos_idx] + prefix_tokens
    target_tokens = target_tokens + [eos_idx]
    
    return {
        'prefix_text': e['prefix'],
        'target_text': e['target'],
        'prefix': prefix_tokens,
        'target': target_tokens,
    }
  ds = []
  ds = [{'prefix': req.args[0], 'target': req.args[1]} 
        for req in requests]
  ds = Dataset.from_list(ds)
  ds = ds.map(_tokenize, num_proc=num_proc)
  ds = ds.with_format('torch')
  seq_lenths = [len(x['prefix']) + len(x['target']) 
                for x in ds]
  
  num_larger = len([x for x in seq_lenths 
                    if x > config.model.length])
  if num_larger > 0:
    print(f'\033[91mThere are some examples that are longer '
          f'than the context length, they will be ignored '
          f'during evaluation. Number of such sequences: '
          f'{num_larger}\033[0m')

  return ds


def _eval_suffix_nll_generators(config, module, prefix, 
  suffix, batch_size, num_samples, loss_avg_mode):
  device = module.device
  assert num_samples % batch_size == 0
  full_sentence = torch.cat([prefix, suffix], dim=-1
                  ).repeat(batch_size, 1).to(module.device)
  all_ts = module._sample_t(num_samples, accum_step=None)
  for idx in range(0, num_samples, batch_size):
    t = all_ts[idx:idx+batch_size].unsqueeze(-1)
    dalpha_t, alpha_t = module.noise(t)
    alpha_t = alpha_t.to(device)
    sigma = module._sigma_from_alphat(alpha_t)
    x0 = full_sentence.to(device)
    # Inject noise
    xt = module.q_xt(full_sentence, alpha_t).to(device)
    if loss_avg_mode == 'full':
      pass  # nothing to do
    elif loss_avg_mode == 'suffix':
      xt[:, :len(prefix)] = prefix
      # recompute alpha_t based on number of masked tokens, 
      #   for conditioning of the backbone
      alpha_t = (xt == x0).float().mean(dim=1)[:, None]
      t = module.noise.get_t_for_alpha(alpha_t)
      # We need to get dalpha_t for the loss:
      alpha_t = alpha_t.to(device)
      sigma = module._sigma_from_alphat(alpha_t)
    else:
      raise ValueError(loss_avg_mode)

    yield xt, x0, t, sigma, alpha_t, dalpha_t
    

def eval_suffix_nll(config, module, prefix, suffix, batch_size, 
                    num_samples, loss_avg_mode):
  if config.algo.name in ('mdlm', 'duo', 'duo_base', 
                          'distillation', 'ot-finetune'):
    return eval_suffix_nll_diffusion(config, module, prefix, 
      suffix, batch_size, num_samples, loss_avg_mode)
  elif config.algo.name == 'ar':
    return eval_suffix_nll_ar(config, module, prefix, 
      suffix, batch_size, num_samples, loss_avg_mode)
  else:
    raise ValueError(config.algo.name)


def eval_suffix_nll_ar(config, module, prefix, suffix,
                    batch_size, num_samples, loss_avg_mode):
  x_cat = torch.cat([prefix, suffix[:-1]], dim=-1)
  x_cat = x_cat.reshape(1, -1).to(module.device)

  with torch.amp.autocast('cuda', dtype=torch.float32):
    out = module.backbone(x_cat, sigma=None)

  out[:, :, module.mask_index] = module.neg_infinity
  suffix_out = out[:, len(prefix) - 1:, :]
  suffix_logits = torch.log_softmax(suffix_out, dim=-1)
  index = suffix[None, :, None].to(module.device)
  nll = torch.gather(-suffix_logits, dim=-1, 
                     index=index).mean()
  return float(nll.cpu())

def eval_suffix_nll_diffusion(config, module, prefix, suffix, 
  batch_size, num_samples, loss_avg_mode):
  all_losses = []
  generator =  _eval_suffix_nll_generators(config, module, 
    prefix, suffix, batch_size, num_samples, loss_avg_mode)
  for xt, x0, t, sigma, alpha_t, dalpha_t in generator:
    log_x_theta = module(xt, sigma, labels=None)
    token_nll = module.nll_per_token(log_x_theta, xt, x0, 
                                     alpha_t, dalpha_t)
    if loss_avg_mode == 'full':
      loss = float(token_nll.mean())
    elif loss_avg_mode == 'suffix':
      loss = float(token_nll[:, len(prefix):].mean())
    all_losses.append(loss)
  return float(np.mean(all_losses))


@register_model("dlm")
class DiscreteDiffusionHarness(LM):
  def __init__(self, pretrained="NONE", max_length=1024,
    num_mc_samples=1024, batch_size=64, device="cuda",
    checkpoint_path=None, num_proc=8, loss_avg_mode='full',
    *args, **kwargs):
    super().__init__()
    # Whether to use the full sequence, or suffix only to 
    #  approximate the NLL. Full should be the correct way.
    assert loss_avg_mode in ('full', 'suffix')
    ckpt = torch.load(checkpoint_path, map_location='cpu', 
                      weights_only=False)
    config = ckpt['hyper_parameters']['config']
    # Backfill missing keys into legacy checkpoints
    if not hasattr(config.training, 'class_dropout_p'):
      OmegaConf.set_struct(config, False)
      config.training.class_dropout_p = 0.0
      OmegaConf.set_struct(config, True)
    self.tokenizer = dataloader.get_tokenizer(config)
    if config.algo.name == 'mdlm':
      self.model = algo.MDLM(config, self.tokenizer)
    elif config.algo.name in ('duo', 'duo_base', 
                              'distillation', 'ot-finetune'):
      self.model = algo.DUO_BASE(config, self.tokenizer)
    elif config.algo.name == 'ar':
      self.model = algo.AR(config, self.tokenizer)
    else:
      raise ValueError(f'Implement for {config.algo.name}')
    self.config = config
    self.num_proc = num_proc
    self.num_mc_samples = num_mc_samples
    self.batch_size = int(batch_size)
    self.device = device
    self.loss_avg_mode = loss_avg_mode

    self.model.load_state_dict(ckpt['state_dict'])
    self.model.to(device)
    self.model.eval()

  def suffix_greedy_prediction(self, prefix, target):
    if self.config.algo.name == 'mdlm':
      return self._suffix_greedy_prediction_mdlm(prefix, 
                                                 target)
    elif self.config.algo.name in ('duo', 'duo_base', 
                              'distillation', 'ot-finetune'):
      return self._suffix_greedy_prediction_duo_base(prefix, 
                                                     target)
    elif self.config.algo.name == 'ar':
      return self._suffix_greedy_prediction_ar(prefix, target)
    else:
      raise ValueError(self.config.algo.name)
    
  def _suffix_greedy_prediction_ar(self, prefix, target):
    x_cat = torch.cat([prefix, target[:-1]], 
      dim=-1).reshape(1, -1).to(self.device)

    # Follows generate_samples in AR (algo.py)
    out = self.model.backbone(x_cat, sigma=None)
    out[:, :, self.model.mask_index] = self.model.neg_infinity
    out = out.log_softmax(-1)
    preds_suffix = out[:, len(prefix) - 1:, :]
    greedy_preds = preds_suffix.argmax(-1).flatten()
    return (greedy_preds.cpu() == target).all().item()


  def _suffix_greedy_prediction_mdlm(self, prefix, target):
    mask_idx = self.model.mask_index
    eos_idx = self.tokenizer.eos_token_id
    # Note: because of the preprocessing, we know that the 
    #  last token is an eos token.
    noisy_target = [mask_idx] * (len(target) - 1) + [eos_idx]
    noisy_target = torch.tensor(noisy_target, 
                                device=self.device)
    prefix = prefix.to(self.device)
    seq = torch.concatenate([prefix, noisy_target], 
                            dim=-1).reshape(1, -1)
    sigma = torch.zeros(size=(seq.shape[0], 1), 
                        device=self.device)
    logits = self.model(seq, sigma, labels=None)
    assert logits.shape[0] == 1
    suffix_logits = logits[0, len(prefix):]
    target_preds = suffix_logits.argmax(-1).cpu()
    correct = target_preds == target
    correct = correct.all()
    return bool(correct)
  
  def _suffix_greedy_prediction_duo_base(self, prefix, target):
    noisy_suffix = torch.randint(
      0, self.model.vocab_size, size=target.shape, 
      dtype=target.dtype, device=self.device)

    prefix = prefix.to(self.device)
    # shape: (1, len)
    noisy_seq = torch.concatenate([prefix, noisy_suffix])[None, :]
    # Set the EOS token at the end
    noisy_seq[0, -1] = target[-1]
    clean_seq = torch.concatenate([prefix, target.to(self.device)])[None, :]
    
    alpha_t = (noisy_seq == clean_seq).float().mean(dim=1)[:, None]
    sigma = self.model._sigma_from_alphat(alpha_t)
    t = self.model.noise.get_t_for_alpha(alpha_t)
    logits = self.model(noisy_seq, sigma, labels=None)
    assert logits.shape[0] == 1
    suffix_logits = logits[0, len(prefix):]
    target_preds = suffix_logits.argmax(-1).cpu()

    correct = target_preds == target
    correct = correct.all()
    return bool(correct)
  
  @torch.no_grad()
  def loglikelihood(self, requests: list[Instance]) \
                                -> list[tuple[float, bool]]:
    dataset = requests_to_dataset(self.config, requests, 
                                  self.tokenizer, self.num_proc)
    out = []
    for elem in tqdm(dataset, 'Computing likelihood...'):
      prefix = elem['prefix']
      target = elem['target']
      
      if len(prefix) + len(target) > self.model.config.model.length:
        # If the request is too long, skip it.
        ll = 0.0
        is_target_greedy_dec = False
        out.append((ll, is_target_greedy_dec))
        print("SKIPPING")
        continue

      ll = -eval_suffix_nll(self.config, self.model, prefix, 
        target, self.batch_size, self.num_mc_samples, 
        self.loss_avg_mode)
      is_target_greedy_dec = self.suffix_greedy_prediction(
        prefix, target)
      out.append((ll, bool(is_target_greedy_dec)))
    return out

  def loglikelihood_rolling(
        self, requests: list[Instance]
    ) -> list[tuple[float, bool]]:
    raise NotImplementedError
  
  def generate_until(self, context, max_length, stop, 
                     **generation_kwargs):
    raise NotImplementedError


if __name__ == "__main__":
    cli_evaluate()