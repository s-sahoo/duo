"""Console logger utilities.

Copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
"""

import argparse
import logging
import os
import pickle
import time
from typing import Optional

import fsspec
import lightning
import numpy as np
import torch
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.stats import norm
from timm.scheduler import CosineLRScheduler


def count_parameters(model):
  return sum(p.numel()
             for p in model.parameters()
             if p.requires_grad)

def fsspec_exists(filename):
  """Check if a file exists using fsspec."""
  fs, _ = fsspec.core.url_to_fs(filename)
  return fs.exists(filename)


def fsspec_listdir(dirname):
  """Listdir in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
  """Mkdirs in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  fs.makedirs(dirname, exist_ok=exist_ok)


def print_nans(tensor, name):
  if torch.isnan(tensor).any():
    print(name, tensor)


class LRHalveScheduler:
  def __init__(self, warmup_steps, n_halve_steps):
    self.warmup_steps = warmup_steps
    self.n_halve_steps = n_halve_steps
  
  def __call__(self, current_step):
    if current_step < self.warmup_steps:
      return current_step / self.warmup_steps
    return 0.5 ** ((current_step - self.warmup_steps)
                   // self.n_halve_steps)


class CosineDecayWarmupLRScheduler(
  CosineLRScheduler,
  torch.optim.lr_scheduler._LRScheduler):
  """Wrap timm.scheduler.CosineLRScheduler
  Enables calling scheduler.step() without passing in epoch.
  Supports resuming as well.
  Adapted from:
    https://github.com/HazyResearch/hyena-dna/blob/main/src/utils/optim/schedulers.py
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._last_epoch = -1
    self.step(epoch=0)

  def step(self, epoch=None):
    if epoch is None:
      self._last_epoch += 1
    else:
      self._last_epoch = epoch
    # We call either step or step_update, depending on
    # whether we're using the scheduler every epoch or every
    # step.
    # Otherwise, lightning will always call step (i.e.,
    # meant for each epoch), and if we set scheduler
    # interval to "step", then the learning rate update will
    # be wrong.
    if self.t_in_epochs:
      super().step(epoch=self._last_epoch)
    else:
      super().step_update(num_updates=self._last_epoch)


class LoggingContext:
  """Context manager for selective logging."""
  def __init__(self, logger, level=None, handler=None, close=True):
    self.logger = logger
    self.level = level
    self.handler = handler
    self.close = close

  def __enter__(self):
    if self.level is not None:
      self.old_level = self.logger.level
      self.logger.setLevel(self.level)
    if self.handler:
      self.logger.addHandler(self.handler)

  def __exit__(self, et, ev, tb):
    if self.level is not None:
      self.logger.setLevel(self.old_level)
    if self.handler:
      self.logger.removeHandler(self.handler)
    if self.handler and self.close:
      self.handler.close()


class GradientInspectionCallback(lightning.Callback):
    def __init__(self, num_grads_log):
        self.num_grads_log = 10

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
      gradients = []
      for name, param in pl_module.backbone.blocks.named_parameters():
          gradients.append(param.grad.view(-1))

      if gradients:
        grads = torch.cat((gradients))
        if not hasattr(pl_module, 'grad_accum_buffer'):
          pl_module.grad_step = torch.tensor(
            0, device=pl_module.device)
          pl_module.grad_accum_buffer = torch.zeros(
            self.num_grads_log,
            grads.shape[0],
            device=pl_module.device)
        pl_module.grad_accum_buffer[pl_module.grad_step] = grads
        pl_module.grad_step += 1

      if (hasattr(pl_module, 'grad_accum_buffer') 
          and pl_module.grad_step == self.num_grads_log):
        grads = pl_module.grad_accum_buffer
        grad_var = grads.std(0).mean()
        pl_module.log(name='trainer/grad_var',
                      value=grad_var.item(),
                      on_step=True,
                      on_epoch=False,
                      sync_dist=True)
        # import ipdb; ipdb.set_trace()
        # should save the grads tensor as a numpy array
        # and visualize mean, median, top-k
        pl_module.grad_accum_buffer.zero_()
        pl_module.grad_step = 0


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
  """Initializes multi-GPU-friendly python logger."""

  logger = logging.getLogger(name)
  logger.setLevel(level)

  # this ensures all logging levels get marked with the rank zero decorator
  # otherwise logs would get multiplied for each GPU process in multi-GPU setup
  for level in ('debug', 'info', 'warning', 'error',
                'exception', 'fatal', 'critical'):
    setattr(logger,
            level,
            lightning.pytorch.utilities.rank_zero_only(
              getattr(logger, level)))

  return logger


# Copied from https://github.com/jdeschena/sdtt/blob/bbc54d5b3c5fcffd79602cff17ed34dde1f3eff6/src/sdtt/core/sampling/utils.py#L10
def top_k_top_p_filtering(
    logits,
    top_k=0,
    top_p=0.0,
    filter_value=-float("Inf"),
    dim=-1):
    """Filter a distribution of logits using top-k/top-p (nucleus) filtering.
    Adapted from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

    Args:
      logits (Tensor): Tensor of logits
      top_k (int, optional): Number of top values to keep.
          Deactivated if k is 0. Defaults to 0.
      top_p (float, optional): Cumulative mass to retain.
          Deactivated if p = 0. Defaults to 0.0.
      filter_value (float, optional): Fill value to replace
          the entries removed by top-k/top-p filtering.
          Defaults to -float('Inf').
      dim (int, optional): Dimension of the filtering. Defaults to -1.

    Returns:
        logits: Tensor whose axis `dim` was filtered.
    """
    if dim != -1:
      logits = torch.transpose(logits, dim, -1)

    assert top_k < logits.size(dim)
    if top_k > 0:
      # Remove all tokens with a probability less than
      # the last token of the top-k
      values, _ = torch.topk(logits, k=top_k, dim=-1)
      to_remove_mask = (
          logits < torch.min(values, dim=-1, keepdim=True)[0]
      )  # min returns a tuple (values, indices)
      logits[to_remove_mask] = filter_value

    if top_p > 0.0:
      sorted_logits, sorted_indices = torch.sort(
        logits, descending=True, dim=-1)
      cum_probs = torch.cumsum(
        torch.softmax(sorted_logits, dim=-1), dim=-1)

      sorted_indices_to_remove = cum_probs > top_p
      # Ensures at least one token is kept
      sorted_indices_to_remove[..., 1:] = \
        sorted_indices_to_remove[..., :-1].clone()
      sorted_indices_to_remove[..., 0] = 0

      mask_to_remove = torch.empty_like(
        sorted_indices_to_remove)
      mask_to_remove.scatter_(dim=-1,
                              index=sorted_indices,
                              src=sorted_indices_to_remove)
      logits[mask_to_remove] = filter_value

    if dim != -1:
      logits = torch.transpose(logits, dim, -1)
    
    # Re-normalize as in ReMDM. Alternatively, 
    #  could apply a log-softmax. This assumes that the input
    #  tensor `logits` has been pre-processed with log_softmax.
    probs = logits.exp()
    Z = probs.sum(-1, keepdim=True)
    logits = (probs / Z).log()
    return logits


def _discrete_prob_map(gamma_t, N=10):
  snr_sqrt = np.exp(-gamma_t / 2)
  def value(x):
    cdf = norm.cdf(x, scale=1) ** (N - 1)
    pdf = norm.pdf(x, loc=snr_sqrt, scale=1)
    return pdf * cdf
  return value


def _discrete_prob_grad(gamma_t, N=10):
  snr_sqrt = np.exp(-gamma_t / 2)
  def value(x):
    coef = -0.5 * snr_sqrt * (x - snr_sqrt)
    cdf = norm.cdf(x, scale=1) ** (N - 1)
    pdf = norm.pdf(x, loc=snr_sqrt, scale=1)
    return coef * pdf * cdf
  return value


def _cache_prob_usdm_in_partition(
  vocab_size=30522, partition_index=0, num_partitions=1,
  log10_num_points=5):
  print(f'Caching partition:{partition_index} / {num_partitions}')
  path = 'integral'
  gamma_min = -5
  gamma_max = -1
  num_points = 10 ** log10_num_points
  p_cache = []
  grad_p_cache = []
  start_time = time.time()
  gammas = np.linspace(gamma_min, gamma_max, num_points)
  n = num_points // num_partitions
  for gamma in gammas[partition_index * n:
                      (partition_index + 1) * n]:
    pt, _ = quad(_discrete_prob_map(gamma, vocab_size),
                 -np.inf, np.inf)
    p_cache.append(pt)
    grad_pt, _ = quad(_discrete_prob_grad(gamma, vocab_size),
                      -np.inf, np.inf)
    grad_p_cache.append(grad_pt)
    if len(p_cache) % 100 == 0:
      print('{}% completed. Time elapsed:{:.2f} mins'.format(
        int(100 * len(p_cache) / num_points),
        (time.time() - start_time) / 60))

  filename = os.path.join(
    path, '{}_{}_{}-{}.pkl'.format(
      vocab_size, log10_num_points, partition_index,
      num_partitions))
  with open(filename, 'wb') as f:
    pickle.dump({
      'vocab_size': vocab_size,
      'gamma_min': gamma_min,
      'gamma_max': gamma_max,
      'num_points': num_points,
      'pt': np.asarray(p_cache),
      'grad_pt': np.asarray(grad_p_cache)}, f)


def test_cache_prob_usdm_in_partition(
  partition_index=0, num_partitions=1, vocab_size=30522,
  log10_num_points=5):
  path = 'integral/{}_{}_{}-{}.pkl'.format(
    vocab_size, log10_num_points, partition_index,
    num_partitions)
  with open(path, 'rb') as f:
    data = pickle.load(f)
  num_points = data['num_points']
  def _get_index(x):
    return round((num_points - 1) * (x - data['gamma_min']) / (
      data['gamma_max'] - data['gamma_min']))

  pt_errors = []
  grad_pt_errors = []
  gammas = np.linspace(data['gamma_min'],
                       data['gamma_max'],
                       num_points)
  n = num_points // num_partitions
  for gamma in gammas[partition_index * n:
                      (partition_index + 1) * n]:
    pt, _ = quad(
      _discrete_prob_map(gamma, data['vocab_size']),
      -np.inf, np.inf)
    grad_pt, _ = quad(
      _discrete_prob_grad(gamma, data['vocab_size']),
      -np.inf, np.inf)
    idx = _get_index(gamma)
    print(idx)
    pt_errors.append((pt - data['pt'][idx]) ** 2)
    grad_pt_errors.append((grad_pt - data['grad_pt'][idx]) ** 2)
  print('Integral MSE:{} Integral Squared:{:.4f}'.format(
    np.mean(pt_errors), np.mean(data['pt'] ** 2)))
  print('Integral Grad MSE:{} Integral Grad Squared:{:.4f}'.format(
    np.mean(grad_pt_errors), np.mean(data['grad_pt'] ** 2)))


def compute_duo_series_coefficients(num_coefficients, 
                                    vocab_size):
  def integrand_m(z, n, K):
      z = np.float64(z)
      return z**n * norm.pdf(z) * norm.cdf(z)**(K-1)
    
  def integrand_i(z, n, K):
    z = np.float64(z)
    return z ** (n+1) * norm.pdf(z) * norm.cdf(z) ** (K-1)
  
  arange = np.cumprod(np.arange(1, num_coefficients
                                ).astype(np.float64))
  factorials = np.concatenate([[1.0], arange], 
                              dtype=np.float64)
  lo = np.array([-100], dtype=np.float64)
  hi = np.array([100], dtype=np.float64)
  coefficients_m = []
  coefficients_i = []

  for n in range(num_coefficients):
    f = lambda z: integrand_m(np.float64(z), np.float64(n), 
                              vocab_size)
    g = lambda z: integrand_i(np.float64(z), np.float64(n), 
                              vocab_size)
    m, _ = quad(f, lo, hi)
    i, _ = quad(g, lo, hi)
    coefficients_m.append(m / factorials[n])
    coefficients_i.append(i / factorials[n])

  return (torch.tensor(coefficients_m)[None], 
          torch.tensor(coefficients_i)[None])


def compute_duo_gamma_to_alpha_dalpha_series(
    gamma_t, coefficients_m, coefficients_i, power_arange, 
    vocab_size, gamma_min, gamma_max):
  gamma_t = gamma_t.to(torch.float64)[:, None]
    
  sigmoid_neg_gamma = torch.sigmoid(-gamma_t)
  sigmoid_gamma = torch.sigmoid(gamma_t)
  alpha_t_squared = sigmoid_neg_gamma
  alpha_t = alpha_t_squared.sqrt()

  one_minus_alpha_t_squared = 1 - alpha_t_squared
  mu_t = alpha_t / one_minus_alpha_t_squared.sqrt()
  
  arange = power_arange.to(device=gamma_t.device)
  mu_t_pow = mu_t ** arange
  
  exp_term = (-mu_t**2/2).exp().squeeze(-1)
  vocab_scale = vocab_size / (vocab_size - 1)
  
  # Compute alpha
  sum_term_alpha = (mu_t_pow * coefficients_m).sum(-1)
  alpha_usdm = (sum_term_alpha * exp_term - 1 \
              / vocab_size) * vocab_scale

  # Compute alpha'
  sum_term_dalpha = (
    mu_t_pow * (coefficients_i - mu_t * coefficients_m)).sum(-1)
  dalpha_usdm = exp_term * sum_term_dalpha * vocab_scale
  
  final_scale = - (sigmoid_gamma.squeeze(-1) 
                  * sigmoid_neg_gamma.squeeze(-1) ** 0.5 * 
                  0.5 * (gamma_max - gamma_min))
  dalpha_usdm = dalpha_usdm \
              / one_minus_alpha_t_squared.squeeze(-1) ** 1.5 \
              * final_scale
  return alpha_usdm.squeeze(-1), dalpha_usdm.squeeze(-1)


def duo_t_to_alpha_dalpha_sigm_corrected(
  t, a: float, b: float, c: float, d: float, e: float, 
  f: float, alpha: float):
  # Shared quantities
  sigm_bc = (torch.tanh(b * t + c) + 1) / 2
  sigm_ef = (torch.tanh(e * t + f) + 1) / 2

  # Compute alpha_t
  base = a * sigm_bc + d
  edge_gate = 1 - 4 * sigm_ef * (1 - sigm_ef)
  edge_correction = alpha * (t - 0.5) * edge_gate
  alpha_t = base + edge_correction

  # Compute d_alpha_t
  dbase = a * b * sigm_bc * (1 - sigm_bc)
  dgate = -4 * e * sigm_ef * (1 - sigm_ef) * (1 - 2 * sigm_ef)
  dcorrection = alpha * edge_gate + alpha * (t - 0.5) * dgate
  dalpha_t = dbase + dcorrection
  return alpha_t, dalpha_t


def duo_to_alpha_dalpha_sigmoid(t: torch.Tensor, a: float, 
                                b: float, c: float, d: float):
  sigm_bc = (torch.tanh(b * t + c) + 1) / 2
  alpha_t = a * sigm_bc + d
  dalpha_t = a * b * sigm_bc * (1 - sigm_bc)
  return alpha_t, dalpha_t


def duo_to_alpha_dalpha_poly(t: torch.Tensor, 
                              *coefficients: float):
  alpha_t = coefficients[0]  # a0 term
  for i, a in enumerate(coefficients[1:], 1):
    alpha_t = alpha_t + a * t**i
    
  dalpha_t = coefficients[1]  # a1 term
  for i, a in enumerate(coefficients[2:], 2):
    dalpha_t = dalpha_t + i * a * t**(i-1)
    
  return alpha_t, dalpha_t


def compute_duo_operator_approx(num_coefficients, vocab_size, 
                                gamma_min, gamma_max, 
                                fct_name='sigmoid'):
  series_m, series_i = compute_duo_series_coefficients(
    num_coefficients, vocab_size)
  ts = torch.linspace(0, 1, steps=100_000)
  gammas = gamma_min + ts * (gamma_max - gamma_min)
  power_arange = torch.arange(num_coefficients, 
                              dtype=torch.float64)[None]
  alpha_approx = compute_duo_gamma_to_alpha_dalpha_series(
    gammas, series_m, series_i, power_arange, vocab_size, 
    gamma_min, gamma_max)[0].float()
  
  t_np = ts.numpy()
  y_np = alpha_approx.numpy()

  def sigmoid(x):
    return (np.tanh(x) + 1) / 2
  
  if fct_name == 'sigmoid':
    def func(t, a, b, c, d):
      return a * sigmoid(b * t + c) + d
    p0 = [0.5, 2.0, -1.0, 0.5]

  elif fct_name == 'sigmoid-edge-corrected':
    def func(t, a, b, c, d, e, f, alpha):
      base = a * sigmoid(b * t + c) + d
      edge_gate = 1 - 4 * sigmoid(e * t + f) \
                  * sigmoid(-e * t - f)
      edge_correction = alpha * (t - 0.5) * edge_gate
      return base + edge_correction
    p0 = [0.5, 2.0, -1.0, 0.1, 3.0, 0.0, 0.1]
  elif fct_name == 'poly3':
    def func(t, a0, a1, a2, a3):
      return a0 + a1*t + a2*t**2 + a3*t**3
    p0 = [0.1] * 4
  elif fct_name == 'poly5':
    def func(t, a0, a1, a2, a3, a4, a5):
      return a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    p0 = [0.1] * 6
  elif fct_name == 'poly7':
    def func(t, a0, a1, a2, a3, a4, a5, a6, a7):
      return (a0 + a1*t + a2*t**2 + a3*t**3 + 
              a4*t**4 + a5*t**5 + a6*t**6 + a7*t**7)
    p0 = [0.1] * 8
  elif fct_name == 'poly9':
    def func(t, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9):
      return (a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + 
              a5*t**5 + a6*t**6 + a7*t**7 + a8*t**8 + a9*t**9)
    p0 = [0.1] * 10
  else:
    raise ValueError(fct_name)
  
  popt, _ = curve_fit(func, t_np, y_np, p0=p0, maxfev=10000)
  preds = func(t_np, *popt)
  return list(popt), y_np, preds, t_np


def _sample_k_int(bs: int, l: int, k: int, max_value: int, 
                 device: torch.device):
  # Robert Floyd's algorithm: 
  #  https://www.nowherenearithaca.com/2013/05/robert-floyds-tiny-and-beautiful.html
  out = torch.empty(size=(bs, l, k), dtype=torch.int64, 
                    device=device)
  for t, i in enumerate(range(max_value - k, max_value)):
    j = torch.randint(0, i + 1, size=(bs, l), device=device)
    if t > 0:
      # Does j already appear in previously chosen slots?
      dup = (out[..., :t] == j[..., None]).any(dim=-1)
      # write j where it is new, otherwise write i
      out[..., t] = torch.where(dup, i, j)
    else:
      out[..., 0] = j
  return out


def _sample_topk_gaussian(N: int, 
  sigma: Optional[torch.Tensor]=None, l: int=0, k: int=0, 
  batch: int=None, device: str=None, 
  dtype: torch.dtype=torch.float64):
  """
  Sample from the order statistic of N Gaussians with zero
  mean (top k). Operate in logspace for stability.
  """
  if sigma is None:
    assert batch is not None
    assert device is not None
    assert dtype is not None
  else:
    batch = sigma.shape[0]
    device = sigma.device
    dtype = sigma.dtype
  log_u = torch.log(torch.rand(batch, l, k, device=device, 
                               dtype=dtype))
  divisors = torch.arange(N, N - k, -1, device=device, 
                          dtype=dtype)  # (k,)
  log_rj = log_u / divisors  # (batch, l, k)
  log_prod = torch.cumsum(log_rj, dim=-1)  # (batch, l, k)
  uniforms = torch.exp(log_prod)  # (batch, l, k)
  # convert to Gaussian and rescale
  topk = torch.special.ndtri(uniforms)
  if sigma is not None:
    topk = topk * sigma[:, None, None]
  return topk


def _sample_topk_and_extra(N: int, alpha: torch.Tensor, 
  sigma: torch.Tensor, l: int, k: int):
  """
  Sample the top k order statistics between N - 1 zero mean 
  Gaussians, and a single Gaussian with mean alpha.
  """
  top_k_others = _sample_topk_gaussian(N - 1, sigma, l, k)
  extra = alpha[:, None] + torch.randn(
    size=(alpha.shape[0], l), device=alpha.device
    ) * sigma[:, None]  # (bs, l)
  min_values = top_k_others[:, :, -1]
  is_extra_in_topk = (extra > min_values)  # bs x l
  top_k_others[:, :, -1][is_extra_in_topk] = extra[is_extra_in_topk]
  return extra, top_k_others, is_extra_in_topk


def _log_mean_exp_trunc_normal(c: torch.Tensor, 
                              sigma: torch.Tensor):
  """
  Compute log(E[exp(X) | X < c] for X ~ N(0, sigma^2).
  Closed-form expression:
    mu = exp(sigma**2 / 2) 
         * Phi((c - sigma**2) / sigma) 
         / Phi(c / sigma)
  where Phi is the standard normal CDF. Operate in log space
  for stability.
  """
  log_num = torch.special.log_ndtr((c - sigma**2) / sigma)
  log_den = torch.special.log_ndtr(c / sigma)
  return sigma**2 / 2.0 + log_num - log_den


def sample_tempered_softmax_topk(
  extra_index: torch.Tensor, 
  alpha: torch.Tensor, 
  sigma: torch.Tensor, 
  l: int, 
  k: int, 
  vocab_size: int,
  # 1 / T. If low temperature, inverse will be large
  inverse_temperature: float = 1.0):
  assert alpha.ndim == 1
  assert sigma.ndim == 1
  # float64 needed for numerical precision
  alpha = alpha.to(torch.float64)
  sigma = sigma.to(torch.float64)
  # Sample the top k between (vocab_size - 1) zero-mean
  #  Gaussians, and a single Gaussian with mean alpha.
  extra, top_k, is_extra_in_topk = _sample_topk_and_extra(
    vocab_size, alpha, sigma, l, k)
  min_rv = torch.where(is_extra_in_topk, top_k[:, :, -2],
                         top_k[:, :, -1])  # (bs, l)
  
  scaled_sigma = sigma[:, None] * inverse_temperature  # (bs, 1)
  scaled_c = min_rv * inverse_temperature  # (bs, l)

  log_mu = _log_mean_exp_trunc_normal(scaled_c, scaled_sigma)
  log_topk = top_k * inverse_temperature

  # Approximate contribution of the unknown variables
  count = torch.where(is_extra_in_topk, vocab_size - k,
            vocab_size - k - 1).to(log_mu.dtype)
  log_tail = torch.log(count) + log_mu

  # Contribution of the extra variable, when NOT in the topk
  log_extra = extra * inverse_temperature
  extra_not_in_topk = ~is_extra_in_topk
  log_extra_masked = torch.full_like(log_tail, float('-inf'))
  log_extra_masked[extra_not_in_topk] = \
                                log_extra[extra_not_in_topk]

  log_contribs = torch.cat([
      log_topk, 
      log_tail[..., None], 
      log_extra_masked[..., None]], 
    dim=-1)  # (bs, l, k+2)
  log_denom = torch.logsumexp(log_contribs, dim=-1, 
                              keepdim=True)  # (bs, l, 1)
  softmax_approx = torch.exp(log_topk - log_denom)  # (bs, l, k)
  # If sum over k categories is zero, just set to one-hot on 
  #  the largest. Fix div by zero
  normalizer = softmax_approx.sum(dim=-1, keepdim=True)
  zero_sum = normalizer == 0.0
  softmax_approx = torch.where(zero_sum, 0.0, softmax_approx)
  softmax_approx[..., 0][zero_sum[..., 0]] = 1.0
  
  indices = _sample_k_int(alpha.shape[0], l, k, 
                         # Note the -1:
                         max_value=vocab_size - 1,
                         device=alpha.device)
  # Ensure x0 (true token) is not generated
  indices[indices >= extra_index[..., None]] += 1
  indices[..., -1][is_extra_in_topk] = \
                              extra_index[is_extra_in_topk]
  xt_usdm = torch.where(is_extra_in_topk, extra_index, 
                        indices[..., 0])
  return softmax_approx, indices, xt_usdm


if __name__ == "__main__":
  # Usage: python utils.py --vocab_size=N
  parser = argparse.ArgumentParser(
    description='Caches the integral appearing in the '
                'Diffusion Transformation operator.')
  parser.add_argument(
    '--vocab_size',
    type=int,
    default=50257,  # For the gpt2 tokenizer
    help='Vocabulary size (default: 50257)')
  parser.add_argument(
    '--partition_index',
    type=int,
    default=0,
    help='Helps parallelize caching')
  parser.add_argument(
    '--num_partitions',
    type=int,
    default=1,
    help='Helps parallelize caching')
  parser.add_argument(
    '--log10_num_points',
    type=int,
    default=5,
    help=('The integral is function that needs to be '
          'evaluated for inputs with a range [-5, 1]. '
          'This argument represents the logarithm base 10 '
          'of number of bins of discretization.'))
  args = parser.parse_args()

  # Computing the integral over [-5, 1] can be slow,
  # so one might prefer splitting it into `num_partitions`
  # bins and compute each separately and merge them later.
  _cache_prob_usdm_in_partition(
    partition_index=args.partition_index,
    num_partitions=args.num_partitions,
    vocab_size=args.vocab_size,
    log10_num_points=args.log10_num_points)
  
  test_cache_prob_usdm_in_partition(
    partition_index=args.partition_index,
    num_partitions=args.num_partitions,
    vocab_size=args.vocab_size,
    log10_num_points=args.log10_num_points)