import torch
from typing import Optional


def sample_k_int(bs: int, l: int, k: int, max_value: int, 
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


def sample_topk_gaussian(N: int, sigma: Optional[torch.Tensor]=None, 
  l: int=0, k: int=0, batch: int=None, device: str=None, 
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


def sample_topk_and_extra(N: int, alpha: torch.Tensor, 
                          sigma: torch.Tensor, l: int, k: int):
  """
  Sample the top k order statistics between N - 1 zero mean 
  Gaussians, and a single Gaussian with mean alpha.
  """
  top_k_others = sample_topk_gaussian(N - 1, sigma, l, k)
  extra = alpha[:, None] + torch.randn(
    size=(alpha.shape[0], l), device=alpha.device
    ) * sigma[:, None]  # (bs, l)
  min_values = top_k_others[:, :, -1]
  is_extra_in_topk = (extra > min_values)  # bs x l
  top_k_others[:, :, -1][is_extra_in_topk] = extra[is_extra_in_topk]
  return extra, top_k_others, is_extra_in_topk


def log_mean_exp_trunc_normal(c: torch.Tensor, 
                              sigma: torch.Tensor):
  """
  Compute log(E[exp(X) | X < c] for X ~ N(0, sigma^2).
  Closed-form expression:
    mu = exp(sigma**2 / 2) * Phi((c - sigma**2) / sigma) / Phi(c / sigma)
  where Phi is the standard normal CDF. Operate in log space
  for stability.
  """
  log_num = torch.special.log_ndtr((c - sigma**2) / sigma)
  log_den = torch.special.log_ndtr(c / sigma)
  return sigma**2 / 2.0 + log_num - log_den

#@torch.jit.script
def sample_tempered_softmax_topk(
  extra_index: torch.Tensor, 
  alpha: torch.Tensor, 
  sigma: torch.Tensor, 
  l: int, 
  k: int, 
  vocab_size: int,
  # 1 / T. If low temperature, inverse will be large, like 1000
  inverse_temperature: float = 1.0):
  assert alpha.ndim == 1
  assert sigma.ndim == 1
  # float64 needed for numerical precision
  alpha = alpha.to(torch.float64)
  sigma = sigma.to(torch.float64)
  # Sample the top k between (vocab_size - 1) zero-mean
  #  Gaussians, and a single Gaussian with mean alpha.
  extra, top_k, is_extra_in_topk = sample_topk_and_extra(
    vocab_size, alpha, sigma, l, k)
  min_rv = torch.where(is_extra_in_topk, top_k[:, :, -2],
                         top_k[:, :, -1])  # (bs, l)
  
  scaled_sigma = sigma[:, None] * inverse_temperature  # (bs, 1)
  scaled_c = min_rv * inverse_temperature  # (bs, l)

  log_mu = log_mean_exp_trunc_normal(scaled_c, scaled_sigma)
  log_topk = top_k * inverse_temperature

  # Approximate contribution of the unknown variables
  count = torch.where(is_extra_in_topk, vocab_size - k,
            vocab_size - k - 1).to(log_mu.dtype)
  log_tail = torch.log(count) + log_mu

  # Contribution of the extra variable, when NOT in the topk
  log_extra = extra * inverse_temperature
  extra_not_in_topk = ~is_extra_in_topk
  log_extra_masked = torch.full_like(log_tail, float('-inf'))
  log_extra_masked[extra_not_in_topk] = log_extra[extra_not_in_topk]

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
  
  indices = sample_k_int(alpha.shape[0], l, k, 
                         # Note the -1:
                         max_value=vocab_size - 1,
                         device=alpha.device)
  # Ensure x0 (true token) is not generated
  indices[indices >= extra_index[..., None]] += 1
  indices[..., -1][is_extra_in_topk] = extra_index[is_extra_in_topk]
  xt_usdm = torch.where(is_extra_in_topk, extra_index, 
                        indices[..., 0])
  return softmax_approx, indices, xt_usdm