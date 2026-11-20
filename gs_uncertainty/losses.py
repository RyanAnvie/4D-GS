# -*- coding: utf-8 -*-
"""Uncertainty losses for photogrammetry-guided 4DGS."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .beta import model_beta_parameters


def _target_tensor(target, reference: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
    target_t = torch.as_tensor(target, dtype=reference.dtype, device=reference.device).reshape(-1, 1)
    return target_t if idx is None else target_t[idx]


def kappa_evidence_loss(model, target_kappa, idx: torch.Tensor | None = None, beta: float = 0.1) -> torch.Tensor:
    """Train Beta concentration against photogrammetry evidence in log-space."""
    params = model_beta_parameters(model, idx=idx)
    target = torch.as_tensor(target_kappa, dtype=params.concentration.dtype, device=params.concentration.device).reshape(-1, 1)
    if idx is not None and target.shape[0] != params.concentration.shape[0]:
        target = _target_tensor(target, params.concentration, idx)
    if target.shape[0] != params.concentration.shape[0]:
        raise ValueError("target_kappa must match the selected Gaussian count or the full model count.")
    target = torch.clamp(target, min=1e-6)
    return F.smooth_l1_loss(torch.log(params.concentration.clamp(min=1e-6)), torch.log(target), beta=beta)


def beta_concentration_prior_loss(model, prior_kappa: float = 3.0, idx: torch.Tensor | None = None) -> torch.Tensor:
    """Weak prior that prevents concentration from drifting when evidence is absent."""
    params = model_beta_parameters(model, idx=idx)
    target = torch.full_like(params.concentration, float(prior_kappa))
    return F.mse_loss(torch.log(params.concentration.clamp(min=1e-6)), torch.log(target))


def beta_kl_loss(
    model,
    prior_mu: float = 0.1,
    prior_kappa: float = 2.0,
    idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL(Beta(alpha,beta) || Beta(alpha0,beta0)) weak prior from the research plan."""
    params = model_beta_parameters(model, idx=idx)
    prior_mu_t = torch.as_tensor(prior_mu, dtype=params.alpha.dtype, device=params.alpha.device).clamp(1e-6, 1.0 - 1e-6)
    prior_kappa_t = torch.as_tensor(prior_kappa, dtype=params.alpha.dtype, device=params.alpha.device).clamp(min=1e-6)
    alpha0 = prior_mu_t * prior_kappa_t
    beta0 = (1.0 - prior_mu_t) * prior_kappa_t

    alpha = params.alpha.clamp(min=1e-6)
    beta = params.beta.clamp(min=1e-6)
    log_b_prior = torch.lgamma(alpha0) + torch.lgamma(beta0) - torch.lgamma(alpha0 + beta0)
    log_b_post = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    digamma_sum = torch.digamma(alpha + beta)
    kl = (
        log_b_prior
        - log_b_post
        + (alpha - alpha0) * (torch.digamma(alpha) - digamma_sum)
        + (beta - beta0) * (torch.digamma(beta) - digamma_sum)
    )
    return kl.mean()
