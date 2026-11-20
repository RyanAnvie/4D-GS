# -*- coding: utf-8 -*-
"""Beta opacity parameterization for Gaussian measurement confidence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BetaOpacityParameters:
    """Container for opacity mean, concentration, Beta shape, and variance."""

    mean: torch.Tensor
    concentration: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    variance: torch.Tensor


def inverse_softplus(value: torch.Tensor | float, eps: float = 1e-6) -> torch.Tensor:
    """Numerically stable inverse of softplus for positive concentration init."""
    value_t = torch.as_tensor(value, dtype=torch.float32)
    value_t = torch.clamp(value_t, min=eps)
    return value_t + torch.log(-torch.expm1(-value_t))


def concentration_from_log(opacity_logconc: torch.Tensor, kappa_min: float = 1.0) -> torch.Tensor:
    """Map unconstrained concentration logits to positive Beta concentration."""
    return F.softplus(opacity_logconc) + float(kappa_min)


def beta_opacity_parameters(
    opacity_logit: torch.Tensor,
    opacity_logconc: torch.Tensor,
    kappa_min: float = 1.0,
) -> BetaOpacityParameters:
    """Compute Beta opacity statistics from mean and concentration logits."""
    mean = torch.sigmoid(opacity_logit)
    concentration = concentration_from_log(opacity_logconc, kappa_min=kappa_min)
    alpha = torch.clamp(mean * concentration, min=1e-6)
    beta = torch.clamp((1.0 - mean) * concentration, min=1e-6)
    variance = mean * (1.0 - mean) / torch.clamp(concentration + 1.0, min=1e-6)
    return BetaOpacityParameters(mean=mean, concentration=concentration, alpha=alpha, beta=beta, variance=variance)


def model_beta_parameters(model, idx: torch.Tensor | None = None, kappa_min: float | None = None) -> BetaOpacityParameters:
    """Read Beta opacity statistics from a model with optional active indices."""
    if hasattr(model, "get_beta_params"):
        return model.get_beta_params(idx=idx, kappa_min=kappa_min)
    if not hasattr(model, "opacity"):
        raise AttributeError("model must expose opacity or get_beta_params().")
    opacity_logit = model.opacity if idx is None else model.opacity[idx]
    if hasattr(model, "opacity_logconc"):
        opacity_logconc = model.opacity_logconc if idx is None else model.opacity_logconc[idx]
    else:
        opacity_logconc = torch.zeros_like(opacity_logit)
    return beta_opacity_parameters(opacity_logit, opacity_logconc, kappa_min=1.0 if kappa_min is None else kappa_min)

