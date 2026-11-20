# -*- coding: utf-8 -*-
"""Render Gaussian scalar reliability fields through the existing rasterizer."""

from __future__ import annotations

from typing import Dict

import torch

from data_io import CameraInfo
from renderer import build_rasterizer, exposure_sample_offsets, rasterize_gaussians

from .beta import model_beta_parameters


def render_scalar_exposure_integrated(
    model,
    cam: CameraInfo,
    scalar_values: torch.Tensor,
    exposure_samples: int = 5,
    use_exposure: bool = True,
    temporal_window: float | None = None,
    temporal_sigma: float | None = None,
    znear: float = 0.01,
    zfar: float = 100.0,
    exposure_rule: str = "legacy_endpoints",
    shutter_reference_mode: str = "sample_nearest",
) -> Dict[str, torch.Tensor | None]:
    """Alpha-composite one per-Gaussian scalar into a 2D map."""
    rasterizer = build_rasterizer(cam, 0, znear=znear, zfar=zfar)
    scalar_values = scalar_values.reshape(-1, 1).to(cam.image.device)
    offsets = exposure_sample_offsets(cam.exposure if use_exposure else 0.0, exposure_samples,
                                     cam.image.device, rule=exposure_rule)
    if shutter_reference_mode not in ("sample_nearest", "center_nearest"):
        raise ValueError(f"Unknown shutter reference selection: {shutter_reference_mode}")
    selection_timestamp = None
    if model.temporal_mode == "nearest_frame" and shutter_reference_mode == "center_nearest":
        selection_timestamp = cam.timestamp
    maps = []
    alphas = []
    for offset in offsets:
        params = model.get_render_params(
            float(cam.timestamp + offset.item()),
            temporal_window=temporal_window,
            temporal_sigma=temporal_sigma,
            selection_timestamp=selection_timestamp,
        )
        idx = params["active_indices"]
        scalar_rgb = scalar_values[idx].expand(-1, 3)
        params["colors_precomp"] = scalar_rgb
        params["shs"] = None
        image, _, _, alpha = rasterize_gaussians(rasterizer, params)
        maps.append(image[:1])
        if alpha is not None:
            alphas.append(alpha)
    return {
        "map": torch.stack(maps, dim=0).mean(dim=0),
        "alpha": torch.stack(alphas, dim=0).mean(dim=0) if alphas else None,
    }


def render_confidence_map(model, cam: CameraInfo, **kwargs) -> Dict[str, torch.Tensor | None]:
    """Render log concentration as a confidence map."""
    params = model_beta_parameters(model)
    confidence = torch.log(params.concentration.clamp(min=1e-6))
    return render_scalar_exposure_integrated(model, cam, confidence, **kwargs)


def render_uncertainty_map(model, cam: CameraInfo, **kwargs) -> Dict[str, torch.Tensor | None]:
    """Render Beta opacity variance as an uncertainty map."""
    params = model_beta_parameters(model)
    return render_scalar_exposure_integrated(model, cam, params.variance, **kwargs)

