# -*- coding: utf-8 -*-
"""Rasterization and exposure integration."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from data_io import CameraInfo


def get_projection_matrix(znear, zfar, fx, fy, cx, cy, width, height, device):
    P = torch.zeros(4, 4, dtype=torch.float32, device=device)
    P[0, 0] = 2.0 * fx / width
    P[1, 1] = 2.0 * fy / height
    P[0, 2] = (2.0 * cx / width) - 1.0
    P[1, 2] = (2.0 * cy / height) - 1.0
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def build_rasterizer(cam: CameraInfo, sh_degree: int, bg=None, antialiasing: bool = False,
                     znear: float = 0.01, zfar: float = 100.0):
    device = cam.image.device
    if bg is None:
        bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    # ndc2Pix(v,S)=((v+1)*S-1)/2 in the CUDA backend. Explicit +0.5 makes
    # projected OpenCV integer pixel centers agree with image sampling.
    cx = cam.cx + cam.raster_pixel_offset
    cy = cam.cy + cam.raster_pixel_offset
    proj = get_projection_matrix(znear, zfar, cam.fx, cam.fy, cx, cy, cam.width, cam.height, device)
    full_proj = proj @ cam.w2c
    settings = GaussianRasterizationSettings(
        image_height=cam.height,
        image_width=cam.width,
        tanfovx=math.tan(cam.fovx / 2),
        tanfovy=math.tan(cam.fovy / 2),
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=cam.w2c.transpose(0, 1),
        projmatrix=full_proj.transpose(0, 1),
        sh_degree=sh_degree,
        campos=cam.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=antialiasing,
    )
    return GaussianRasterizer(raster_settings=settings)


def rasterize_gaussians(rasterizer, render_params) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    render_params = dict(render_params)
    render_params.pop("active_indices", None)
    outputs = rasterizer(**render_params)
    if not isinstance(outputs, tuple):
        return outputs, None, None, None
    color = outputs[0]
    radii = outputs[1] if len(outputs) > 1 else None
    depth = outputs[2] if len(outputs) > 2 else None
    alpha = outputs[3] if len(outputs) > 3 else None
    return color, radii, depth, alpha


def exposure_sample_offsets(exposure: float, n_samples: int, device, rule="legacy_endpoints") -> torch.Tensor:
    if n_samples < 1 or exposure < 0:
        raise ValueError("Exposure must be nonnegative and sample count positive.")
    if rule not in ("legacy_endpoints", "midpoint", "center_repeated"):
        raise ValueError(f"Unknown shutter quadrature: {rule}")
    if n_samples <= 1 or exposure <= 0.0:
        return torch.zeros(1, dtype=torch.float32, device=device)
    if rule == "center_repeated":
        # Integration-off ablation with the same positive-exposure raster budget.
        return torch.zeros(n_samples, dtype=torch.float32, device=device)
    if rule == "midpoint":
        centers = (torch.arange(n_samples, dtype=torch.float32, device=device) + 0.5) / n_samples
        return exposure * (centers - 0.5)
    return torch.linspace(-0.5 * exposure, 0.5 * exposure, n_samples, dtype=torch.float32, device=device)


def render_exposure_integrated(
    model,
    cam: CameraInfo,
    exposure_samples: int = 5,
    use_exposure: bool = True,
    retain_screen_grad: bool = False,
    temporal_window: float | None = None,
    temporal_sigma: float | None = None,
    znear: float = 0.01,
    zfar: float = 100.0,
    exposure_rule: str = "legacy_endpoints",
    shutter_reference_mode: str = "sample_nearest",
    reference_timestamp: float | None = None,
) -> Dict[str, torch.Tensor | None]:
    """The optional cohort reference does not change the camera's physical time."""
    if reference_timestamp is not None:
        if not math.isfinite(reference_timestamp):
            raise ValueError("Cohort reference time must be finite.")
        if model.temporal_mode != "nearest_frame" or shutter_reference_mode != "center_nearest":
            raise ValueError("Explicit cohort selection requires nearest_frame / center_nearest.")
    rasterizer = build_rasterizer(cam, model.active_sh_degree, znear=znear, zfar=zfar)
    exposure = cam.exposure if use_exposure else 0.0
    offsets = exposure_sample_offsets(exposure, exposure_samples, cam.image.device, rule=exposure_rule)
    if shutter_reference_mode not in ("sample_nearest", "center_nearest"):
        raise ValueError(f"Unknown shutter reference selection: {shutter_reference_mode}")
    selection_timestamp = reference_timestamp
    if selection_timestamp is None and model.temporal_mode == "nearest_frame" and shutter_reference_mode == "center_nearest":
        selection_timestamp = cam.timestamp

    images = []
    depths = []
    alphas = []
    radii_acc = None
    first_viewspace = None
    first_indices = None
    screen_samples = []
    for offset in offsets:
        params = model.get_render_params(
            float(cam.timestamp + offset.item()),
            retain_screen_grad=retain_screen_grad,
            temporal_window=temporal_window,
            temporal_sigma=temporal_sigma,
            selection_timestamp=selection_timestamp,
        )
        if first_viewspace is None:
            first_viewspace = params["means2D"]
            first_indices = params["active_indices"]
        image, radii, depth, alpha = rasterize_gaussians(rasterizer, params)
        images.append(image)
        if depth is not None:
            depths.append(depth)
        if alpha is not None:
            alphas.append(alpha)
        if radii is not None:
            if len(offsets) == 1:
                radii_acc = radii
            else:
                # Active IDs may differ between shutter samples, even at equal lengths.
                if radii_acc is None:
                    radii_acc = torch.zeros(model.n_gaussians, dtype=radii.dtype, device=radii.device)
                idx = params["active_indices"]
                radii_acc[idx] = torch.maximum(radii_acc[idx], radii)
        if retain_screen_grad:
            screen_samples.append({"indices": params["active_indices"],
                                   "viewspace": params["means2D"], "radii": radii})

    if radii_acc is not None and len(offsets) > 1:
        radii_acc = radii_acc[first_indices]

    out = {
        "image": torch.stack(images, dim=0).mean(dim=0),
        "depth": torch.stack(depths, dim=0).mean(dim=0) if depths else None,
        "alpha": torch.stack(alphas, dim=0).mean(dim=0) if alphas else None,
        "radii": radii_acc,
        "viewspace": first_viewspace,
        "active_indices": first_indices,
        "screen_samples": screen_samples,
    }
    return out
