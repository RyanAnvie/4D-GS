# -*- coding: utf-8 -*-
"""Dynamic Gaussian model: surface Gaussians plus linear motion."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch
from torch import nn

from gs_uncertainty.beta import beta_opacity_parameters, inverse_softplus
from ply_utils import write_vertex_ply


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 1e-6, 1.0 - 1e-6)
    return torch.log(x / (1.0 - x))


class DynamicGaussianModel(nn.Module):
    def __init__(self, init: dict, max_sh_degree: int = 0, device: str = "cuda"):
        super().__init__()
        self.max_sh_degree = int(max_sh_degree)
        self.active_sh_degree = 0
        self.device_name = device

        # CPU as_tensor can alias the caller's measurements; trainable arrays own storage.
        mu_ref = torch.as_tensor(init["mu_ref"], dtype=torch.float32, device=device).clone()
        velocity = torch.as_tensor(init["velocity"], dtype=torch.float32, device=device).clone()
        t_ref = torch.as_tensor(init["t_ref"], dtype=torch.float32, device=device).reshape(-1, 1)
        colors = torch.as_tensor(init["colors"], dtype=torch.float32, device=device)
        scales = torch.as_tensor(init["scales"], dtype=torch.float32, device=device)
        rotations = torch.as_tensor(init["rotations"], dtype=torch.float32, device=device).clone()

        n = mu_ref.shape[0]
        sh_count = (self.max_sh_degree + 1) ** 2
        sh_features = torch.zeros((n, 3, sh_count), dtype=torch.float32, device=device)
        sh_features[:, :, 0] = (colors - 0.5) / 0.28209

        self.mu_ref = nn.Parameter(mu_ref)
        self.velocity = nn.Parameter(velocity)
        self.features_dc = nn.Parameter(sh_features[:, :, 0:1].transpose(1, 2).contiguous())
        self.features_rest = nn.Parameter(sh_features[:, :, 1:].transpose(1, 2).contiguous())
        init_opacity = torch.full((n, 1), 0.1, dtype=torch.float32, device=device)
        init_confidence = torch.as_tensor(init.get("confidence", np.ones(n)), dtype=torch.float32, device=device).reshape(-1, 1)
        init_kappa = 1.0 + 30.0 * torch.clamp(init_confidence, 0.0, 1.0)
        self.opacity_logit = nn.Parameter(inverse_sigmoid(init_opacity))
        self.opacity_logconc = nn.Parameter(inverse_softplus(init_kappa - 1.0).to(device))
        self.scaling = nn.Parameter(torch.log(torch.clamp(scales, min=1e-7)))
        self.rotation = nn.Parameter(rotations)

        self.register_buffer("t_ref", t_ref)
        self.register_buffer("anchor_mu_ref", mu_ref.clone())
        self.register_buffer("anchor_velocity", velocity.clone())
        self.register_buffer("anchor_rotation", torch.nn.functional.normalize(rotations.clone(), dim=1))
        self.register_buffer("track_ids", torch.as_tensor(init["track_ids"], dtype=torch.long, device=device))
        self.register_buffer("track_times", torch.as_tensor(init["track_times"], dtype=torch.float32, device=device))
        self.register_buffer("track_points", torch.as_tensor(init["track_points"], dtype=torch.float32, device=device))
        self.register_buffer("track_mask", torch.as_tensor(init["track_mask"], dtype=torch.float32, device=device))
        frame_indices = init.get("frame_indices", np.zeros(n, dtype=np.int64))
        self.register_buffer("frame_indices", torch.as_tensor(frame_indices, dtype=torch.long, device=device))
        self.register_buffer("source_indices", torch.as_tensor(init.get("source_indices", np.arange(n)), dtype=torch.long, device=device))
        self.register_buffer("parent_ids", torch.as_tensor(init.get("parent_ids", np.full(n, -1)), dtype=torch.long, device=device))
        self.register_buffer("generation", torch.as_tensor(init.get("generation", np.zeros(n)), dtype=torch.long, device=device))
        self.mode = init.get("mode", "track")
        self.temporal_mode = init.get("temporal_mode", "legacy_window")
        if self.temporal_mode not in ("legacy_window", "nearest_frame"):
            raise ValueError(f"Unknown temporal mode: {self.temporal_mode}")

    @property
    def n_gaussians(self) -> int:
        return int(self.mu_ref.shape[0])

    @property
    def opacity(self) -> torch.Tensor:
        """Backward-compatible alias for the opacity mean logit parameter."""
        return self.opacity_logit

    def get_features(self) -> torch.Tensor:
        return torch.cat((self.features_dc, self.features_rest), dim=1)

    def get_beta_params(self, idx: torch.Tensor | None = None, kappa_min: float | None = None):
        opacity_logit = self.opacity_logit if idx is None else self.opacity_logit[idx]
        opacity_logconc = self.opacity_logconc if idx is None else self.opacity_logconc[idx]
        return beta_opacity_parameters(opacity_logit, opacity_logconc, kappa_min=1.0 if kappa_min is None else kappa_min)

    def get_opacity_mean(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        return self.get_beta_params(idx=idx).mean

    def get_opacity_concentration(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        return self.get_beta_params(idx=idx).concentration

    def get_opacity_variance(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        return self.get_beta_params(idx=idx).variance

    def oneup_sh_degree(self) -> None:
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def position_at(self, timestamp: float | torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(timestamp, dtype=torch.float32, device=self.mu_ref.device)
        return self.mu_ref + self.velocity * (t.reshape(1, 1) - self.t_ref)

    def anchor_position_at(self, timestamp: float | torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(timestamp, dtype=torch.float32, device=self.mu_ref.device)
        return self.anchor_mu_ref + self.anchor_velocity * (t.reshape(1, 1) - self.t_ref)

    def active_indices_at(self, timestamp: float, temporal_window: float | None = None) -> torch.Tensor:
        if self.temporal_mode == "nearest_frame":
            times = self.t_ref.reshape(-1)
            distance = torch.abs(times - timestamp)
            closest = torch.argmin(distance)
            if temporal_window is not None and temporal_window > 0:
                if float(distance[closest]) > temporal_window + 1e-6:
                    raise ValueError("Requested time is outside the configured local motion support.")
            # A frame is a whole surface sample, not one arbitrary point. Do not
            # alpha-composite two independent reconstructions of the same sheet.
            return torch.nonzero(times == times[closest], as_tuple=False).reshape(-1)
        if temporal_window is None or temporal_window <= 0:
            return torch.arange(self.n_gaussians, device=self.mu_ref.device)
        dt = torch.abs(torch.as_tensor(timestamp, dtype=torch.float32, device=self.mu_ref.device) - self.t_ref.reshape(-1))
        idx = torch.nonzero(dt <= temporal_window, as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            idx = torch.argmin(dt).reshape(1)
        return idx

    def temporal_weights(self, timestamp: float, idx: torch.Tensor, temporal_sigma: float | None = None) -> torch.Tensor:
        if self.temporal_mode == "nearest_frame" or temporal_sigma is None or temporal_sigma <= 0:
            return torch.ones((idx.shape[0], 1), dtype=torch.float32, device=self.mu_ref.device)
        dt = torch.as_tensor(timestamp, dtype=torch.float32, device=self.mu_ref.device) - self.t_ref[idx]
        return torch.exp(-0.5 * (dt / temporal_sigma) ** 2)

    def get_render_params(
        self,
        timestamp: float,
        retain_screen_grad: bool = False,
        temporal_window: float | None = None,
        temporal_sigma: float | None = None,
        selection_timestamp: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        selection_time = timestamp if selection_timestamp is None else selection_timestamp
        idx = self.active_indices_at(selection_time, temporal_window)
        if self.temporal_mode == "nearest_frame" and temporal_window is not None and temporal_window > 0:
            if float(torch.abs(timestamp - self.t_ref[idx]).max()) > temporal_window + 1e-6:
                raise ValueError("Shutter sample exceeds the selected frame's local motion support.")
        means3d = self.mu_ref[idx] + self.velocity[idx] * (timestamp - self.t_ref[idx])
        screenspace_points = torch.zeros_like(
            means3d,
            dtype=means3d.dtype,
            requires_grad=retain_screen_grad,
            device=means3d.device,
        )
        opacity = self.get_opacity_mean(idx) * self.temporal_weights(timestamp, idx, temporal_sigma)
        return {
            "means3D": means3d,
            "means2D": screenspace_points,
            "shs": torch.cat((self.features_dc[idx], self.features_rest[idx]), dim=1),
            "colors_precomp": None,
            "opacities": opacity,
            "scales": torch.exp(self.scaling[idx]),
            "rotations": torch.nn.functional.normalize(self.rotation[idx], dim=1),
            "cov3D_precomp": None,
            "active_indices": idx,
        }

    def set_train_stage(self, stage: int) -> None:
        """Freeze risky parameters early so blur is explained by motion."""
        train = {
            "mu_ref": stage >= 2,
            "velocity": stage >= 3,
            "features_dc": True,
            "features_rest": stage >= 3,
            "opacity_logit": True,
            "opacity_logconc": stage >= 2,
            "scaling": stage >= 5,
            "rotation": stage >= 5,
        }
        for name, enabled in train.items():
            getattr(self, name).requires_grad_(enabled)

    def optimizer(self, scene_extent: float, iterations: int, velocity_time_scale: float = 1.0) -> torch.optim.Optimizer:
        spatial = max(float(scene_extent), 1e-6)
        groups = [
            {"params": [self.mu_ref], "lr": 1.6e-4 * spatial, "name": "mu_ref"},
            {"params": [self.velocity], "lr": 8.0e-5 * spatial / velocity_time_scale, "name": "velocity"},
            {"params": [self.features_dc], "lr": 0.0025, "name": "features_dc"},
            {"params": [self.features_rest], "lr": 0.0025 / 20.0, "name": "features_rest"},
            {"params": [self.opacity_logit], "lr": 0.025, "name": "opacity_logit"},
            {"params": [self.opacity_logconc], "lr": 0.005, "name": "opacity_logconc"},
            {"params": [self.scaling], "lr": 0.001, "name": "scaling"},
            {"params": [self.rotation], "lr": 0.0005, "name": "rotation"},
        ]
        return torch.optim.Adam(groups, lr=0.0, eps=1e-15)

    def save_npz(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        beta_params = self.get_beta_params()
        np.savez_compressed(
            path,
            track_ids=self.track_ids.detach().cpu().numpy(),
            mu_ref=self.mu_ref.detach().cpu().numpy(),
            velocity=self.velocity.detach().cpu().numpy(),
            t_ref=self.t_ref.detach().cpu().numpy().reshape(-1),
            colors=self.dc_colors().detach().cpu().numpy(),
            features_dc=self.features_dc.detach().cpu().numpy(),
            features_rest=self.features_rest.detach().cpu().numpy(),
            active_sh_degree=np.asarray(self.active_sh_degree),
            max_sh_degree=np.asarray(self.max_sh_degree),
            opacity_logit=self.opacity_logit.detach().cpu().numpy(),
            opacity_logconc=self.opacity_logconc.detach().cpu().numpy(),
            scaling_parameter=self.scaling.detach().cpu().numpy(),
            rotation_parameter=self.rotation.detach().cpu().numpy(),
            opacity=beta_params.mean.detach().cpu().numpy(),
            opacity_concentration=beta_params.concentration.detach().cpu().numpy(),
            opacity_alpha=beta_params.alpha.detach().cpu().numpy(),
            opacity_beta=beta_params.beta.detach().cpu().numpy(),
            opacity_variance=beta_params.variance.detach().cpu().numpy(),
            scales=torch.exp(self.scaling).detach().cpu().numpy(),
            rotations=torch.nn.functional.normalize(self.rotation, dim=1).detach().cpu().numpy(),
            frame_indices=self.frame_indices.detach().cpu().numpy(),
            source_indices=self.source_indices.detach().cpu().numpy(),
            parent_ids=self.parent_ids.detach().cpu().numpy(),
            generation=self.generation.detach().cpu().numpy(),
            mode=np.asarray(self.mode),
            temporal_mode=np.asarray(self.temporal_mode),
        )

    def dc_colors(self) -> torch.Tensor:
        return torch.clamp(self.features_dc[:, 0, :] * 0.28209 + 0.5, 0.0, 1.0)


def save_ply_at_time(model: DynamicGaussianModel, path: str, timestamp: float, temporal_window: float | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    idx = model.active_indices_at(timestamp, temporal_window)
    xyz = model.position_at(timestamp)[idx].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    colors = model.dc_colors()[idx].detach().cpu().numpy()
    f_dc = ((colors - 0.5) / 0.28209).astype(np.float32)
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    beta_params = model.get_beta_params(idx=idx)
    opacity = beta_params.mean.detach().cpu().numpy()
    concentration = beta_params.concentration.detach().cpu().numpy()
    variance = beta_params.variance.detach().cpu().numpy()
    scales = torch.exp(model.scaling[idx]).detach().cpu().numpy()
    rotations = torch.nn.functional.normalize(model.rotation[idx], dim=1).detach().cpu().numpy()

    write_vertex_ply(
        path,
        {
            "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
            "nx": normals[:, 0], "ny": normals[:, 1], "nz": normals[:, 2],
            "red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2],
            "f_dc_0": f_dc[:, 0], "f_dc_1": f_dc[:, 1], "f_dc_2": f_dc[:, 2],
            "opacity": opacity.reshape(-1),
            "opacity_concentration": concentration.reshape(-1),
            "opacity_variance": variance.reshape(-1),
            "confidence": np.log(np.clip(concentration.reshape(-1), 1e-6, None)),
            "scale_0": scales[:, 0], "scale_1": scales[:, 1], "scale_2": scales[:, 2],
            "rot_0": rotations[:, 0], "rot_1": rotations[:, 1], "rot_2": rotations[:, 2], "rot_3": rotations[:, 3],
        },
        text=False,
    )
