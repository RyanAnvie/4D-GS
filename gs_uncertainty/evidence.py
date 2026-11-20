# -*- coding: utf-8 -*-
"""Photogrammetry evidence targets for dynamic granular-surface 4DGS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from data_io import CameraInfo, SequenceFrame


@dataclass(frozen=True)
class EvidenceWeights:
    support: float = 0.35
    reprojection: float = 0.30
    geometry: float = 0.20
    temporal: float = 0.15


@dataclass(frozen=True)
class EvidenceConfig:
    weights: EvidenceWeights = field(default_factory=EvidenceWeights)
    color_tau: float = 0.12
    geometry_tau: float = 0.50
    temporal_tau: float = 0.02
    spacing_k: int = 12
    visibility_znear: float = 1e-3
    kappa_min: float = 1.0
    kappa_scale: float = 30.0


@dataclass(frozen=True)
class FrameEvidence:
    frame_idx: int
    support_ratio: np.ndarray
    reprojection_error: np.ndarray
    reprojection_confidence: np.ndarray
    geometry_confidence: np.ndarray
    temporal_residual: np.ndarray
    temporal_confidence: np.ndarray
    evidence: np.ndarray


@dataclass(frozen=True)
class PhotogrammetryTargets:
    confidence: np.ndarray
    kappa_target: np.ndarray
    support_ratio: np.ndarray
    reprojection_error: np.ndarray
    geometry_confidence: np.ndarray
    temporal_residual: np.ndarray


def _normalize_weights(weights: EvidenceWeights) -> EvidenceWeights:
    values = np.asarray([weights.support, weights.reprojection, weights.geometry, weights.temporal], dtype=np.float32)
    total = float(np.clip(values.sum(), 1e-8, None))
    return EvidenceWeights(*(float(v / total) for v in values))


def _cameras_for_frame(cameras: Iterable[CameraInfo]) -> dict[int, list[CameraInfo]]:
    by_frame: dict[int, list[CameraInfo]] = {}
    for cam in cameras:
        by_frame.setdefault(int(cam.frame_idx), []).append(cam)
    return by_frame


def project_points(points: np.ndarray, cam: CameraInfo, znear: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world points to one camera and return uv, depth, and image-valid mask."""
    device = cam.image.device
    pts = torch.as_tensor(points, dtype=torch.float32, device=device)
    ones = torch.ones((pts.shape[0], 1), dtype=torch.float32, device=device)
    cam_pts = (cam.w2c @ torch.cat([pts, ones], dim=1).T).T[:, :3]
    z = cam_pts[:, 2]
    u = cam.fx * (cam_pts[:, 0] / z.clamp(min=1e-8)) + cam.cx
    v = cam.fy * (cam_pts[:, 1] / z.clamp(min=1e-8)) + cam.cy
    valid = (z > znear) & (u >= 0.0) & (u <= cam.width - 1) & (v >= 0.0) & (v <= cam.height - 1)
    return torch.stack([u, v], dim=1), z, valid


def sample_image_colors(cam: CameraInfo, uv: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample RGB values at projected pixel coordinates."""
    if uv.numel() == 0:
        return torch.empty((0, 3), dtype=cam.image.dtype, device=cam.image.device)
    x = 2.0 * uv[:, 0] / max(cam.width - 1, 1) - 1.0
    y = 2.0 * uv[:, 1] / max(cam.height - 1, 1) - 1.0
    grid = torch.stack([x, y], dim=1).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(cam.image.unsqueeze(0), grid, align_corners=True, mode="bilinear", padding_mode="zeros")
    return sampled.squeeze(0).squeeze(-1).T


def support_and_reprojection(frame: SequenceFrame, cameras: Iterable[CameraInfo], config: EvidenceConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate multi-view support and color reprojection consistency for one frame."""
    points = frame.points
    colors = np.clip(frame.colors, 0.0, 1.0)
    n = points.shape[0]
    support = np.zeros(n, dtype=np.float32)
    error_sum = np.zeros(n, dtype=np.float32)
    error_count = np.zeros(n, dtype=np.float32)
    frame_cameras = list(cameras)
    if n == 0 or not frame_cameras:
        return support, np.ones(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

    color_t = None
    for cam in frame_cameras:
        uv, _, valid = project_points(points, cam, znear=config.visibility_znear)
        valid_idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if valid_idx.numel() == 0:
            continue
        if color_t is None or color_t.device != cam.image.device:
            color_t = torch.as_tensor(colors, dtype=torch.float32, device=cam.image.device)
        sampled = sample_image_colors(cam, uv[valid_idx])
        residual = torch.mean(torch.abs(sampled - color_t[valid_idx]), dim=1)
        idx_np = valid_idx.detach().cpu().numpy()
        support[idx_np] += 1.0
        error_sum[idx_np] += residual.detach().cpu().numpy().astype(np.float32)
        error_count[idx_np] += 1.0

    support_ratio = support / max(float(len(frame_cameras)), 1.0)
    reprojection_error = np.divide(error_sum, np.clip(error_count, 1.0, None)).astype(np.float32)
    no_support = error_count < 1.0
    reprojection_error[no_support] = 1.0
    reprojection_confidence = np.exp(-reprojection_error / max(config.color_tau, 1e-8)).astype(np.float32)
    return support_ratio.astype(np.float32), reprojection_error, reprojection_confidence


def geometry_confidence(frame: SequenceFrame, config: EvidenceConfig) -> np.ndarray:
    """Use local normal variation and density variation as geometric stability evidence."""
    points = frame.points
    normals = frame.normals
    n = points.shape[0]
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    k = min(max(config.spacing_k, 3), n)
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k)
    neigh_normals = normals[indices]
    normal_alignment = np.abs(np.sum(neigh_normals * normals[:, None, :], axis=2))
    normal_instability = 1.0 - np.clip(np.mean(normal_alignment[:, 1:], axis=1), 0.0, 1.0)
    local_spacing = np.mean(distances[:, 1:], axis=1)
    median_spacing = np.median(local_spacing[local_spacing > 0]) if np.any(local_spacing > 0) else 1.0
    density_instability = np.abs(np.log(np.clip(local_spacing / max(median_spacing, 1e-8), 1e-6, None)))
    instability = 0.65 * normal_instability + 0.35 * np.clip(density_instability, 0.0, 3.0) / 3.0
    return np.exp(-instability / max(config.geometry_tau, 1e-8)).astype(np.float32)


def temporal_residuals(frames: Iterable[SequenceFrame]) -> Mapping[int, np.ndarray]:
    """Measure nearest-neighbor surface mismatch to adjacent frames."""
    ordered = sorted(frames, key=lambda f: f.timestamp)
    residual_by_frame: dict[int, np.ndarray] = {}
    for i, frame in enumerate(ordered):
        residuals = []
        for j in (i - 1, i + 1):
            if j < 0 or j >= len(ordered) or frame.points.shape[0] == 0 or ordered[j].points.shape[0] == 0:
                continue
            tree = cKDTree(ordered[j].points)
            dist, _ = tree.query(frame.points, k=1)
            residuals.append(dist.astype(np.float32))
        if residuals:
            residual_by_frame[frame.frame_idx] = np.mean(np.stack(residuals, axis=0), axis=0).astype(np.float32)
        else:
            residual_by_frame[frame.frame_idx] = np.zeros(frame.points.shape[0], dtype=np.float32)
    return residual_by_frame


def frame_evidence(frame: SequenceFrame, cameras: Iterable[CameraInfo], temporal_residual: np.ndarray, config: EvidenceConfig) -> FrameEvidence:
    """Combine all photogrammetry cues for one sequence frame."""
    weights = _normalize_weights(config.weights)
    support_ratio, reproj_error, reproj_conf = support_and_reprojection(frame, cameras, config)
    geom_conf = geometry_confidence(frame, config)
    temporal_conf = np.exp(-temporal_residual / max(config.temporal_tau, 1e-8)).astype(np.float32)
    evidence = (
        weights.support * support_ratio
        + weights.reprojection * reproj_conf
        + weights.geometry * geom_conf
        + weights.temporal * temporal_conf
    )
    return FrameEvidence(
        frame_idx=frame.frame_idx,
        support_ratio=support_ratio,
        reprojection_error=reproj_error,
        reprojection_confidence=reproj_conf,
        geometry_confidence=geom_conf,
        temporal_residual=temporal_residual.astype(np.float32),
        temporal_confidence=temporal_conf,
        evidence=np.clip(evidence, 0.0, 1.0).astype(np.float32),
    )


def align_frame_evidence_to_gaussians(init: Mapping[str, np.ndarray], frames: Iterable[SequenceFrame], evidence_by_frame: Mapping[int, FrameEvidence]) -> PhotogrammetryTargets:
    """Transfer per-point frame evidence to initialized Gaussians by frame-local NN."""
    mu_ref = np.asarray(init["mu_ref"], dtype=np.float32)
    frame_indices = np.asarray(init.get("frame_indices", np.zeros(mu_ref.shape[0], dtype=np.int64)), dtype=np.int64)
    frame_by_idx = {int(f.frame_idx): f for f in frames}
    fields = {
        "confidence": np.zeros(mu_ref.shape[0], dtype=np.float32),
        "support_ratio": np.zeros(mu_ref.shape[0], dtype=np.float32),
        "reprojection_error": np.ones(mu_ref.shape[0], dtype=np.float32),
        "geometry_confidence": np.zeros(mu_ref.shape[0], dtype=np.float32),
        "temporal_residual": np.zeros(mu_ref.shape[0], dtype=np.float32),
    }
    for frame_idx in np.unique(frame_indices):
        mask = frame_indices == int(frame_idx)
        frame = frame_by_idx.get(int(frame_idx))
        evidence = evidence_by_frame.get(int(frame_idx))
        if frame is None or evidence is None or frame.points.shape[0] == 0:
            continue
        _, nn_idx = cKDTree(frame.points).query(mu_ref[mask], k=1)
        fields["confidence"][mask] = evidence.evidence[nn_idx]
        fields["support_ratio"][mask] = evidence.support_ratio[nn_idx]
        fields["reprojection_error"][mask] = evidence.reprojection_error[nn_idx]
        fields["geometry_confidence"][mask] = evidence.geometry_confidence[nn_idx]
        fields["temporal_residual"][mask] = evidence.temporal_residual[nn_idx]
    return PhotogrammetryTargets(
        confidence=fields["confidence"],
        kappa_target=np.zeros_like(fields["confidence"]),
        support_ratio=fields["support_ratio"],
        reprojection_error=fields["reprojection_error"],
        geometry_confidence=fields["geometry_confidence"],
        temporal_residual=fields["temporal_residual"],
    )


def build_photogrammetry_targets(
    init: Mapping[str, np.ndarray],
    frames: Iterable[SequenceFrame],
    cameras: Iterable[CameraInfo],
    config: EvidenceConfig = EvidenceConfig(),
) -> PhotogrammetryTargets:
    """Build per-Gaussian confidence and concentration targets from photogrammetry evidence."""
    frames = list(frames)
    cameras_by_frame = _cameras_for_frame(cameras)
    temporal_by_frame = temporal_residuals(frames)
    evidence_by_frame = {
        int(frame.frame_idx): frame_evidence(
            frame,
            cameras_by_frame.get(int(frame.frame_idx), []),
            temporal_by_frame[int(frame.frame_idx)],
            config,
        )
        for frame in frames
    }
    targets = align_frame_evidence_to_gaussians(init, frames, evidence_by_frame)
    confidence = np.clip(targets.confidence, 0.0, 1.0).astype(np.float32)
    kappa_target = (config.kappa_min + config.kappa_scale * confidence).astype(np.float32)
    return PhotogrammetryTargets(
        confidence=confidence,
        kappa_target=kappa_target,
        support_ratio=targets.support_ratio,
        reprojection_error=targets.reprojection_error,
        geometry_confidence=targets.geometry_confidence,
        temporal_residual=targets.temporal_residual,
    )
