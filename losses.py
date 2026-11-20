# -*- coding: utf-8 -*-
"""Losses for point-cloud anchored dynamic blur-aware GS."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _fspecial_gauss_1d(size, sigma, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def ssim_loss(img1, img2, window_size=11):
    C, _, _ = img1.shape
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    one_d = _fspecial_gauss_1d(window_size, 1.5, img1.device)
    two_d = one_d.transpose(-1, -2) @ one_d
    window = two_d.expand(C, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return 1.0 - ssim.mean()


def photometric_loss(rendered, target, lambda_dssim=0.2, objective="l1_ssim"):
    if objective == "mse":
        mse = (rendered - target).square().mean()
        return mse, {"mse": mse.detach()}
    if objective != "l1_ssim":
        raise ValueError("Unknown photometric objective: " + str(objective))
    l1 = torch.abs(rendered - target).mean()
    dssim = ssim_loss(rendered, target)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim, {"l1": l1.detach(), "dssim": dssim.detach()}


def anchor_loss(model, timestamp: float, sample_count: int = 4096):
    n = model.n_gaussians
    if n == 0:
        return torch.zeros((), device=model.mu_ref.device)
    if sample_count and n > sample_count:
        idx = torch.randint(0, n, (sample_count,), device=model.mu_ref.device)
        pred = model.position_at(timestamp)[idx]
        ref = model.anchor_position_at(timestamp)[idx]
    else:
        pred = model.position_at(timestamp)
        ref = model.anchor_position_at(timestamp)
    return torch.abs(pred - ref).mean()


def track_sample_loss(model, sample_count: int = 4096):
    n = model.n_gaussians
    if sample_count and n > sample_count:
        idx = torch.randint(0, n, (sample_count,), device=model.mu_ref.device)
    else:
        idx = torch.arange(n, device=model.mu_ref.device)
    times = model.track_times[idx]
    points = model.track_points[idx]
    mask = model.track_mask[idx].unsqueeze(-1)
    tref = model.t_ref[idx].unsqueeze(1)
    pred = model.mu_ref[idx].unsqueeze(1) + model.velocity[idx].unsqueeze(1) * (times.unsqueeze(-1) - tref)
    denom = mask.sum().clamp(min=1.0)
    return (torch.abs(pred - points) * mask).sum() / (denom * 3.0)


def scale_regularization(
    model,
    max_anisotropy: float = 5.0,
    max_scale: float | None = None,
    normal_ratio: float = 0.35,
    rotation_anchor_weight: float = 0.1,
):
    scales = torch.exp(model.scaling)
    # Sort ascending so sorted_s[:, 0] is always the thinnest axis regardless of rotation
    sorted_s, _ = scales.sort(dim=1)
    thin_s = sorted_s[:, 0].clamp(min=1e-8)
    thick_s = sorted_s[:, 2]
    anis = thick_s / thin_s
    loss = F.relu(anis - max_anisotropy).mean()
    if max_scale is not None and max_scale > 0:
        loss = loss + F.relu(thick_s - max_scale).mean()
    # Enforce flat-disc shape: thinnest axis must stay thin relative to the two larger axes
    tangent_mean = sorted_s[:, 1:].mean(dim=1)
    loss = loss + F.relu(thin_s - normal_ratio * tangent_mean).mean()
    # Axis 2 is initialized as the surface-normal axis; keep that axis thin so
    # the sorted-scale constraint cannot silently move thickness into a tangent axis.
    normal_axis = scales[:, 2]
    local_tangent_mean = scales[:, :2].mean(dim=1)
    loss = loss + F.relu(normal_axis - normal_ratio * local_tangent_mean).mean()
    if rotation_anchor_weight > 0 and hasattr(model, "anchor_rotation"):
        current = torch.nn.functional.normalize(model.rotation, dim=1)
        anchor = torch.nn.functional.normalize(model.anchor_rotation, dim=1)
        # q and -q represent the same rotation, so compare with sign invariance.
        alignment = torch.abs(torch.sum(current * anchor, dim=1)).clamp(0.0, 1.0)
        loss = loss + rotation_anchor_weight * (1.0 - alignment).mean()
    return loss


def velocity_regularization(model, time_scale=1.0):
    return time_scale * torch.mean(torch.linalg.norm(model.velocity - model.anchor_velocity, dim=1))


def nearest_point_indices(query, reference, block_size=8192):
    """Exact Euclidean correspondences with bounded GPU memory, not sampled targets."""
    if len(reference) == 0:
        raise ValueError("Nearest-neighbor reference cannot be empty.")
    with torch.no_grad():
        nearest = torch.zeros(len(query), dtype=torch.long, device=query.device)
        best_distance = torch.full((len(query),), torch.inf, dtype=query.dtype, device=query.device)
        for start in range(0, len(reference), block_size):
            distance = torch.cdist(query.detach(), reference[start:start + block_size].detach(),
                                   compute_mode="donot_use_mm_for_euclid_dist")
            block_distance, block_index = distance.min(dim=1)
            closer = block_distance < best_distance
            nearest[closer] = block_index[closer] + start
            best_distance[closer] = block_distance[closer]
    return nearest


def sand_surface_loss(
    model,
    timestamp: float,
    target_points: torch.Tensor,
    target_normals: torch.Tensor,
    temporal_window: float,
    sample_count: int = 2048,
    point_plane_weight: float = 1.0,
    correspondence_mode: str = "sampled_legacy",
    objective: str = "chamfer",
):
    """
    Anchor short-lived sand surfels to the visible free surface.

    This is not a long track loss. It compares the active Gaussians at the
    current time to the reconstructed point cloud for that frame.
    """
    idx = model.active_indices_at(timestamp, temporal_window)
    if idx.numel() == 0 or target_points.numel() == 0:
        return torch.zeros((), device=model.mu_ref.device), {"chamfer": 0.0, "point_plane": 0.0}

    if correspondence_mode not in ("sampled_legacy", "full_reference"):
        raise ValueError(f"Unknown surface correspondence mode: {correspondence_mode}")
    if objective not in ("chamfer", "plane_coverage"):
        raise ValueError(f"Unknown surface objective: {objective}")
    if objective == "plane_coverage" and correspondence_mode != "full_reference":
        raise ValueError("Plane/coverage loss requires complete reference correspondences.")
    full_pred = model.position_at(timestamp)[idx]
    pred = full_pred
    if pred.shape[0] > sample_count:
        pred = pred[torch.randperm(pred.shape[0], device=pred.device)[:sample_count]]
    tgt = target_points
    nrm = target_normals
    if tgt.shape[0] > sample_count:
        pick = torch.randperm(tgt.shape[0], device=tgt.device)[:sample_count]
        tgt = tgt[pick]
        nrm = nrm[pick]

    if correspondence_mode == "full_reference":
        target_indices = nearest_point_indices(pred, target_points)
        prediction_indices = nearest_point_indices(tgt, full_pred)
        pred_to_tgt = torch.linalg.vector_norm(pred - target_points[target_indices], dim=1)
        tgt_to_pred = torch.linalg.vector_norm(tgt - full_pred[prediction_indices], dim=1)
        nearest_normals = target_normals[target_indices]
        matched_targets = target_points[target_indices]
    else:
        dist = torch.cdist(pred, tgt)
        pred_to_tgt, nn_idx = dist.min(dim=1)
        tgt_to_pred, _ = dist.min(dim=0)
        nearest_normals = nrm[nn_idx]
        matched_targets = tgt[nn_idx]
    chamfer = pred_to_tgt.mean() + tgt_to_pred.mean()
    point_plane = torch.abs(torch.sum((pred - matched_targets) * nearest_normals, dim=1)).mean()
    loss = chamfer + point_plane_weight * point_plane
    if objective == "plane_coverage":
        # Tangent displacement between samples is not itself surface error.
        # Reverse distance retains measurement coverage; planes remain local approximations.
        loss = tgt_to_pred.mean() + point_plane_weight * point_plane
    return loss, {"chamfer": float(chamfer.detach()), "point_plane": float(point_plane.detach()),
                  "coverage": float(tgt_to_pred.mean().detach())}


def depth_loss(rendered_depth, reference_depth):
    if rendered_depth is None:
        return torch.zeros((), device=reference_depth.device)
    rd = rendered_depth.squeeze()
    mask = reference_depth > 0.01
    if not mask.any():
        return torch.zeros((), device=reference_depth.device)
    return torch.abs(rd[mask] - reference_depth[mask]).mean()


def silhouette_loss(alpha, mask):
    if alpha is None or mask is None:
        if alpha is not None:
            device = alpha.device
        elif mask is not None:
            device = mask.device
        else:
            return torch.zeros(())
        return torch.zeros((), device=device)
    # Clamp strictly inside (0, 1) to avoid log(0) = -inf in BCE
    a = alpha.squeeze().clamp(1e-7, 1.0 - 1e-7)
    m = mask.float().squeeze().clamp(0.0, 1.0)
    return F.binary_cross_entropy(a, m)
