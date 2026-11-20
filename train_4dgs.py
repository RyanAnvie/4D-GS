# -*- coding: utf-8 -*-
"""Train point-cloud anchored dynamic blur-aware Gaussian Splatting."""

from __future__ import annotations

import os
import json
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from data_io import (  # noqa: E402
    build_track_database,
    initialize_from_tracks,
    initialize_sand_surface,
    load_fixed_rig_continue_frame,
    load_openmvg_sequence,
    load_sequence_clouds,
    project_cloud_to_depth,
)
from losses import (  # noqa: E402
    anchor_loss,
    depth_loss,
    photometric_loss,
    sand_surface_loss,
    scale_regularization,
    silhouette_loss,
    track_sample_loss,
    velocity_regularization,
)
from model import DynamicGaussianModel, save_ply_at_time  # noqa: E402
from renderer import render_exposure_integrated  # noqa: E402
from densification import (  # noqa: E402
    new_densification_stats, accumulate_screen_gradients,
    select_split_parents, split_surface_gaussians,
)
from surface_support import build_surface_support, surface_support_loss, project_surface_centers, project_surface_scales
from gs_uncertainty.evidence import EvidenceConfig, EvidenceWeights, build_photogrammetry_targets  # noqa: E402
from gs_uncertainty.losses import beta_concentration_prior_loss, beta_kl_loss, kappa_evidence_loss  # noqa: E402


@dataclass
class TrainConfig:
    seed: int = 42
    held_out_camera_ids: tuple = ()
    held_out_frame_ids: tuple = ()
    mode: str = "flow_sand"
    sfm_json: str | None = None
    sequence_cloud: str | None = None
    continue_frame_root: str | None = None
    output: str = "gs_model_4d"
    views_manifest: str | None = None
    exposure_json: str | None = None
    mask_dir: str | None = None
    fps: float = 30.0
    num_cameras: int = 3
    default_exposure: float = 0.0
    iterations: int = 30000
    downsample: float = 0.0
    target_width: int = 1600
    sh_degree: int = 0
    prefer_undistorted: bool = True
    cpu: bool = False
    min_track_len: int = 2
    spacing_k: int = 6
    max_track_samples: int = 32
    normal_scale_ratio: float = 0.08
    max_points_per_frame: int = 20000
    flow_max_distance: float = 0.02
    flow_min_normal_dot: float = 0.35
    velocity_source: str = "nearest_cloud"
    temporal_window: float = 0.0
    temporal_sigma: float = 0.0
    temporal_mode: str = "legacy_window"
    zero_initial_velocity: bool = False
    learn_velocity: bool = True
    velocity_lr_time_scale: float = 1.0  # Seconds; 1.0 preserves the legacy optimizer.
    velocity_prior_time_scale: float = 1.0  # Converts velocity difference to a displacement.
    freeze_surface_shape: bool = True
    refine_surface_tangents: bool = False  # Opt-in stage5; normal thickness/rotation stay fixed.
    exposure_samples: int = 5
    exposure_rule: str = "legacy_endpoints"
    shutter_reference_mode: str = "sample_nearest"
    temporal_photo_mode: str = "disabled"  # disabled, same_time (control), adjacent_train
    temporal_photo_weight: float = 0.0  # Normalized with the primary photo loss; stage3 onward.
    loss_sample_count: int = 4096
    surface_sample_count: int = 2048
    surface_correspondence: str = "sampled_legacy"
    surface_objective: str = "chamfer"
    photometric_objective: str = "l1_ssim"  # l1_ssim (legacy) or mse; both use raw renders.
    lambda_dssim: float = 0.2
    lambda_depth: float = 0.0
    lambda_surface: float = 1.0
    lambda_point_plane: float = 1.0
    lambda_anchor: float = 0.0
    lambda_track: float = 0.0
    lambda_motion: float = 0.05
    lambda_scale: float = 0.1
    lambda_silhouette: float = 0.0
    max_anisotropy: float = 5.0
    max_scale: float = 0.0
    normal_ratio_limit: float = 0.35
    lambda_rotation_anchor: float = 0.1
    lambda_kappa_evidence: float = 1e-3
    lambda_beta_kl: float = 1e-5
    lambda_beta_prior: float = 0.0
    beta_prior_mu: float = 0.1
    beta_prior_kappa: float = 2.0
    kappa_min: float = 1.0
    kappa_scale: float = 20.0
    evidence_weight_support: float = 0.35
    evidence_weight_reprojection: float = 0.30
    evidence_weight_geometry: float = 0.20
    evidence_weight_temporal: float = 0.15
    evidence_color_tau: float = 0.12
    evidence_geometry_tau: float = 0.50
    evidence_temporal_tau: float = 0.02
    log_interval: int = 100
    save_interval: int = 2000
    sh_up_interval: int = 1000
    export_stride: int = 1
    adaptive_densification: bool = False
    audit_split_geometry: bool = False
    densify_geometry_filter: bool = False
    densify_from: int = 1200
    densify_until: int = 4200
    densify_interval: int = 300
    densify_fraction: float = 0.10
    densify_max_per_frame: int = 100000
    densify_max_per_source: int = 0  # 0 disables the cap; counts use immutable source IDs.
    densify_position_mode: str = "symmetric"  # Opt-in birth-position control, not a precision claim.
    densify_min_observations: float = 2.0
    densify_grad_threshold: float = 0.0  # Rank positive gradients; not a calibrated error threshold.
    densify_min_radius: float = 2.0  # Rasterizer footprint radius in pixels.
    densify_min_opacity: float = 0.01
    surface_constraint: bool = False
    surface_origin_path: Optional[str] = None  # Explicit TRAIN-derived plane origins; raw measurements are unchanged.
    surface_projection_reset_moments: bool = True  # Legacy projection policy; False is an experimental control.
    surface_tangent_radius: float = 0.5  # Multiples of original mean k-neighbor spacing.
    surface_normal_band: float = 0.08  # Algorithmic support band, NOT measurement uncertainty.
    surface_split_center_fraction: float = 0.5
    surface_split_room_fraction: float = 0.9


def continue_frame_config(**overrides) -> TrainConfig:
    """Default IDE-friendly experiment aligned with the bundled granular-surface data."""
    config = TrainConfig(
        mode="flow_sand",
        continue_frame_root=os.path.join(THIS_DIR, "continue_frame"),
        sfm_json=os.path.join(THIS_DIR, "continue_frame", "robust.json"),
        sequence_cloud=os.path.join(THIS_DIR, "continue_frame"),
        output=os.path.join(THIS_DIR, "continue_frame_output"),
        fps=30.0,
        default_exposure=0.002,
        iterations=3000,
        target_width=960,
        sh_degree=1,
        normal_scale_ratio=0.08,
        flow_max_distance=0.02,
        lambda_surface=1.0,
        lambda_scale=0.1,
        lambda_depth=0.0,
        lambda_silhouette=0.0,
        max_points_per_frame=50000,
    )
    return replace(config, **overrides) if overrides else config


def get_scene_extent(cameras):
    centers = torch.stack([cam.camera_center for cam in cameras], dim=0)
    center = centers.mean(dim=0, keepdim=True)
    return max(torch.norm(centers - center, dim=1).max().item() * 1.1, 1e-6)


def stage_for_iteration(iteration: int, iterations: int) -> int:
    frac = iteration / max(iterations, 1)
    if frac < 0.12:
        return 1
    if frac < 0.30:
        return 2
    if frac < 0.55:
        return 3
    if frac < 0.85:
        return 4
    return 5


def load_mask(mask_dir: str | None, cam):
    if not mask_dir:
        return None
    stem = Path(cam.image_path).stem
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        path = Path(mask_dir) / f"{stem}{suffix}"
        if path.exists():
            img = Image.open(path).convert("L").resize((cam.width, cam.height), Image.Resampling.NEAREST)
            return torch.tensor(np.asarray(img) > 127, dtype=torch.float32, device=cam.image.device)
    return None


def cache_reference_depths(sequence_frames, cameras):
    frame_map = {f.frame_idx: f for f in sequence_frames}
    depth_cache = {}
    for cam in cameras:
        frame = frame_map.get(cam.frame_idx)
        if frame is None:
            continue
        depth_cache[(cam.frame_idx, cam.camera_id)] = project_cloud_to_depth(frame.points, cam)
    return depth_cache


def estimate_clip_planes(sequence_frames, cameras, scene_extent: float):
    depths = []
    frame_map = {f.frame_idx: f for f in sequence_frames}
    for cam in cameras:
        frame = frame_map.get(cam.frame_idx)
        if frame is None or frame.points.shape[0] == 0:
            continue
        pts = torch.as_tensor(frame.points, dtype=torch.float32, device=cam.w2c.device)
        ones = torch.ones((pts.shape[0], 1), dtype=torch.float32, device=cam.w2c.device)
        pts_h = torch.cat([pts, ones], dim=1)
        z = (cam.w2c @ pts_h.T).T[:, 2]
        valid = z[z > 1e-6]
        if valid.numel() > 0:
            depths.append(valid.detach().cpu().numpy())

    if depths:
        # torch.quantile rejects more than 2**24 entries. Keep all valid depths
        # and the same linear quantile definition; do not subsample camera geometry.
        z_all = np.concatenate(depths)
        near_q, far_q = np.quantile(z_all, [0.001, 0.999], method="linear")
        near_q, far_q = float(near_q), float(far_q)
        znear = max(min(near_q * 0.5, far_q * 0.1), 1e-3)
        zfar = max(far_q * 1.5, znear * 10.0, 100.0, scene_extent * 10.0)
        return znear, zfar

    cam_centers = torch.stack([cam.camera_center for cam in cameras], dim=0)
    scene_center = cam_centers.mean(dim=0)
    cam_dists = torch.norm(cam_centers - scene_center.unsqueeze(0), dim=1)
    mean_cam_dist = float(cam_dists.mean().item())
    znear = max(mean_cam_dist * 0.005, 1e-3)
    zfar = max(mean_cam_dist * 30.0, 100.0, scene_extent * 10.0)
    return znear, zfar


def cache_masks(cameras, mask_dir, device):
    if not mask_dir:
        return {}
    cache = {}
    for cam in cameras:
        mask = load_mask(mask_dir, cam)
        if mask is not None:
            cache[cam.image_path] = mask.to(device)
    return cache


def cache_frame_cloud_tensors(sequence_frames, device):
    return {
        f.frame_idx: {
            "points": torch.as_tensor(f.points, dtype=torch.float32, device=device),
            "normals": torch.as_tensor(f.normals, dtype=torch.float32, device=device),
        }
        for f in sequence_frames
    }


def select_training_observations(cameras, frames, held_out_camera_ids, held_out_frame_ids):
    """Exclude held-out times before initialization, scene flow, and evidence."""
    training_cameras = []
    for cam in cameras:
        if cam.camera_id not in held_out_camera_ids and cam.frame_idx not in held_out_frame_ids:
            training_cameras.append(cam)
    training_frames = [frame for frame in frames if frame.frame_idx not in held_out_frame_ids]
    if not training_cameras or not training_frames:
        raise ValueError("The split leaves no training cameras or sequence clouds.")
    split = {
        "held_out_camera_ids": list(held_out_camera_ids),
        "held_out_frame_ids": list(held_out_frame_ids),
        "training_views": [{"frame": cam.frame_idx, "camera": cam.camera_id,
                            "timestamp": cam.timestamp, "image": cam.image_path}
                           for cam in training_cameras],
        "training_cloud_frames": [frame.frame_idx for frame in training_frames],
        "cloud_provenance_requirement": "Input clouds must be reconstructed from training cameras only.",
    }
    return training_cameras, training_frames, split


def build_temporal_image_pairs(cameras, frames):
    """Only call with already-split training observations; pairs use the same camera ID."""
    frame_times = {frame.frame_idx: frame.timestamp for frame in frames}
    if not np.isfinite(list(frame_times.values())).all():
        raise ValueError("Training-cloud timestamps must be finite.")
    if len(frame_times) != len(frames) or len(set(frame_times.values())) != len(frames):
        raise ValueError("Cross-time training needs unique cloud frames/timestamps.")
    ordered = sorted(frame_times, key=frame_times.get)
    if len(ordered) < 2:
        raise ValueError("Cross-time training requires at least two training-cloud times.")
    views = {}
    for cam in cameras:
        key = (cam.frame_idx, cam.camera_id)
        if key in views:
            raise ValueError("Duplicate camera/frame in temporal training observations.")
        if not np.isfinite(cam.timestamp) or cam.frame_idx not in frame_times or abs(cam.timestamp - frame_times[cam.frame_idx]) > 1e-6:
            raise ValueError("Temporal image and cloud timestamps do not agree.")
        views[key] = cam
    partners, records = {}, []
    for cam in cameras:
        index = ordered.index(cam.frame_idx)
        neighbors = []
        if index > 0:
            neighbors.append(ordered[index - 1])
        if index + 1 < len(ordered):
            neighbors.append(ordered[index + 1])
        key = (cam.frame_idx, cam.camera_id)
        partners[key] = []
        for frame in neighbors:
            target = views.get((frame, cam.camera_id))
            if target is None:
                continue
            partners[key].append(target)
            records.append({"source_frame": cam.frame_idx, "target_frame": target.frame_idx,
                            "camera": cam.camera_id, "source_timestamp": cam.timestamp,
                            "target_timestamp": target.timestamp,
                            "source_image": cam.image_path, "target_image": target.image_path})
        if not partners[key]:
            raise ValueError("A training view has no same-camera adjacent training-time observation.")
    return partners, records


def train(args):
    if args.densify_position_mode not in ("symmetric", "parent_preserving"):
        raise ValueError("Unknown densify_position_mode.")
    if type(args.densify_max_per_source) is not int or args.densify_max_per_source < 0:
        raise ValueError("densify_max_per_source must be a nonnegative integer.")
    if args.photometric_objective not in ("l1_ssim", "mse"):
        raise ValueError("Unknown photometric objective.")
    if args.surface_origin_path is not None and not args.surface_constraint:
        raise ValueError("Derived surface origins require surface_constraint=True.")
    if args.refine_surface_tangents:
        if args.mode != "flow_sand" or not args.surface_constraint or not args.freeze_surface_shape:
            raise ValueError("Tangent refinement requires flow_sand with frozen shape and surface support.")
        if not np.isfinite(args.normal_ratio_limit) or not 0 < args.normal_ratio_limit < 1:
            raise ValueError("Tangent refinement needs a normal ratio strictly between zero and one.")
    if args.temporal_photo_mode not in ("disabled", "same_time", "adjacent_train"):
        raise ValueError("Unknown temporal photo mode.")
    if not np.isfinite(args.temporal_photo_weight) or args.temporal_photo_weight < 0:
        raise ValueError("Temporal photo weight must be finite and nonnegative.")
    temporal_photo_enabled = args.temporal_photo_mode != "disabled"
    if temporal_photo_enabled:
        if args.mode != "flow_sand" or args.temporal_mode != "nearest_frame" or args.shutter_reference_mode != "center_nearest":
            raise ValueError("Temporal photo loss requires flow_sand / nearest_frame / center_nearest.")
        if args.temporal_photo_weight <= 0:
            raise ValueError("Enabled temporal photo loss requires positive weight.")
    elif args.temporal_photo_weight != 0:
        raise ValueError("A nonzero temporal photo weight needs an explicit mode.")
    time_scales = (args.velocity_lr_time_scale, args.velocity_prior_time_scale)
    if not all(np.isfinite(value) and value > 0 for value in time_scales):
        raise ValueError("Velocity optimization time scales must be finite and positive.")
    if args.densify_geometry_filter and (args.mode != "flow_sand" or args.surface_normal_band < 0):
        raise ValueError("Geometry-aware splitting requires flow_sand and a nonnegative normal band.")
    if args.surface_constraint:
        if args.mode != "flow_sand" or not args.freeze_surface_shape:
            raise ValueError("Surface support requires flow_sand and frozen surface shape/rotation.")
        if args.surface_tangent_radius <= 0 or args.surface_normal_band < 0:
            raise ValueError("Invalid surface support radius/band.")
        if not 0 < args.surface_split_center_fraction < 1 or not 0 < args.surface_split_room_fraction < 1:
            raise ValueError("Split center and room fractions must be between zero and one.")
    if args.adaptive_densification:
        if args.mode != "flow_sand":
            raise ValueError("Adaptive surface splitting currently requires mode='flow_sand'.")
        if not (0 <= args.densify_from < args.densify_until < args.iterations):
            raise ValueError("Densification must finish before training ends.")
        if args.densify_interval <= 0 or not 0 < args.densify_fraction <= 1:
            raise ValueError("Invalid densification interval/fraction.")
        if args.densify_max_per_frame <= 0 or args.densify_min_observations <= 0:
            raise ValueError("Densification capacity and observation count must be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    if args.continue_frame_root:
        robust_json = args.sfm_json or os.path.join(args.continue_frame_root, "robust.json")
        cameras = load_fixed_rig_continue_frame(
            robust_json,
            args.continue_frame_root,
            exposure_json=args.exposure_json,
            default_exposure=args.default_exposure,
            fps=args.fps,
            downsample_factor=args.downsample,
            target_width=args.target_width,
            device=device,
        )
        sequence_root = args.sequence_cloud or args.continue_frame_root
    else:
        if not args.sfm_json:
            raise ValueError(
                "Missing input camera file. Pass --sfm_json for a generic OpenMVG sequence, "
                "or use --continue_frame_root for the bundled fixed-rig continue_frame dataset."
            )
        cameras = load_openmvg_sequence(
            args.sfm_json,
            views_manifest=args.views_manifest,
            exposure_json=args.exposure_json,
            default_exposure=args.default_exposure,
            fps=args.fps,
            num_cameras=args.num_cameras,
            downsample_factor=args.downsample,
            target_width=args.target_width,
            prefer_undistorted=args.prefer_undistorted,
            device=device,
        )
        sequence_root = args.sequence_cloud
    if not cameras:
        raise RuntimeError("No cameras loaded from OpenMVG data.")
    if not sequence_root:
        raise ValueError("Missing sequence point-cloud directory. Pass --sequence_cloud, or use --continue_frame_root.")
    sequence_frames = load_sequence_clouds(sequence_root, fps=args.fps)
    cameras, sequence_frames, split = select_training_observations(
        cameras, sequence_frames, args.held_out_camera_ids, args.held_out_frame_ids)
    with open(os.path.join(args.output, "training_split.json"), "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    temporal_pairs, pair_records = {}, []
    temporal_pair_counts = {}
    auxiliary_render_count = 0
    raster_sample_count = 0
    if temporal_photo_enabled:
        temporal_pairs, pair_records = build_temporal_image_pairs(cameras, sequence_frames)
        temporal_rng = random.Random(args.seed + 1)
        with open(os.path.join(args.output, "temporal_photo_protocol.json"), "w", encoding="utf-8") as f:
            json.dump({"mode": args.temporal_photo_mode, "weight": args.temporal_photo_weight,
                       "neighbor_seed": args.seed + 1, "eligible_training_pairs": pair_records,
                       "scope": "Same-cohort auxiliary image loss; no held-out observations or reference geometry."}, f, indent=2)
    if args.mode == "track":
        tracks = build_track_database(sequence_frames)
        init = initialize_from_tracks(
            tracks,
            min_track_len=args.min_track_len,
            normal_scale_ratio=args.normal_scale_ratio,
            spacing_k=args.spacing_k,
            max_track_samples=args.max_track_samples,
        )
    else:
        init = initialize_sand_surface(
            sequence_frames,
            normal_scale_ratio=args.normal_scale_ratio,
            spacing_k=args.spacing_k,
            max_points_per_frame=args.max_points_per_frame,
            flow_max_distance=args.flow_max_distance,
            flow_min_normal_dot=args.flow_min_normal_dot,
            velocity_source=args.velocity_source,
        )
    init["temporal_mode"] = args.temporal_mode
    if args.zero_initial_velocity:
        init["velocity"] = np.zeros_like(init["velocity"])
    model = DynamicGaussianModel(init, max_sh_degree=args.sh_degree, device=device)
    surface_support = build_surface_support(init, device, args.surface_origin_path) if args.surface_constraint else None
    density_support = build_surface_support(init, device) if args.densify_geometry_filter else None
    surface_history = []
    if args.refine_surface_tangents:
        surface_support["log_scales"] = model.scaling.detach().clone()
        lower = surface_support["log_scales"][:, 2:3] - np.log(args.normal_ratio_limit)
        if torch.any(lower > surface_support["log_scales"][:, :2]):
            raise ValueError("Initial surface scales cannot satisfy the tangent refinement bounds.")
    if surface_support is not None:
        np.savez_compressed(os.path.join(args.output, "surface_support.npz"),
                            **{name: value.cpu().numpy() for name, value in surface_support.items()})
    initial_frames, initial_counts = np.unique(model.frame_indices.detach().cpu().numpy(), return_counts=True)
    if args.adaptive_densification and initial_counts.max() > args.densify_max_per_frame:
        raise ValueError("Initial per-frame count exceeds the densification cap.")
    density_history = []
    density_stats = new_densification_stats(model) if args.adaptive_densification else None
    scene_extent = get_scene_extent(cameras)
    optimizer = model.optimizer(scene_extent, args.iterations, args.velocity_lr_time_scale)
    kappa_target = None
    if args.lambda_kappa_evidence > 0:
        evidence_cfg = EvidenceConfig(
            weights=EvidenceWeights(
                support=args.evidence_weight_support,
                reprojection=args.evidence_weight_reprojection,
                geometry=args.evidence_weight_geometry,
                temporal=args.evidence_weight_temporal,
            ),
            color_tau=args.evidence_color_tau,
            geometry_tau=args.evidence_geometry_tau,
            temporal_tau=args.evidence_temporal_tau,
            spacing_k=args.spacing_k,
            kappa_min=args.kappa_min,
            kappa_scale=args.kappa_scale,
        )
        targets = build_photogrammetry_targets(init, sequence_frames, cameras, evidence_cfg)
        kappa_target = torch.as_tensor(targets.kappa_target, dtype=torch.float32, device=device).reshape(-1, 1)
        np.savez_compressed(
            os.path.join(args.output, "photogrammetry_uncertainty_targets_initial.npz"
                         if args.adaptive_densification else "photogrammetry_uncertainty_targets.npz"),
            confidence=targets.confidence,
            kappa_target=targets.kappa_target,
            support_ratio=targets.support_ratio,
            reprojection_error=targets.reprojection_error,
            geometry_confidence=targets.geometry_confidence,
            temporal_residual=targets.temporal_residual,
        )

    znear, zfar = estimate_clip_planes(sequence_frames, cameras, scene_extent)
    depth_cache = cache_reference_depths(sequence_frames, cameras) if args.lambda_depth > 0 else {}
    frame_cloud_cache = cache_frame_cloud_tensors(sequence_frames, device)
    mask_cache = cache_masks(cameras, args.mask_dir, device) if args.lambda_silhouette > 0 else {}
    temporal_window = args.temporal_window if args.temporal_window > 0 else (0.75 / args.fps + args.default_exposure)
    temporal_sigma = args.temporal_sigma if args.temporal_sigma > 0 else max(temporal_window / 2.0, 1e-6)
    if args.temporal_mode == "nearest_frame" and args.temporal_window <= 0:
        times = sorted({frame.timestamp for frame in sequence_frames})
        largest_gap = max(np.diff(times)) if len(times) > 1 else 1.0 / args.fps
        longest_exposure = max(cam.exposure for cam in cameras)
        temporal_window = 0.5 * (largest_gap + longest_exposure) + 1e-6

    print(f"Loaded {len(cameras)} views, {len(sequence_frames)} sequence clouds, {model.n_gaussians} dynamic Gaussians.")
    print(f"Mode: {args.mode}. Temporal active window: {temporal_window:.6f}s.")
    print("Stages: 1 color/opacity, 2 anchored position, 3 velocity, 4 exposure integration, 5 optional scale refinement.")

    current_stage = None
    for iteration in range(args.iterations):
        stage = stage_for_iteration(iteration, args.iterations)
        if stage != current_stage:
            current_stage = stage
            model.set_train_stage(stage)
            if not args.learn_velocity:
                model.velocity.requires_grad_(False)
            if args.mode == "flow_sand" and args.freeze_surface_shape:
                model.scaling.requires_grad_(args.refine_surface_tangents and stage >= 5)
                model.rotation.requires_grad_(False)
            print(f"\nEntering stage {stage} at iter {iteration}.")
        if args.sh_up_interval > 0 and iteration > 0 and iteration % args.sh_up_interval == 0:
            model.oneup_sh_degree()

        cam = random.choice(cameras)
        use_exposure = stage >= 4
        render = render_exposure_integrated(
            model,
            cam,
            exposure_samples=args.exposure_samples,
            exposure_rule=args.exposure_rule,
            shutter_reference_mode=args.shutter_reference_mode,
            use_exposure=use_exposure,
            retain_screen_grad=args.adaptive_densification and iteration < args.densify_until,
            temporal_window=temporal_window if args.mode == "flow_sand" else None,
            temporal_sigma=temporal_sigma if args.mode == "flow_sand" else None,
            znear=znear,
            zfar=zfar,
        )
        raster_sample_count += args.exposure_samples if use_exposure and cam.exposure > 0 else 1

        photo, photo_parts = photometric_loss(render["image"], cam.image,
            lambda_dssim=args.lambda_dssim, objective=args.photometric_objective)
        loss = photo
        terms = {"photo": float(photo.detach())}
        temporal_render = None
        if temporal_photo_enabled and stage >= 3:
            neighbor = temporal_rng.choice(temporal_pairs[(cam.frame_idx, cam.camera_id)])
            target_cam = neighbor if args.temporal_photo_mode == "adjacent_train" else cam
            # This declared auxiliary horizon does not widen normal inference support.
            auxiliary_window = abs(target_cam.timestamp - cam.timestamp) + .5 * target_cam.exposure + 1e-6
            temporal_render = render_exposure_integrated(
                model, target_cam, exposure_samples=args.exposure_samples,
                exposure_rule=args.exposure_rule, shutter_reference_mode=args.shutter_reference_mode,
                reference_timestamp=cam.timestamp, use_exposure=use_exposure,
                retain_screen_grad=args.adaptive_densification and iteration < args.densify_until,
                temporal_window=auxiliary_window, temporal_sigma=temporal_sigma, znear=znear, zfar=zfar)
            auxiliary_photo, _ = photometric_loss(temporal_render["image"], target_cam.image,
                lambda_dssim=args.lambda_dssim, objective=args.photometric_objective)
            loss = (photo + args.temporal_photo_weight * auxiliary_photo) / (1. + args.temporal_photo_weight)
            terms["temporal_photo"] = float(auxiliary_photo.detach())
            auxiliary_render_count += 1
            raster_sample_count += args.exposure_samples if use_exposure and target_cam.exposure > 0 else 1
            key = (cam.frame_idx, target_cam.frame_idx, cam.camera_id)
            temporal_pair_counts[key] = temporal_pair_counts.get(key, 0) + 1

        if surface_support is not None and args.lambda_surface > 0:
            val = surface_support_loss(model, surface_support, render["active_indices"])
            loss = loss + args.lambda_surface * args.lambda_point_plane * val
            terms["support_plane"] = float(val.detach())
        elif args.mode == "flow_sand" and args.lambda_surface > 0:
            cloud = frame_cloud_cache.get(cam.frame_idx)
            if cloud is not None:
                val, parts = sand_surface_loss(
                    model,
                    cam.timestamp,
                    cloud["points"],
                    cloud["normals"],
                    temporal_window=temporal_window,
                    sample_count=args.surface_sample_count,
                    point_plane_weight=args.lambda_point_plane,
                    correspondence_mode=args.surface_correspondence,
                    objective=args.surface_objective,
                )
                loss = loss + args.lambda_surface * val
                terms["surface"] = float(val.detach())
                terms["plane"] = parts["point_plane"]
        if args.mode == "track" and stage >= 2 and args.lambda_anchor > 0:
            val = anchor_loss(model, cam.timestamp, sample_count=args.loss_sample_count)
            loss = loss + args.lambda_anchor * val
            terms["anchor"] = float(val.detach())
        if args.mode == "track" and stage >= 3 and args.lambda_track > 0:
            val = track_sample_loss(model, sample_count=args.loss_sample_count)
            loss = loss + args.lambda_track * val
            terms["track"] = float(val.detach())
        if stage >= 3 and args.lambda_motion > 0:
            val = velocity_regularization(model, args.velocity_prior_time_scale)
            loss = loss + args.lambda_motion * val
            terms["motion"] = float(val.detach())
        if args.lambda_depth > 0:
            ref_depth = depth_cache.get((cam.frame_idx, cam.camera_id))
            if ref_depth is not None:
                val = depth_loss(render["depth"], ref_depth)
                loss = loss + args.lambda_depth * val
                terms["depth"] = float(val.detach())
        if args.lambda_silhouette > 0:
            mask = mask_cache.get(cam.image_path)
            if mask is not None:
                val = silhouette_loss(render["alpha"], mask)
                loss = loss + args.lambda_silhouette * val
                terms["sil"] = float(val.detach())
        if (args.mode == "flow_sand" or stage >= 5) and args.lambda_scale > 0:
            max_scale = args.max_scale if args.max_scale > 0 else 0.05 * scene_extent
            val = scale_regularization(
                model,
                args.max_anisotropy,
                max_scale,
                args.normal_ratio_limit,
                args.lambda_rotation_anchor,
            )
            loss = loss + args.lambda_scale * val
            terms["scale"] = float(val.detach())
        if kappa_target is not None:
            current_target = kappa_target[model.source_indices] if args.adaptive_densification else kappa_target
            val = kappa_evidence_loss(model, current_target)
            loss = loss + args.lambda_kappa_evidence * val
            terms["kappa"] = float(val.detach())
        if args.lambda_beta_kl > 0:
            val = beta_kl_loss(model, prior_mu=args.beta_prior_mu, prior_kappa=args.beta_prior_kappa)
            loss = loss + args.lambda_beta_kl * val
            terms["beta_kl"] = float(val.detach())
        if args.lambda_beta_prior > 0:
            val = beta_concentration_prior_loss(model, prior_kappa=args.beta_prior_kappa)
            loss = loss + args.lambda_beta_prior * val
            terms["beta_prior"] = float(val.detach())

        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite training loss at iteration {iteration}.")
        loss.backward()
        if args.refine_surface_tangents and model.scaling.grad is not None:
            model.scaling.grad[:, 2] = 0
        if args.adaptive_densification and iteration < args.densify_until:
            accumulate_screen_gradients(density_stats, render["screen_samples"])
            if temporal_render is not None:
                accumulate_screen_gradients(density_stats, temporal_render["screen_samples"])
        optimizer.step()
        if args.refine_surface_tangents and stage >= 5:
            project_surface_scales(model, optimizer, surface_support, args)
        if surface_support is not None and stage >= 2:
            report = args.log_interval > 0 and iteration % args.log_interval == 0
            projection = project_surface_centers(model, optimizer, surface_support, args, report=report)
            if report:
                surface_history.append({"iteration": iteration + 1, **projection})

        if args.log_interval > 0 and iteration % args.log_interval == 0:
            with torch.no_grad():
                mse = torch.mean((render["image"] - cam.image) ** 2)
                psnr = -10.0 * torch.log10(mse + 1e-10)
            detail = " ".join(f"{k}={v:.4f}" for k, v in terms.items())
            print(
                f"Iter {iteration:06d}/{args.iterations} stage={stage} "
                f"loss={loss.item():.4f} psnr={psnr.item():.2f} {detail}"
            )

        step = iteration + 1
        if (args.adaptive_densification and args.densify_from <= step <= args.densify_until
                and step % args.densify_interval == 0):
            parents, frame_rows = select_split_parents(model, density_stats, args, density_support)
            optimizer.zero_grad(set_to_none=True)
            if args.audit_split_geometry and len(parents):
                split_audit = {
                    "parent_mu_ref": model.mu_ref[parents].detach().cpu().numpy().copy(),
                    "parent_scales": model.scaling[parents].detach().exp().cpu().numpy().copy(),
                    "parent_rotation": model.rotation[parents].detach().cpu().numpy().copy(),
                    "parent_source_indices": model.source_indices[parents].detach().cpu().numpy().copy(),
                    "parent_track_ids": model.track_ids[parents].detach().cpu().numpy().copy(),
                    "parent_frame_indices": model.frame_indices[parents].detach().cpu().numpy().copy(),
                    "parent_generation": model.generation[parents].detach().cpu().numpy().copy(),
                    "parent_t_ref": model.t_ref[parents].detach().cpu().numpy().copy(),
                }
            split_surface_gaussians(model, optimizer, parents, surface_support, args)
            if args.refine_surface_tangents and stage >= 5:
                project_surface_scales(model, optimizer, surface_support, args)
            if args.audit_split_geometry and len(parents):
                child_start = model.n_gaussians - 2 * len(parents)
                split_audit["child_mu_ref"] = model.mu_ref[child_start:].detach().cpu().numpy()
                split_audit["child_track_ids"] = model.track_ids[child_start:].detach().cpu().numpy()
                split_audit["child_parent_ids"] = model.parent_ids[child_start:].detach().cpu().numpy()
                split_audit["child_source_indices"] = model.source_indices[child_start:].detach().cpu().numpy()
                split_audit["child_generation"] = model.generation[child_start:].detach().cpu().numpy()
                audit_folder = os.path.join(args.output, "split_geometry_audit")
                os.makedirs(audit_folder, exist_ok=True)
                np.savez_compressed(os.path.join(audit_folder, f"step_{step:06d}.npz"), **split_audit)
            if surface_support is not None:
                projection = project_surface_centers(model, optimizer, surface_support, args, report=True)
                surface_history.append({"iteration": step, "after_split": True, **projection})
            density_history.append({"iteration": step, "split_parents": len(parents),
                                    "total_gaussians": model.n_gaussians, "frames": frame_rows})
            with open(os.path.join(args.output, "densification_history.json"), "w", encoding="utf-8") as f:
                json.dump(density_history, f, indent=2)
            print(f"Densify {step}: split {len(parents)} parents -> {model.n_gaussians} Gaussians.", flush=True)
            density_stats = new_densification_stats(model)

        if args.save_interval > 0 and iteration > 0 and iteration % args.save_interval == 0:
            model.save_npz(os.path.join(args.output, f"checkpoint_{iteration:06d}.npz"))
            save_ply_at_time(
                model,
                os.path.join(args.output, f"frame_{cam.frame_idx:06d}_checkpoint.ply"),
                cam.timestamp,
                temporal_window=temporal_window if args.mode == "flow_sand" else None,
            )

    if temporal_photo_enabled:
        used_pairs = []
        for (source_frame, target_frame, camera_id), count in sorted(temporal_pair_counts.items()):
            used_pairs.append({"source_frame": source_frame, "target_frame": target_frame,
                               "camera": camera_id, "count": count})
        with open(os.path.join(args.output, "temporal_photo_counts.json"), "w", encoding="utf-8") as f:
            json.dump({"auxiliary_render_calls": auxiliary_render_count,
                       "total_rasterizer_samples": raster_sample_count, "used_training_pairs": used_pairs}, f, indent=2)
    if args.adaptive_densification and kappa_target is not None:
        source = model.source_indices.detach().cpu().numpy()
        np.savez_compressed(
            os.path.join(args.output, "photogrammetry_uncertainty_targets.npz"),
            confidence=targets.confidence[source], kappa_target=targets.kappa_target[source],
            support_ratio=targets.support_ratio[source], reprojection_error=targets.reprojection_error[source],
            geometry_confidence=targets.geometry_confidence[source], temporal_residual=targets.temporal_residual[source],
            source_indices=source,
        )
    final_frames, final_counts = np.unique(model.frame_indices.detach().cpu().numpy(), return_counts=True)
    density_summary = {"initial_total": int(initial_counts.sum()), "final_total": model.n_gaussians,
                       "initial_per_frame": dict(zip(initial_frames.tolist(), initial_counts.tolist())),
                       "final_per_frame": dict(zip(final_frames.tolist(), final_counts.tolist())),
                       "split_events": len(density_history),
                       "evidence_transfer": "Children inherit source targets; no new independent evidence."}
    with open(os.path.join(args.output, "density_summary.json"), "w", encoding="utf-8") as f:
        json.dump(density_summary, f, indent=2)
    if surface_support is not None:
        with open(os.path.join(args.output, "surface_constraint_history.json"), "w", encoding="utf-8") as f:
            json.dump({"scope": "Centers at their own reference times; not a continuous-motion guarantee.",
                       "support": "Original training points/normals, unchanged by splitting.",
                       "logged_projection_steps": surface_history}, f, indent=2)
    model.save_npz(os.path.join(args.output, "dynamic_gaussians_final.npz"))
    unique_times = sorted({(cam.frame_idx, cam.timestamp) for cam in cameras})
    for frame_idx, timestamp in unique_times[:: max(1, args.export_stride)]:
        save_ply_at_time(
            model,
            os.path.join(args.output, f"corrected_frame_{frame_idx:06d}.ply"),
            timestamp,
            temporal_window=temporal_window if args.mode == "flow_sand" else None,
        )
    print(f"Training complete. Outputs written to {args.output}")
    return model, {
        "exposure_samples": args.exposure_samples,
        "exposure_rule": args.exposure_rule,
        "shutter_reference_mode": args.shutter_reference_mode,
        "temporal_window": temporal_window if args.mode == "flow_sand" else None,
        "temporal_sigma": temporal_sigma if args.mode == "flow_sand" else None,
        "znear": znear,
        "zfar": zfar,
    }


from demo_config import load_training_values, new_output

# Spyder F5 settings. Use "full" for the saved 3000-iteration configuration.
EXPERIMENT = "translation"
PROFILE = "quick"
QUICK_ITERATIONS = 120

_values = load_training_values(EXPERIMENT)
if PROFILE == "quick":
    _values.update(iterations=QUICK_ITERATIONS, log_interval=30,
                   densify_from=40, densify_until=100, densify_interval=20)
elif PROFILE != "full":
    raise ValueError("PROFILE must be quick or full.")
IDE_CONFIG = TrainConfig(**_values)


def main(config: TrainConfig | None = None):
    if config is None:
        output = new_output("train_" + EXPERIMENT + "_" + PROFILE)
        config = replace(IDE_CONFIG, output=str(output))
    model, settings = train(config)
    with open(Path(config.output) / "render_settings.json", "w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2)
    return model, settings


if __name__ == "__main__":
    main()
