# -*- coding: utf-8 -*-
"""Data loading and initialization helpers for dynamic 4D Gaussian Splatting."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

from ply_utils import read_vertex_ply


@dataclass
class CameraInfo:
    image: torch.Tensor
    image_path: str
    w2c: torch.Tensor
    camera_center: torch.Tensor
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    fovx: float
    fovy: float
    frame_idx: int
    timestamp: float
    camera_id: int
    exposure: float
    raster_pixel_offset: float = 0.0  # Legacy replay; 0.5 for integer-centered OpenCV coordinates.


@dataclass
class SequenceFrame:
    frame_idx: int
    timestamp: float
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    track_ids: np.ndarray
    confidence: np.ndarray
    velocity: np.ndarray | None = None  # Optional measured scene flow, world-coordinate units/s.


def estimate_normals(points: np.ndarray, k: int = 16) -> np.ndarray:
    """Estimate unoriented surface normals with local PCA."""
    if points.shape[0] < 4:
        return np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (points.shape[0], 1))
    tree = cKDTree(points)
    kk = min(max(k, 4), points.shape[0])
    _, idx = tree.query(points, k=kk)
    normals = np.zeros_like(points, dtype=np.float32)
    for i in range(points.shape[0]):
        neigh = points[idx[i]]
        cov = np.cov((neigh - neigh.mean(axis=0)).T)
        _, vecs = np.linalg.eigh(cov)
        normals[i] = vecs[:, 0]
    return normalize_vectors(normals, np.array([0.0, 0.0, 1.0], dtype=np.float32))


def _read_ply_cloud(path: Path) -> dict:
    vertex = read_vertex_ply(path)
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    names = set(vertex.keys())
    if {"red", "green", "blue"}.issubset(names):
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32) / 255.0
    else:
        colors = np.ones_like(points, dtype=np.float32)
    if {"nx", "ny", "nz"}.issubset(names):
        normals = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=1).astype(np.float32)
    else:
        normals = estimate_normals(points)
    track_ids = np.asarray(vertex["track_id"], dtype=np.int64) if "track_id" in names else None
    confidence = np.asarray(vertex["confidence"], dtype=np.float32) if "confidence" in names else np.ones(points.shape[0], dtype=np.float32)
    return {"points": points, "colors": colors, "normals": normals, "track_ids": track_ids, "confidence": confidence}


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_manifest(path: Optional[str]) -> Dict[str, dict]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "views" in data:
        rows = data["views"]
    elif isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = [{"filename": k, **v} for k, v in data.items()]
    else:
        raise ValueError("views manifest must be a list, dict, or {'views': list}")

    manifest = {}
    for row in rows:
        name = row.get("filename") or row.get("image") or row.get("path")
        if not name:
            continue
        manifest[os.path.basename(name)] = row
    return manifest


def _infer_cam_frame(filename: str, view_index: int, fps: float, num_cameras: int) -> Tuple[int, int, float]:
    stem = Path(filename).stem.lower()
    patterns = [
        r"cam[_-]?(\d+).*?(?:frame|frm|f)[_-]?(\d+)",
        r"(?:frame|frm|f)[_-]?(\d+).*?cam[_-]?(\d+)",
        r"c[_-]?(\d+).*?t[_-]?(\d+)",
        r"t[_-]?(\d+).*?c[_-]?(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, stem)
        if not m:
            continue
        a = int(m.group(1))
        b = int(m.group(2))
        if pat.startswith("(?:frame") or pat.startswith("t"):
            frame_idx, camera_id = a, b
        else:
            camera_id, frame_idx = a, b
        return camera_id, frame_idx, frame_idx / fps

    numbers = [int(x) for x in re.findall(r"\d+", stem)]
    if len(numbers) >= 2:
        return numbers[0], numbers[-1], numbers[-1] / fps
    safe_num_cameras = max(int(num_cameras), 1)
    camera_id = view_index % safe_num_cameras
    frame_idx = view_index // safe_num_cameras
    return camera_id, frame_idx, frame_idx / fps


def load_exposure_config(path: Optional[str], default_exposure: float) -> dict:
    if not path:
        return {"default": float(default_exposure), "frames": {}, "cameras": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "default": float(data.get("default", default_exposure)),
        "frames": data.get("frames", {}),
        "cameras": data.get("cameras", {}),
    }


def lookup_exposure(exposure_cfg: dict, frame_idx: int, camera_id: int) -> float:
    frame_key = str(frame_idx)
    cam_key = str(camera_id)
    if frame_key in exposure_cfg.get("frames", {}):
        val = exposure_cfg["frames"][frame_key]
        if isinstance(val, dict):
            return _safe_float(val.get(cam_key), exposure_cfg["default"])
        return _safe_float(val, exposure_cfg["default"])
    if cam_key in exposure_cfg.get("cameras", {}):
        return _safe_float(exposure_cfg["cameras"][cam_key], exposure_cfg["default"])
    return float(exposure_cfg.get("default", 0.0))


def load_openmvg_sequence(
    json_path: str,
    views_manifest: Optional[str] = None,
    exposure_json: Optional[str] = None,
    default_exposure: float = 0.0,
    fps: float = 30.0,
    num_cameras: int = 3,
    downsample_factor: float = 0.0,
    target_width: int = 1600,
    prefer_undistorted: bool = True,
    device: str = "cuda",
) -> List[CameraInfo]:
    """Load OpenMVG cameras and attach frame/time metadata for a sequence."""
    with open(json_path, "r", encoding="utf-8") as f:
        sfm_data = json.load(f)

    root_path = sfm_data.get("root_path", "")
    json_dir = os.path.dirname(os.path.abspath(json_path))
    if root_path and not os.path.isabs(root_path):
        root_path = os.path.normpath(os.path.join(json_dir, root_path))
    undistorted_root = os.path.join(json_dir, "scene_undistorted_images")
    manifest = _load_manifest(views_manifest)
    exposure_cfg = load_exposure_config(exposure_json, default_exposure)

    pixel_convention = sfm_data.get("pixel_coordinates", "legacy")
    if pixel_convention not in ("legacy", "opencv_integer_centers"):
        raise ValueError(f"Unsupported pixel coordinate convention: {pixel_convention}")
    raster_pixel_offset = 0.5 if pixel_convention == "opencv_integer_centers" else 0.0

    intrinsics = {}
    for intrinsic in sfm_data.get("intrinsics", []):
        key = intrinsic["key"]
        data = intrinsic["value"]["ptr_wrapper"]["data"]
        focal = float(data["focal_length"])
        intrinsics[key] = {
            "fx": float(data.get("focal_length_x", focal)),
            "fy": float(data.get("focal_length_y", focal)),
            "cx": float(data["principal_point"][0]),
            "cy": float(data["principal_point"][1]),
        }

    extrinsics = {}
    for extrinsic in sfm_data.get("extrinsics", []):
        key = extrinsic["key"]
        val = extrinsic["value"]
        rotation = np.asarray(val["rotation"], dtype=np.float32)
        center = np.asarray(val["center"], dtype=np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rotation.T
        c2w[:3, 3] = center
        extrinsics[key] = c2w

    cameras: List[CameraInfo] = []
    for view_index, view in enumerate(sfm_data.get("views", [])):
        data = view["value"]["ptr_wrapper"]["data"]
        filename = data["filename"]
        img_path = os.path.join(root_path, data.get("local_path", ""), filename)
        if not os.path.exists(img_path):
            json_relative = os.path.join(json_dir, data.get("local_path", ""), filename)
            if os.path.exists(json_relative):
                img_path = json_relative
        undistorted_candidate = os.path.join(undistorted_root, filename)
        if prefer_undistorted and os.path.exists(undistorted_candidate):
            img_path = undistorted_candidate
        if not os.path.exists(img_path):
            continue

        pose_id = data["id_pose"]
        intrinsic_id = data["id_intrinsic"]
        if pose_id not in extrinsics or intrinsic_id not in intrinsics:
            continue

        row = manifest.get(os.path.basename(filename), {})
        if row:
            camera_id = int(row.get("camera_id", row.get("cam", 0)))
            frame_idx = int(row.get("frame_idx", row.get("frame", view_index)))
            timestamp = _safe_float(row.get("timestamp", row.get("time")), frame_idx / fps)
        else:
            camera_id, frame_idx, timestamp = _infer_cam_frame(filename, view_index, fps, num_cameras)

        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.width, img_pil.height
        if downsample_factor and downsample_factor > 0:
            resize_factor = max(float(downsample_factor), 1.0)
        else:
            resize_factor = max(1.0, orig_w / float(target_width))
        width = max(1, int(round(orig_w / resize_factor)))
        height = max(1, int(round(orig_h / resize_factor)))
        if width != orig_w or height != orig_h:
            img_pil = img_pil.resize((width, height), Image.Resampling.LANCZOS)
        image = (
            torch.tensor(np.asarray(img_pil) / 255.0, dtype=torch.float32)
            .permute(2, 0, 1)
            .to(device)
        )

        c2w = torch.tensor(extrinsics[pose_id], dtype=torch.float32, device=device)
        w2c = torch.inverse(c2w)
        K = intrinsics[intrinsic_id].copy()
        sx = width / float(orig_w)
        sy = height / float(orig_h)
        K["fx"] *= sx
        K["fy"] *= sy
        if pixel_convention == "opencv_integer_centers":
            K["cx"] = (K["cx"] + 0.5) * sx - 0.5
            K["cy"] = (K["cy"] + 0.5) * sy - 0.5
        else:
            K["cx"] *= sx
            K["cy"] *= sy
        exposure = row.get("exposure", None)
        if exposure is None:
            exposure = lookup_exposure(exposure_cfg, frame_idx, camera_id)

        cameras.append(
            CameraInfo(
                image=image,
                image_path=img_path,
                w2c=w2c,
                camera_center=c2w[:3, 3],
                fx=K["fx"],
                fy=K["fy"],
                cx=K["cx"],
                cy=K["cy"],
                width=width,
                height=height,
                fovx=2 * math.atan(width / (2 * K["fx"])),
                fovy=2 * math.atan(height / (2 * K["fy"])),
                frame_idx=frame_idx,
                timestamp=float(timestamp),
                camera_id=camera_id,
                exposure=float(exposure),
                raster_pixel_offset=raster_pixel_offset,
            )
        )

    cameras.sort(key=lambda c: (c.frame_idx, c.camera_id, c.image_path))
    return cameras


def load_fixed_rig_continue_frame(
    robust_json: str,
    data_root: str,
    exposure_json: Optional[str] = None,
    default_exposure: float = 0.0,
    fps: float = 30.0,
    downsample_factor: float = 0.0,
    target_width: int = 1600,
    device: str = "cuda",
) -> List[CameraInfo]:
    """
    Load a fixed multi-camera rig sequence stored as:

        data_root/
          robust.json
          249/image_json_000.jpg
          249/image_json_001.jpg
          249/image_json_002.jpg
          250/...

    The same intrinsics/extrinsics from robust.json are reused for every frame.
    """
    with open(robust_json, "r", encoding="utf-8") as f:
        sfm_data = json.load(f)
    data_root = os.path.abspath(data_root)
    exposure_cfg = load_exposure_config(exposure_json, default_exposure)

    intrinsics = {}
    for intrinsic in sfm_data.get("intrinsics", []):
        key = intrinsic["key"]
        data = intrinsic["value"]["ptr_wrapper"]["data"]
        focal = float(data["focal_length"])
        intrinsics[key] = {
            "fx": focal,
            "fy": focal,
            "cx": float(data["principal_point"][0]),
            "cy": float(data["principal_point"][1]),
        }

    extrinsics = {}
    for extrinsic in sfm_data.get("extrinsics", []):
        key = extrinsic["key"]
        val = extrinsic["value"]
        rotation = np.asarray(val["rotation"], dtype=np.float32)
        center = np.asarray(val["center"], dtype=np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rotation.T
        c2w[:3, 3] = center
        extrinsics[key] = c2w

    view_rows = []
    for view in sfm_data.get("views", []):
        data = view["value"]["ptr_wrapper"]["data"]
        view_rows.append(
            {
                "filename": data["filename"],
                "camera_id": int(data["id_view"]),
                "intrinsic_id": data["id_intrinsic"],
                "pose_id": data["id_pose"],
            }
        )
    view_rows.sort(key=lambda x: x["camera_id"])

    frame_dirs = [p for p in Path(data_root).iterdir() if p.is_dir() and p.name.isdigit()]
    frame_dirs.sort(key=lambda p: int(p.name))
    cameras: List[CameraInfo] = []
    for frame_dir in frame_dirs:
        frame_idx = int(frame_dir.name)
        timestamp = frame_idx / fps
        for row in view_rows:
            img_path = frame_dir / row["filename"]
            if not img_path.exists():
                continue
            if row["pose_id"] not in extrinsics or row["intrinsic_id"] not in intrinsics:
                continue

            img_pil = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img_pil.width, img_pil.height
            if downsample_factor and downsample_factor > 0:
                resize_factor = max(float(downsample_factor), 1.0)
            else:
                resize_factor = max(1.0, orig_w / float(target_width))
            width = max(1, int(round(orig_w / resize_factor)))
            height = max(1, int(round(orig_h / resize_factor)))
            if width != orig_w or height != orig_h:
                img_pil = img_pil.resize((width, height), Image.Resampling.LANCZOS)
            image = (
                torch.tensor(np.asarray(img_pil) / 255.0, dtype=torch.float32)
                .permute(2, 0, 1)
                .to(device)
            )

            c2w = torch.tensor(extrinsics[row["pose_id"]], dtype=torch.float32, device=device)
            w2c = torch.inverse(c2w)
            K = intrinsics[row["intrinsic_id"]].copy()
            sx = width / float(orig_w)
            sy = height / float(orig_h)
            K["fx"] *= sx
            K["fy"] *= sy
            K["cx"] *= sx
            K["cy"] *= sy
            exposure = lookup_exposure(exposure_cfg, frame_idx, row["camera_id"])

            cameras.append(
                CameraInfo(
                    image=image,
                    image_path=str(img_path),
                    w2c=w2c,
                    camera_center=c2w[:3, 3],
                    fx=K["fx"],
                    fy=K["fy"],
                    cx=K["cx"],
                    cy=K["cy"],
                    width=width,
                    height=height,
                    fovx=2 * math.atan(width / (2 * K["fx"])),
                    fovy=2 * math.atan(height / (2 * K["fy"])),
                    frame_idx=frame_idx,
                    timestamp=float(timestamp),
                    camera_id=row["camera_id"],
                    exposure=float(exposure),
                )
            )
    cameras.sort(key=lambda c: (c.frame_idx, c.camera_id))
    return cameras


def load_sequence_clouds(sequence_dir: str, fps: float = 30.0) -> List[SequenceFrame]:
    files = sorted(Path(sequence_dir).glob("frame_*.npz"))
    if not files:
        files = sorted(Path(sequence_dir).glob("*.npz"))
    if not files:
        files = sorted(Path(sequence_dir).glob("frame_*.ply"))
    if not files:
        files = sorted(Path(sequence_dir).glob("*.ply"))
    if not files:
        files = sorted(Path(sequence_dir).glob("*/robust_scene_dense.ply"))
    frames = []
    for order, path in enumerate(files):
        velocity = None
        if path.suffix.lower() == ".ply":
            cloud = _read_ply_cloud(path)
            points = cloud["points"]
            colors = cloud["colors"]
            normals = cloud["normals"]
            track_ids = cloud["track_ids"]
            conf = cloud["confidence"]
            timestamp_from_file = None
        else:
            data = np.load(path)
            points = np.asarray(data["points"], dtype=np.float32)
            colors = np.asarray(data.get("colors", np.ones_like(points)), dtype=np.float32)
            if colors.max() > 1.5:
                colors = colors / 255.0
            if "normals" in data:
                normals = np.asarray(data["normals"], dtype=np.float32)
            else:
                normals = estimate_normals(points)
            track_ids = np.asarray(data["track_ids"], dtype=np.int64) if "track_ids" in data else None
            conf = np.asarray(data.get("confidence", np.ones(points.shape[0])), dtype=np.float32)
            timestamp_from_file = float(data["time"]) if "time" in data else None
            if "velocity" in data:
                velocity = np.asarray(data["velocity"], dtype=np.float32)
                if velocity.shape != points.shape or not np.isfinite(velocity).all():
                    raise ValueError(f"Invalid velocity field in {path}")
        if normals.shape != points.shape or np.linalg.norm(normals, axis=1).mean() < 1e-6:
            normals = estimate_normals(points)
        normals = normalize_vectors(normals, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        if track_ids is None:
            track_ids = np.full(points.shape[0], -1, dtype=np.int64)
        m = re.search(r"(\d+)", path.stem)
        if m is None and path.parent.name.isdigit():
            m = re.search(r"(\d+)", path.parent.name)
        frame_idx = int(m.group(1)) if m else order
        timestamp = timestamp_from_file if timestamp_from_file is not None else frame_idx / fps
        frames.append(
            SequenceFrame(
                frame_idx=frame_idx,
                timestamp=timestamp,
                points=points,
                colors=colors,
                normals=normals,
                track_ids=track_ids,
                confidence=conf,
                velocity=velocity,
            )
        )
    return frames


def build_track_database(frames: Iterable[SequenceFrame]) -> Dict[int, List[dict]]:
    tracks: Dict[int, List[dict]] = {}
    for frame in frames:
        for j, tid in enumerate(frame.track_ids):
            if int(tid) < 0:
                continue
            tracks.setdefault(int(tid), []).append(
                {
                    "frame_idx": frame.frame_idx,
                    "t": frame.timestamp,
                    "point": frame.points[j],
                    "color": frame.colors[j],
                    "normal": frame.normals[j],
                    "confidence": frame.confidence[j],
                }
            )
    for samples in tracks.values():
        samples.sort(key=lambda x: x["t"])
    return tracks


def _frame_spacing(points: np.ndarray, spacing_k: int) -> np.ndarray:
    if points.shape[0] <= spacing_k:
        return np.full(points.shape[0], 1e-3, dtype=np.float32)
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(spacing_k + 1, points.shape[0]))
    return np.clip(np.mean(distances[:, 1:], axis=1).astype(np.float32), 1e-6, None)


def _estimate_pair_flow(src: SequenceFrame, dst: SequenceFrame, max_distance: float, min_normal_dot: float) -> np.ndarray:
    """Short scene-flow estimate from src to dst with nearest-neighbor gating."""
    if src.points.shape[0] == 0 or dst.points.shape[0] == 0:
        return np.zeros_like(src.points, dtype=np.float32)
    tree = cKDTree(dst.points)
    dist, idx = tree.query(src.points, k=1)
    matched = dist < max_distance
    normal_dot = np.sum(src.normals * dst.normals[idx], axis=1)
    matched &= normal_dot > min_normal_dot
    flow = np.zeros_like(src.points, dtype=np.float32)
    dt = float(dst.timestamp - src.timestamp)
    if abs(dt) < 1e-8:
        dt = 1e-8
    flow[matched] = (dst.points[idx[matched]] - src.points[matched]) / dt
    if matched.any():
        fallback = np.median(flow[matched], axis=0)
        flow[~matched] = fallback
    return flow


def initialize_sand_surface(
    frames: List[SequenceFrame],
    normal_scale_ratio: float = 0.08,
    spacing_k: int = 6,
    max_points_per_frame: int = 20000,
    flow_max_distance: float = 0.02,
    flow_min_normal_dot: float = 0.35,
    velocity_source: str = "nearest_cloud",
) -> dict:
    """
    Initialize short-lived free-surface Gaussians for sand-like non-rigid flow.

    Each input point becomes a surfel centered at its own frame time. It does not
    claim long-term identity; it only carries a short scene-flow velocity for
    exposure-time blur.
    """
    if not frames:
        raise RuntimeError("No sequence clouds found.")
    frames = sorted(frames, key=lambda f: f.timestamp)
    if velocity_source not in ("nearest_cloud", "input"):
        raise ValueError(f"Unknown velocity source: {velocity_source}")
    if velocity_source == "input" and any(frame.velocity is None for frame in frames):
        raise ValueError("Input velocity requested, but a training cloud has no velocity field.")
    forward_flow = {}
    backward_flow = {}
    if velocity_source == "nearest_cloud":
        for i in range(len(frames) - 1):
            forward_flow[i] = _estimate_pair_flow(frames[i], frames[i + 1], flow_max_distance, flow_min_normal_dot)
        for i in range(1, len(frames)):
            backward_flow[i] = _estimate_pair_flow(frames[i], frames[i - 1], flow_max_distance, flow_min_normal_dot)

    xyz_list = []
    vel_list = []
    tref_list = []
    color_list = []
    scale_list = []
    rot_list = []
    frame_idx_list = []
    confidence_list = []

    rng = np.random.default_rng(42)
    for i, frame in enumerate(frames):
        n = frame.points.shape[0]
        if n == 0:
            continue
        if max_points_per_frame > 0 and n > max_points_per_frame:
            keep = np.sort(rng.choice(n, size=max_points_per_frame, replace=False))
        else:
            keep = np.arange(n)
        spacing = _frame_spacing(frame.points, spacing_k)[keep]
        scales = np.stack([spacing, spacing, np.maximum(spacing * normal_scale_ratio, 1e-7)], axis=1)
        rotations = np.stack([normal_to_quaternion(nrm) for nrm in frame.normals[keep]], axis=0)
        if velocity_source == "input":
            vel = frame.velocity[keep]
        elif i in forward_flow:
            vel = forward_flow[i][keep]
        elif i in backward_flow and backward_flow[i] is not None:
            vel = backward_flow[i][keep]
        else:
            vel = np.zeros((keep.shape[0], 3), dtype=np.float32)

        xyz_list.append(frame.points[keep])
        vel_list.append(vel.astype(np.float32))
        tref_list.append(np.full(keep.shape[0], frame.timestamp, dtype=np.float32))
        color_list.append(np.clip(frame.colors[keep], 0.0, 1.0))
        scale_list.append(scales.astype(np.float32))
        rot_list.append(rotations.astype(np.float32))
        frame_idx_list.append(np.full(keep.shape[0], frame.frame_idx, dtype=np.int64))
        confidence_list.append(frame.confidence[keep].astype(np.float32))

    if not xyz_list:
        raise RuntimeError("All sequence clouds are empty.")
    xyz = np.concatenate(xyz_list, axis=0)
    t_ref = np.concatenate(tref_list, axis=0)
    n_total = xyz.shape[0]
    return {
        "mode": "flow_sand",
        "track_ids": np.arange(n_total, dtype=np.int64),
        "mu_ref": xyz.astype(np.float32),
        "velocity": np.concatenate(vel_list, axis=0).astype(np.float32),
        "t_ref": t_ref.astype(np.float32),
        "colors": np.concatenate(color_list, axis=0).astype(np.float32),
        "scales": np.concatenate(scale_list, axis=0).astype(np.float32),
        "rotations": np.concatenate(rot_list, axis=0).astype(np.float32),
        "confidence": np.concatenate(confidence_list, axis=0).astype(np.float32),
        "frame_indices": np.concatenate(frame_idx_list, axis=0),
        "track_times": t_ref.reshape(-1, 1).astype(np.float32),
        "track_points": xyz.reshape(-1, 1, 3).astype(np.float32),
        "track_mask": np.ones((n_total, 1), dtype=np.float32),
    }


def normalize_vectors(vectors: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    out = vectors.copy()
    bad = norms.squeeze(-1) < 1e-8
    out[~bad] = out[~bad] / norms[~bad]
    out[bad] = fallback
    return out


def rotation_matrix_to_quaternion(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to [w, x, y, z]."""
    m = rot
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s],
            dtype=np.float32,
        )
    idx = int(np.argmax(np.diag(m)))
    if idx == 0:
        s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-8)) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif idx == 1:
        s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-8)) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-8)) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return np.asarray(q, dtype=np.float32)


def normal_to_quaternion(normal: np.ndarray) -> np.ndarray:
    n = normal.astype(np.float32)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        n = n / n_norm
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(helper, n))) > 0.9:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    t1 = np.cross(helper, n)
    t1 = t1 / max(np.linalg.norm(t1), 1e-8)
    t2 = np.cross(n, t1)
    rot = np.stack([t1, t2, n], axis=1)
    q = rotation_matrix_to_quaternion(rot)
    return q / max(np.linalg.norm(q), 1e-8)


def fit_linear_track(samples: List[dict]) -> Tuple[float, np.ndarray, np.ndarray]:
    times = np.asarray([s["t"] for s in samples], dtype=np.float32)
    points = np.asarray([s["point"] for s in samples], dtype=np.float32)
    weights = np.asarray([s.get("confidence", 1.0) for s in samples], dtype=np.float32)
    weights = np.clip(weights, 1e-4, None)
    t_ref = float(np.average(times, weights=weights))
    dt = times - t_ref
    denom = float(np.sum(weights * dt * dt))
    if denom < 1e-10:
        velocity = np.zeros(3, dtype=np.float32)
    else:
        centered = points - np.average(points, axis=0, weights=weights)
        velocity = np.sum(weights[:, None] * dt[:, None] * centered, axis=0) / denom
    mu_ref = np.average(points - velocity[None, :] * dt[:, None], axis=0, weights=weights)
    return t_ref, mu_ref.astype(np.float32), velocity.astype(np.float32)


def initialize_from_tracks(
    tracks: Dict[int, List[dict]],
    min_track_len: int = 2,
    normal_scale_ratio: float = 0.2,
    spacing_k: int = 6,
    max_track_samples: int = 32,
) -> dict:
    """Create Gaussian tensors from temporally linked sequence-cloud tracks."""
    rows = []
    for tid, samples in tracks.items():
        if len(samples) < min_track_len:
            continue
        t_ref, mu_ref, velocity = fit_linear_track(samples)
        mid = min(range(len(samples)), key=lambda i: abs(samples[i]["t"] - t_ref))
        colors = np.asarray([s["color"] for s in samples], dtype=np.float32)
        normals = np.asarray([s["normal"] for s in samples], dtype=np.float32)
        conf = np.asarray([s["confidence"] for s in samples], dtype=np.float32)
        normal = np.average(normals, axis=0, weights=np.clip(conf, 1e-4, None))
        normal = normalize_vectors(normal[None, :], np.array([0.0, 0.0, 1.0], dtype=np.float32))[0]
        rows.append(
            {
                "track_id": tid,
                "t_ref": t_ref,
                "mu_ref": mu_ref,
                "velocity": velocity,
                "color": np.median(colors, axis=0),
                "normal": normal,
                "confidence": float(np.mean(conf)),
                "samples": samples,
                "frame_ref": samples[mid]["frame_idx"],
            }
        )

    if not rows:
        raise RuntimeError("No valid tracks found. Check track_ids and min_track_len.")

    xyz = np.stack([r["mu_ref"] for r in rows], axis=0)
    if xyz.shape[0] > spacing_k:
        tree = cKDTree(xyz)
        distances, _ = tree.query(xyz, k=min(spacing_k + 1, xyz.shape[0]))
        spacing = np.mean(distances[:, 1:], axis=1)
    else:
        spacing = np.full(xyz.shape[0], 1e-3, dtype=np.float32)
    spacing = np.clip(spacing.astype(np.float32), 1e-6, None)

    scale_t = spacing
    scale_n = np.maximum(spacing * normal_scale_ratio, 1e-7)
    scales = np.stack([scale_t, scale_t, scale_n], axis=1)
    rotations = np.stack([normal_to_quaternion(r["normal"]) for r in rows], axis=0)

    track_times = np.zeros((len(rows), max_track_samples), dtype=np.float32)
    track_points = np.zeros((len(rows), max_track_samples, 3), dtype=np.float32)
    track_mask = np.zeros((len(rows), max_track_samples), dtype=np.float32)
    for i, row in enumerate(rows):
        samples = row["samples"]
        if len(samples) > max_track_samples:
            keep = np.linspace(0, len(samples) - 1, max_track_samples).round().astype(int)
            samples = [samples[j] for j in keep]
        for j, sample in enumerate(samples[:max_track_samples]):
            track_times[i, j] = sample["t"]
            track_points[i, j] = sample["point"]
            track_mask[i, j] = 1.0

    return {
        "track_ids": np.asarray([r["track_id"] for r in rows], dtype=np.int64),
        "mu_ref": xyz.astype(np.float32),
        "velocity": np.stack([r["velocity"] for r in rows], axis=0).astype(np.float32),
        "t_ref": np.asarray([r["t_ref"] for r in rows], dtype=np.float32),
        "colors": np.clip(np.stack([r["color"] for r in rows], axis=0), 0.0, 1.0).astype(np.float32),
        "scales": scales.astype(np.float32),
        "rotations": rotations.astype(np.float32),
        "confidence": np.asarray([r["confidence"] for r in rows], dtype=np.float32),
        "track_times": track_times,
        "track_points": track_points,
        "track_mask": track_mask,
    }


def project_cloud_to_depth(points: np.ndarray, cam: CameraInfo) -> torch.Tensor:
    """Nearest z-buffer projection of a reference sequence cloud."""
    device = cam.image.device
    pts = torch.as_tensor(points, dtype=torch.float32, device=device)
    ones = torch.ones((pts.shape[0], 1), dtype=torch.float32, device=device)
    pts_h = torch.cat([pts, ones], dim=1)
    cam_pts = (cam.w2c @ pts_h.T).T[:, :3]
    z = cam_pts[:, 2]
    u = cam.fx * (cam_pts[:, 0] / z.clamp(min=1e-8)) + cam.cx
    v = cam.fy * (cam_pts[:, 1] / z.clamp(min=1e-8)) + cam.cy
    ui = torch.round(u).long()
    vi = torch.round(v).long()
    valid = (z > 0.01) & (ui >= 0) & (ui < cam.width) & (vi >= 0) & (vi < cam.height)
    depth = torch.zeros((cam.height, cam.width), dtype=torch.float32, device=device)
    if not valid.any():
        return depth
    flat_idx = vi[valid] * cam.width + ui[valid]
    z_valid = z[valid]
    order = torch.argsort(z_valid, descending=True)
    flat = depth.reshape(-1)
    flat[flat_idx[order]] = z_valid[order]
    return depth
