# -*- coding: utf-8 -*-
"""Spyder-ready 3D Gaussian sequence viewer (PLY ellipsoids, not splat rendering)."""

from __future__ import annotations

import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from ply_utils import read_vertex_ply


# Local public-benchmark results; edit PUBLIC_PROJECT for another data location.
PUBLIC_PROJECT = THIS_DIR
if not (PUBLIC_PROJECT / "benchmark_results/cook_spinach_30f_surface_v1").is_dir():
    PUBLIC_PROJECT = THIS_DIR.parent / "paper/AI_image/4D_GS"
    if not PUBLIC_PROJECT.is_dir():
        PUBLIC_PROJECT = THIS_DIR.parent / "4D_GS"
PUBLIC_RESULTS = PUBLIC_PROJECT / "benchmark_results/cook_spinach_30f_surface_v1"
PUBLIC_EXPERIMENTS = {
    "Adaptive densification": PUBLIC_RESULTS / "dense_adaptive",
    "Surface-constrained": PUBLIC_RESULTS / "surface_adaptive",
}
DEFAULT_EXPERIMENT = "Surface-constrained"

# Edit these values, then run this file directly in Spyder.
IDE_CONFIG = SimpleNamespace(
    corrected_root=str(PUBLIC_EXPERIMENTS[DEFAULT_EXPERIMENT]),
    continue_frame_root=None,  # Optional original clouds; otherwise read train_config.json.
    mode="corrected",  # "corrected", "original", "both"
    color_mode="true",  # "true" (RGB), "confidence", "uncertainty"
    corrected_display="gaussians",  # "gaussians" or "points"
    fps=None,  # Only a fallback when model timestamps are unavailable; never rescales them.
    playback_speed=0.25,  # Slow down presentation; preserve the saved timestamps.
    loop=True,
    point_size=2.0,
    sigma_scale=1.0,  # Multiplier of Gaussian spatial scale, NOT an uncertainty interval.
    circle_segments=10,
    axis_size=0.0,
    min_opacity=0.0,  # Display-only filtering; source files are never changed.
    max_scale=1e9,
    confidence_range=None,  # None: fixed whole-sequence finite min/max, not per-frame percentiles.
    uncertainty_range=None,
    show_colorbar=True,
    width=1100,
    height=720,
    window_x=40,
    window_y=40,
    background=(0.03, 0.03, 0.035),
)


GAUSSIAN_SHAPE_FIELDS = {
    "x",
    "y",
    "z",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
}


def _normalize_scalar(values: np.ndarray, value_range=None):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values), 0.0, 1.0
    if value_range is None:
        lo, hi = float(values[finite].min()), float(values[finite].max())
    else:
        lo, hi = value_range
    norm = np.full(values.shape, 0.5, dtype=np.float64)
    if hi > lo:
        norm[finite] = np.clip((values[finite] - lo) / (hi - lo), 0.0, 1.0)
    norm[~finite] = 0.0
    return norm, float(lo), float(hi)


def _scalar_colormap(norm: np.ndarray, color_mode: str) -> np.ndarray:
    t = np.asarray(norm, dtype=np.float64).reshape(-1, 1)
    if color_mode == "confidence":
        stops = np.asarray(
            [
                [0.75, 0.12, 0.10],
                [0.95, 0.72, 0.18],
                [0.18, 0.78, 0.58],
                [0.12, 0.42, 0.95],
            ],
            dtype=np.float64,
        )
    else:
        stops = np.asarray(
            [
                [0.10, 0.28, 0.85],
                [0.10, 0.75, 0.88],
                [0.95, 0.78, 0.16],
                [0.82, 0.08, 0.08],
            ],
            dtype=np.float64,
        )
    x = np.clip(t * (len(stops) - 1), 0.0, len(stops) - 1)
    idx = np.floor(x).astype(np.int64).reshape(-1)
    frac = (x.reshape(-1) - idx)[:, None]
    idx_next = np.clip(idx + 1, 0, len(stops) - 1)
    return np.clip(stops[idx] * (1.0 - frac) + stops[idx_next] * frac, 0.0, 1.0)


def _scalar_values(vertex: dict, color_mode: str):
    fields = set(vertex)
    if color_mode == "uncertainty":
        if "opacity_variance" not in fields:
            raise RuntimeError("Corrected PLY has no opacity_variance field for uncertainty coloring.")
        values = np.asarray(vertex["opacity_variance"], dtype=np.float64)
    elif color_mode == "confidence":
        if "opacity_concentration" in fields:
            concentration = np.asarray(vertex["opacity_concentration"], dtype=np.float64)
            values = np.full(concentration.shape, np.nan)
            valid = np.isfinite(concentration) & (concentration > 0)
            values[valid] = np.log(concentration[valid])
        elif "confidence" in fields:
            values = np.asarray(vertex["confidence"], dtype=np.float64)
        else:
            raise RuntimeError("Corrected PLY has no opacity_concentration/confidence field for confidence coloring.")
    else:
        raise ValueError(f"Unsupported scalar color mode: {color_mode}")
    return values


def _scalar_colors(vertex: dict, keep: np.ndarray, color_mode: str, value_range=None):
    values = _scalar_values(vertex, color_mode)[keep]
    norm, _, _ = _normalize_scalar(values, value_range)
    colors = _scalar_colormap(norm, color_mode)
    colors[~np.isfinite(values)] = [0.5, 0.5, 0.5]
    return colors


def _read_cloud(path: str, uniform_color=None, force_uniform: bool = False):
    if Path(path).suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data["points"])
            if "colors" in data:
                pcd.colors = o3d.utility.Vector3dVector(data["colors"])
    else:
        pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise RuntimeError(f"Failed to load point cloud: {path}")
    if force_uniform or not pcd.has_colors():
        pcd.paint_uniform_color(uniform_color if uniform_color is not None else [0.8, 0.8, 0.8])
    return pcd


def _display_mask(vertex, min_opacity, max_scale):
    """Display exclusions only. Apply the same mask in every color/geometry mode."""
    fields = set(vertex)
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    keep = np.isfinite(points).all(axis=1)
    if "opacity" in fields:
        opacity = np.asarray(vertex["opacity"]).reshape(-1)
        keep &= np.isfinite(opacity) & (opacity >= min_opacity)
    if {"scale_0", "scale_1", "scale_2"}.issubset(fields):
        scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1).astype(np.float64)
        keep &= np.isfinite(scales).all(axis=1) & (scales > 0).all(axis=1)
        if max_scale > 0:
            keep &= scales.max(axis=1) <= max_scale
    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(fields):
        rotations = np.stack([vertex[f"rot_{i}"] for i in range(4)], axis=1)
        keep &= np.isfinite(rotations).all(axis=1) & (np.linalg.norm(rotations, axis=1) > 0)
    return keep


def _filter_corrected(path, min_opacity, max_scale, color_mode, vertex=None, value_range=None):
    if vertex is None:
        vertex = read_vertex_ply(path)
    keep = _display_mask(vertex, min_opacity, max_scale)
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[keep])
    if color_mode in ("confidence", "uncertainty"):
        colors = _scalar_colors(vertex, keep, color_mode, value_range)
    elif color_mode == "comparison" or not {"red", "green", "blue"}.issubset(vertex):
        colors = np.tile([0.1, 0.65, 1.0], (int(keep.sum()), 1))
    else:
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1)[keep] / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _quaternion_matrices(quaternions: np.ndarray) -> np.ndarray:
    """Convert normalized [w, x, y, z] quaternions to rotation matrices."""
    q = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.clip(norms, 1e-12, None)
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)], axis=1),
            np.stack([2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)], axis=1),
            np.stack([2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)], axis=1),
        ],
        axis=1,
    )


def _wireframe_template(circle_segments: int):
    """Build three unit circles that expose an ellipsoid's principal axes."""
    angles = np.linspace(0.0, 2.0 * np.pi, circle_segments, endpoint=False)
    cosines = np.cos(angles)
    sines = np.sin(angles)
    rings = [
        np.stack([cosines, sines, np.zeros_like(cosines)], axis=1),
        np.stack([cosines, np.zeros_like(cosines), sines], axis=1),
        np.stack([np.zeros_like(cosines), cosines, sines], axis=1),
    ]
    vertices = np.concatenate(rings, axis=0)
    ring_lines = np.stack([np.arange(circle_segments), np.roll(np.arange(circle_segments), -1)], axis=1)
    lines = np.concatenate([ring_lines + i * circle_segments for i in range(len(rings))], axis=0)
    return vertices, lines


def _gaussian_wireframes(
    path: str,
    min_opacity: float,
    max_scale: float,
    sigma_scale: float,
    circle_segments: int,
    color_mode: str,
    vertex=None,
    value_range=None,
):
    """Read corrected PLY Gaussian attributes and draw every active ellipsoid."""
    if vertex is None:
        vertex = read_vertex_ply(path)
    fields = set(vertex)
    missing = sorted(GAUSSIAN_SHAPE_FIELDS - fields)
    if missing:
        raise RuntimeError(
            "Corrected PLY has no full Gaussian shape data. "
            f"Missing: {', '.join(missing)}. "
            "Use the training output PLY files with scale_* and rot_* fields."
        )

    centers = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1).astype(np.float64)
    rotations = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1)
    keep = _display_mask(vertex, min_opacity, max_scale)
    if not keep.any():
        return o3d.geometry.LineSet()

    centers = centers[keep]
    scales = scales[keep] * sigma_scale
    rotations = rotations[keep]
    template_vertices, template_lines = _wireframe_template(max(4, circle_segments))
    local_vertices = template_vertices[None, :, :] * scales[:, None, :]
    matrices = _quaternion_matrices(rotations)
    vertices = np.einsum("nij,nvj->nvi", matrices, local_vertices) + centers[:, None, :]

    vertices_per_gaussian = template_vertices.shape[0]
    lines_per_gaussian = template_lines.shape[0]
    offsets = np.arange(centers.shape[0], dtype=np.int64)[:, None, None] * vertices_per_gaussian
    lines = (template_lines[None, :, :] + offsets).reshape(-1, 2)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices.reshape(-1, 3)),
        lines=o3d.utility.Vector2iVector(lines.astype(np.int32)),
    )

    if color_mode in ("confidence", "uncertainty"):
        colors = _scalar_colors(vertex, keep, color_mode, value_range)
        line_colors = np.repeat(colors, lines_per_gaussian, axis=0)
    elif color_mode == "comparison" or not {"red", "green", "blue"}.issubset(fields):
        line_colors = np.repeat(np.asarray([[0.1, 0.65, 1.0]]), centers.shape[0] * lines_per_gaussian, axis=0)
    else:
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float64)[keep] / 255.0
        line_colors = np.repeat(np.clip(colors, 0.0, 1.0), lines_per_gaussian, axis=0)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    return line_set


def _sequence_times(root, frames, config, fallback_fps):
    """Prefer saved reference times; fps is only used for older PLY-only exports."""
    model_path = root / "dynamic_gaussians_final.npz"
    time_by_frame = {}
    sequence_manifest = root / "sequence_manifest.json"
    if sequence_manifest.is_file():
        with open(sequence_manifest, encoding="utf-8") as f:
            records = json.load(f)["frames"]
        for row in records:
            frame = int(row["frame_idx"])
            if frame in time_by_frame:
                raise ValueError(f"Duplicate sequence timestamp for frame {frame}.")
            time_by_frame[frame] = float(row["timestamp"])
    if model_path.exists():
        with np.load(model_path, allow_pickle=False) as data:
            frame_indices = data["frame_indices"].reshape(-1)
            timestamps = data["t_ref"].reshape(-1)
        if frame_indices.shape != timestamps.shape:
            raise ValueError("Model frame_indices and t_ref have different lengths.")
        for frame in frames:
            values = timestamps[frame_indices == frame]
            if values.size and np.isfinite(values).all():
                if not np.allclose(values, values[0], rtol=0, atol=1e-6):
                    raise ValueError(f"Frame {frame} contains multiple reference times.")
                if frame in time_by_frame and not np.isclose(time_by_frame[frame], values[0], atol=1e-6):
                    raise ValueError(f"Sequence/model timestamp mismatch for frame {frame}.")
                time_by_frame[frame] = float(values[0])
    if len(time_by_frame) == len(frames):
        times = np.asarray([time_by_frame[frame] for frame in frames])
        source = "sequence/model timestamps"
    else:
        manifest_path = config.get("views_manifest")
        if manifest_path and Path(manifest_path).is_file():
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            for row in manifest.values():
                frame = int(row["frame_idx"])
                timestamp = float(row["timestamp"])
                if frame in time_by_frame and not np.isclose(time_by_frame[frame], timestamp, atol=1e-6):
                    raise ValueError(f"Inconsistent timestamps for frame {frame}.")
                time_by_frame[frame] = timestamp
        if all(frame in time_by_frame for frame in frames):
            times = np.asarray([time_by_frame[frame] for frame in frames])
            source = "view/model metadata"
        else:
            fps = fallback_fps if fallback_fps is not None else config.get("fps")
            if fps is None or not np.isfinite(fps) or fps <= 0:
                raise ValueError("No complete timestamps. Set IDE_CONFIG.fps for this PLY sequence.")
            times = np.asarray(frames, dtype=np.float64) / fps
            source = f"frame index / {fps:g} fps (fallback)"
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Frame timestamps must be finite and strictly increasing.")
    print(f"Timeline: {source}; {times[0]:.6f} to {times[-1]:.6f} s.")
    return times


def _playback_index(times, timestamp, loop):
    if len(times) == 1:
        return 0, True
    # Hold the last snapshot for the final sample interval. No interpolation.
    end = times[-1] + (times[-1] - times[-2])
    finished = timestamp >= end and not loop
    if timestamp >= end:
        if loop:
            timestamp = times[0] + (timestamp - times[0]) % (end - times[0])
        else:
            timestamp = times[-1]
    index = int(np.searchsorted(times, timestamp, side="right") - 1)
    return min(max(index, 0), len(times) - 1), finished


class SequenceViewer:
    def __init__(self, args, experiments=None):
        self.experiments = {} if experiments is None else dict(experiments)
        self.experiment_names = list(self.experiments)
        self.window = None
        self._load_sequence(args)
        self.experiment_index = None
        if self.experiments:
            roots = [Path(path).resolve() for path in self.experiments.values()]
            if self.corrected_root not in roots:
                raise ValueError("The initial result must be included in experiments.")
            self.experiment_index = roots.index(self.corrected_root)

    def _load_sequence(self, args):
        self.args = args
        self.corrected_root = Path(args.corrected_root).resolve()
        self.config = {}
        config_path = self.corrected_root / "train_config.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                self.config = json.load(f)
        self.paths = {}
        for prefix in ("corrected_frame_", "sequence_frame_"):
            for path in self.corrected_root.glob(f"{prefix}*.ply"):
                suffix = path.stem.removeprefix(prefix)
                if suffix.isdigit():
                    frame = int(suffix)
                    if frame in self.paths:
                        raise ValueError(f"Duplicate PLY frame {frame}: {path}")
                    self.paths[frame] = path
        self.frames = sorted(self.paths)
        if not self.frames:
            raise FileNotFoundError(f"No corrected_frame_*.ply or sequence_frame_*.ply files in {self.corrected_root}")
        self.times = _sequence_times(self.corrected_root, self.frames, self.config, args.fps)
        self.original_paths = {}
        original_root = args.continue_frame_root
        if original_root is None:
            original_root = self.config.get("sequence_cloud") or self.config.get("continue_frame_root")
        if original_root:
            root = Path(original_root)
            for frame in self.frames:
                candidates = [
                    root / f"frame_{frame:06d}.npz",
                    root / str(frame) / "robust_scene_dense.ply",
                    root / f"{frame:06d}" / "robust_scene_dense.ply",
                ]
                for path in candidates:
                    if path.exists():
                        self.original_paths[frame] = path
                        break
        self.has_originals = len(self.original_paths) == len(self.frames)
        self.mode = args.mode
        if self.mode not in ("corrected", "original", "both"):
            raise ValueError(f"Unknown scene mode: {self.mode}")
        if self.mode != "corrected" and not self.has_originals:
            raise ValueError("Original clouds are incomplete. Set mode='corrected' to view results alone.")
        self.color_mode = "true" if args.color_mode == "rgb" else args.color_mode
        self.display = args.corrected_display
        if self.display not in ("gaussians", "points"):
            raise ValueError("corrected_display must be 'gaussians' or 'points'.")
        if args.sigma_scale <= 0 or args.circle_segments < 4:
            raise ValueError("sigma_scale must be positive and circle_segments must be at least 4.")
        if not np.isfinite(args.playback_speed) or args.playback_speed <= 0:
            raise ValueError("playback_speed must be positive.")
        self.index = 0
        self.playing = False
        self.closed = False
        self.loop = args.loop
        self.speed = args.playback_speed
        self.play_started = 0.0
        self.play_time = float(self.times[0])
        self.cache = OrderedDict()
        self.scalar_ranges = {}
        self.counts = {}
        self._scan_sequence()

    def _vertex(self, frame):
        if frame not in self.cache:
            self.cache[frame] = read_vertex_ply(str(self.paths[frame]))
            if len(self.cache) > 3:
                self.cache.popitem(last=False)
        self.cache.move_to_end(frame)
        return self.cache[frame]

    def _scan_sequence(self):
        low = np.full(3, np.inf)
        high = np.full(3, -np.inf)
        ranges = {"confidence": [np.inf, -np.inf], "uncertainty": [np.inf, -np.inf]}
        available = {"true", "confidence", "uncertainty"}
        self.has_shapes = True
        invalid_scalars = 0
        for frame in self.frames:
            vertex = self._vertex(frame)
            keep = _display_mask(vertex, self.args.min_opacity, self.args.max_scale)
            self.counts[frame] = (int(keep.sum()), len(keep))
            self.has_shapes &= GAUSSIAN_SHAPE_FIELDS.issubset(vertex)
            if not {"red", "green", "blue"}.issubset(vertex):
                available.discard("true")
            for mode in ("confidence", "uncertainty"):
                if mode == "confidence" and not ({"confidence", "opacity_concentration"} & set(vertex)):
                    available.discard(mode)
                    continue
                if mode == "uncertainty" and "opacity_variance" not in vertex:
                    available.discard(mode)
                    continue
                values = _scalar_values(vertex, mode)[keep]
                finite = np.isfinite(values)
                invalid_scalars += int((~finite).sum())
                if finite.any():
                    ranges[mode][0] = min(ranges[mode][0], float(values[finite].min()))
                    ranges[mode][1] = max(ranges[mode][1], float(values[finite].max()))
            if keep.any():
                centers = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)[keep]
                radius = np.zeros((len(centers), 1))
                if GAUSSIAN_SHAPE_FIELDS.issubset(vertex):
                    scales = np.stack([vertex[f"scale_{i}"] for i in range(3)], axis=1)[keep]
                    radius = np.max(scales, axis=1, keepdims=True) * self.args.sigma_scale
                low = np.minimum(low, (centers - radius).min(axis=0))
                high = np.maximum(high, (centers + radius).max(axis=0))
        if not np.isfinite(low).all():
            raise ValueError("No finite Gaussians remain after the configured display filters.")
        self.bounds = o3d.geometry.AxisAlignedBoundingBox(low, high)
        for mode in ("confidence", "uncertainty"):
            if not np.isfinite(ranges[mode]).all():
                available.discard(mode)
            if mode in available:
                requested = self.args.confidence_range if mode == "confidence" else self.args.uncertainty_range
                values = ranges[mode] if requested is None else requested
                if len(values) != 2 or not np.isfinite(values).all() or values[1] < values[0]:
                    raise ValueError(f"Invalid {mode}_range: {values}")
                self.scalar_ranges[mode] = tuple(values)
                print(f"Fixed sequence {mode} color range: {tuple(values)}")
        self.color_modes = [mode for mode in ("true", "confidence", "uncertainty") if mode in available]
        if self.color_mode not in self.color_modes:
            raise ValueError(f"Color mode '{self.color_mode}' is unavailable in one or more PLY files.")
        if self.display == "gaussians" and not self.has_shapes:
            raise ValueError("Gaussian shape fields are missing. Set corrected_display='points'.")
        excluded = sum(total - shown for shown, total in self.counts.values())
        print(f"Loaded {len(self.frames)} 3D frames. Display filters exclude {excluded} entries across the sequence.")
        if invalid_scalars:
            print(f"{invalid_scalars} nonfinite/nonpositive scalar entries will be shown in gray.")
        print("RGB uses exported DC colors. Ellipsoid wireframes are not alpha-composited SH splats.")
        print("Uncertainty = Beta opacity variance, not XYZ error or a calibrated confidence interval.")

    def create_window(self):
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "4DGS Sequence Viewer", self.args.width, self.args.height,
            self.args.window_x, self.args.window_y,
        )
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([*self.args.background, 1.0])
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.window.add_child(self.scene)
        em = self.window.theme.font_size
        self.panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        if self.experiments:
            self.panel.add_child(gui.Label("Experiment"))
            self.experiment_box = gui.Combobox()
            for name in self.experiment_names:
                self.experiment_box.add_item(name)
            self.experiment_box.selected_index = self.experiment_index
            self.experiment_box.set_on_selection_changed(self._on_experiment)
            self.panel.add_child(self.experiment_box)
        self.panel.add_child(gui.Label("Scene"))
        self.mode_box = gui.Combobox()
        self.scene_modes = ["corrected", "original", "both"] if self.has_originals else ["corrected"]
        for mode in self.scene_modes:
            self.mode_box.add_item(mode.capitalize())
        self.mode_box.selected_index = self.scene_modes.index(self.mode)
        self.mode_box.set_on_selection_changed(self._on_mode)
        self.panel.add_child(self.mode_box)
        self.panel.add_child(gui.Label("Geometry"))
        self.display_box = gui.Combobox()
        self.display_box.add_item("Gaussian ellipsoids")
        self.display_box.add_item("Gaussian centers")
        self.display_box.selected_index = int(self.display == "points")
        self.display_box.set_on_selection_changed(self._on_display)
        self.display_box.enabled = self.has_shapes
        self.panel.add_child(self.display_box)
        self.panel.add_child(gui.Label("Color"))
        self.color_box = gui.Combobox()
        for mode in self.color_modes:
            self.color_box.add_item("RGB" if mode == "true" else mode.capitalize())
        self.color_box.selected_index = self.color_modes.index(self.color_mode)
        self.color_box.set_on_selection_changed(self._on_color)
        self.panel.add_child(self.color_box)
        self.legend_title = gui.Label("")
        self.legend_image = gui.ImageWidget()
        self.legend_values = gui.Label("")
        self.panel.add_child(self.legend_title)
        self.panel.add_child(self.legend_image)
        self.panel.add_child(self.legend_values)
        self.panel.add_child(gui.Label("Playback speed"))
        self.speed_box = gui.Combobox()
        self.speeds = sorted(set([0.25, 0.5, 1.0, 2.0, self.speed]))
        for speed in self.speeds:
            self.speed_box.add_item(f"{speed:g}x")
        self.speed_box.selected_index = self.speeds.index(self.speed)
        self.speed_box.set_on_selection_changed(self._on_speed)
        self.panel.add_child(self.speed_box)
        self.loop_box = gui.Checkbox("Loop")
        self.loop_box.checked = self.loop
        self.loop_box.set_on_checked(self._on_loop)
        self.panel.add_child(self.loop_box)
        self.axes_box = gui.Checkbox("Axes")
        self.axes_box.set_on_checked(self._on_axes)
        self.panel.add_child(self.axes_box)
        reset = gui.Button("Reset view")
        reset.set_on_clicked(self._reset_view)
        self.panel.add_child(reset)
        self.count_label = gui.Label("")
        self.panel.add_child(self.count_label)
        self.panel.add_stretch()
        self.window.add_child(self.panel)

        self.timeline = gui.Vert(0.3 * em, gui.Margins(em, 0, em, 0))
        self.time_label = gui.Label("")
        self.timeline.add_child(self.time_label)
        self.slider = gui.Slider(gui.Slider.INT)
        self.slider.set_limits(0, max(1, len(self.frames) - 1))
        self.slider.int_value = 0
        self.slider.enabled = len(self.frames) > 1
        self.slider.set_on_value_changed(self._on_seek)
        self.timeline.add_child(self.slider)
        controls = gui.Horiz(0.5 * em)
        self.previous = gui.Button("|<")
        self.previous.tooltip = "Previous frame (P / Left)"
        self.previous.set_on_clicked(self._previous)
        self.play_button = gui.Button(">")
        self.play_button.tooltip = "Play / Pause (Space)"
        self.play_button.set_on_clicked(self._toggle_play)
        self.next_button = gui.Button(">|")
        self.next_button.tooltip = "Next frame (N / Right)"
        self.next_button.set_on_clicked(self._next)
        self.play_button.enabled = len(self.frames) > 1
        self.previous.enabled = len(self.frames) > 1
        self.next_button.enabled = len(self.frames) > 1
        controls.add_stretch()
        controls.add_child(self.previous)
        controls.add_child(self.play_button)
        controls.add_child(self.next_button)
        controls.add_stretch()
        self.timeline.add_child(controls)
        self.window.add_child(self.timeline)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_tick_event(self._on_tick)
        self.window.set_on_close(self._on_close)
        self.scene.set_on_key(self._on_key)
        self._refresh()
        self._update_legend()
        self._reset_view()

    def _on_layout(self, context):
        rect = self.window.content_rect
        em = self.window.theme.font_size
        panel_width = min(int(18 * em), int(rect.width * 0.36))
        timeline_height = int(7 * em)
        scene_height = max(1, rect.height - timeline_height)
        self.scene.frame = gui.Rect(rect.x, rect.y, rect.width - panel_width, scene_height)
        self.panel.frame = gui.Rect(rect.get_right() - panel_width, rect.y, panel_width, scene_height)
        self.timeline.frame = gui.Rect(rect.x, rect.y + scene_height, rect.width, timeline_height)

    def _reset_view(self):
        self.scene.setup_camera(60, self.bounds, self.bounds.get_center())
        sfm_path = self.config.get("sfm_json")
        if sfm_path and Path(sfm_path).exists():
            with open(sfm_path, encoding="utf-8") as f:
                sfm = json.load(f)
            if sfm.get("extrinsics"):
                pose = sfm["extrinsics"][0]["value"]
                rotation = np.asarray(pose["rotation"])
                eye = np.asarray(pose["center"])
                # OpenMVG uses OpenCV camera axes: camera +Y is down, +Z is forward.
                distance = float(np.dot(self.bounds.get_center() - eye, rotation[2]))
                if distance > 0:
                    self.scene.look_at(eye + distance * rotation[2], eye, -rotation[1])
        self.scene.force_redraw()

    def _refresh(self):
        frame = self.frames[self.index]
        for name in ("corrected", "original"):
            if self.scene.scene.has_geometry(name):
                self.scene.scene.remove_geometry(name)
        if self.mode in ("corrected", "both"):
            vertex = self._vertex(frame)
            value_range = self.scalar_ranges.get(self.color_mode)
            if self.display == "gaussians":
                geometry = _gaussian_wireframes(
                    str(self.paths[frame]), self.args.min_opacity, self.args.max_scale,
                    self.args.sigma_scale, self.args.circle_segments, self.color_mode,
                    vertex=vertex, value_range=value_range,
                )
            else:
                geometry = _filter_corrected(
                    str(self.paths[frame]), self.args.min_opacity, self.args.max_scale,
                    self.color_mode, vertex=vertex, value_range=value_range,
                )
            if not geometry.is_empty():
                material = rendering.MaterialRecord()
                material.shader = "unlitLine" if self.display == "gaussians" else "defaultUnlit"
                material.line_width = 1.0
                material.point_size = self.args.point_size * self.window.scaling
                self.scene.scene.add_geometry("corrected", geometry, material, False)
        if self.mode in ("original", "both"):
            geometry = _read_cloud(str(self.original_paths[frame]), [0.72, 0.72, 0.72],
                                   force_uniform=self.color_mode != "true")
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            material.point_size = self.args.point_size * self.window.scaling
            self.scene.scene.add_geometry("original", geometry, material, False)
        self.slider.int_value = self.index
        shown, total = self.counts[frame]
        self.count_label.text = f"Gaussians: {shown:,} / {total:,}"
        self.time_label.text = (
            f"Frame {frame}  |  {self.index + 1} / {len(self.frames)}  |  "
            f"t = {self.times[self.index]:.6f} s"
        )
        self.scene.force_redraw()

    def _update_legend(self):
        visible = self.args.show_colorbar and self.color_mode in self.scalar_ranges and self.mode != "original"
        self.legend_title.visible = visible
        self.legend_image.visible = visible
        self.legend_values.visible = visible
        if visible:
            lo, hi = self.scalar_ranges[self.color_mode]
            self.legend_title.text = "Confidence: log(kappa)" if self.color_mode == "confidence" else "Opacity variance"
            norm = np.linspace(0, 1, 256) if hi > lo else np.full(256, 0.5)
            colors = (_scalar_colormap(norm, self.color_mode) * 255).astype(np.uint8)
            pixels = np.repeat(colors[None, :, :], 16, axis=0)
            self.legend_image.update_image(o3d.geometry.Image(pixels))
            self.legend_values.text = f"{lo:.5g}  ...  {hi:.5g}"
        self.window.set_needs_layout()

    def _on_color(self, text, index):
        self.color_mode = self.color_modes[index]
        self._refresh()
        self._update_legend()

    def _on_experiment(self, text, index):
        if index == self.experiment_index:
            return
        name = self.experiment_names[index]
        config = SimpleNamespace(**vars(self.args))
        config.corrected_root = str(self.experiments[name])
        config.color_mode = self.color_mode
        config.corrected_display = self.display
        config.mode = self.mode
        config.playback_speed = self.speed
        config.loop = self.loop
        self._load_sequence(config)
        self.experiment_index = index
        print(f"Experiment: {name} | {self.corrected_root}")
        if self.window is None:
            return
        self.experiment_box.selected_index = index
        self.mode_box.clear_items()
        self.scene_modes = ["corrected", "original", "both"] if self.has_originals else ["corrected"]
        for mode in self.scene_modes:
            self.mode_box.add_item(mode.capitalize())
        self.mode_box.selected_index = self.scene_modes.index(self.mode)
        self.color_box.clear_items()
        for mode in self.color_modes:
            self.color_box.add_item("RGB" if mode == "true" else mode.capitalize())
        self.color_box.selected_index = self.color_modes.index(self.color_mode)
        self.display_box.enabled = self.has_shapes
        self.slider.set_limits(0, max(1, len(self.frames) - 1))
        self.slider.enabled = len(self.frames) > 1
        self.play_button.enabled = len(self.frames) > 1
        self.previous.enabled = len(self.frames) > 1
        self.next_button.enabled = len(self.frames) > 1
        self.play_button.text = ">"
        self._refresh()
        self._update_legend()
        self._reset_view()
        if self.axes_box.checked:
            self._on_axes(True)

    def _on_display(self, text, index):
        self.display = "points" if index else "gaussians"
        self._refresh()

    def _on_mode(self, text, index):
        self.mode = self.scene_modes[index]
        self._refresh()
        self._update_legend()

    def _on_speed(self, text, index):
        self.speed = self.speeds[index]
        self.play_time = float(self.times[self.index])
        self.play_started = time.perf_counter()

    def _on_loop(self, checked):
        self.loop = checked

    def _on_axes(self, checked):
        if self.scene.scene.has_geometry("axes"):
            self.scene.scene.remove_geometry("axes")
        if checked:
            size = self.args.axis_size
            if size <= 0:
                size = max(float(np.linalg.norm(self.bounds.get_extent())) * 0.08, 1e-6)
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            self.scene.scene.add_geometry("axes", axes, material)
        self.scene.force_redraw()

    def _pause(self):
        self.playing = False
        self.play_button.text = ">"

    def _on_seek(self, value):
        self._pause()
        self.index = min(max(int(round(value)), 0), len(self.frames) - 1)
        self._refresh()

    def _previous(self):
        self._on_seek(max(0, self.index - 1))

    def _next(self):
        self._on_seek(min(len(self.frames) - 1, self.index + 1))

    def _toggle_play(self):
        if len(self.frames) < 2:
            return
        if self.playing:
            self._pause()
            return
        if self.index == len(self.frames) - 1:
            self.index = 0
            self._refresh()
        self.play_time = float(self.times[self.index])
        self.play_started = time.perf_counter()
        self.playing = True
        self.play_button.text = "||"

    def _on_tick(self):
        if self.closed or not self.playing:
            return False
        timestamp = self.play_time + (time.perf_counter() - self.play_started) * self.speed
        index, finished = _playback_index(self.times, timestamp, self.loop)
        if finished:
            self._pause()
        changed = index != self.index
        if changed:
            self.index = index
            self._refresh()
        return changed or not self.playing

    def _on_key(self, event):
        if event.type != gui.KeyEvent.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        if event.key == gui.KeyName.SPACE:
            self._toggle_play()
        elif event.key in (gui.KeyName.RIGHT, gui.KeyName.N):
            self._next()
        elif event.key in (gui.KeyName.LEFT, gui.KeyName.P):
            self._previous()
        elif event.key == gui.KeyName.R:
            self._reset_view()
        elif event.key in (gui.KeyName.ONE, gui.KeyName.TWO, gui.KeyName.THREE):
            index = int(event.key) - int(gui.KeyName.ONE)
            if index < len(self.color_modes):
                self.color_box.selected_index = index
                self._on_color("", index)
        else:
            return gui.Widget.EventCallbackResult.IGNORED
        return gui.Widget.EventCallbackResult.HANDLED

    def _on_close(self):
        self.playing = False
        self.closed = True
        return True

    def run(self):
        print("Mouse: drag to orbit, wheel to zoom, Ctrl-drag to pan.")
        print("Space: play/pause; N/P or arrows: next/previous; 1/2/3: color; R: reset.")
        self.create_window()
        gui.Application.instance.run()


def main(config=None, experiments=None):
    if config is None:
        config = IDE_CONFIG
        if experiments is None:
            public_roots = [path.resolve() for path in PUBLIC_EXPERIMENTS.values()]
            if Path(config.corrected_root).resolve() in public_roots:
                experiments = PUBLIC_EXPERIMENTS
    viewer = SequenceViewer(config, experiments=experiments)
    if viewer.experiments:
        print(f"Experiment: {viewer.experiment_names[viewer.experiment_index]} | {viewer.corrected_root}")
    viewer.run()


if __name__ == "__main__":
    main()
