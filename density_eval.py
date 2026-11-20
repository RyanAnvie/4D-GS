"""3D density/coverage diagnostics against training clouds, NOT ground-truth accuracy."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "benchmark_results/cook_spinach_30f_surface_v1/surface_adaptive"
REFERENCE_CLOUDS = ROOT / "benchmark_data/cook_spinach_30f_50k/clouds"


def evaluate_density(run_dir, reference_clouds, output_dir=None):
    run_dir = Path(run_dir)
    with np.load(run_dir / "dynamic_gaussians_final.npz", allow_pickle=False) as data:
        points = data["mu_ref"]
        frames = data["frame_indices"]
        opacity = data["opacity"].reshape(-1)
        scales = data["scales"]
        generation = data["generation"] if "generation" in data else np.zeros(len(points), dtype=np.int64)
        source_indices = data["source_indices"] if "source_indices" in data else None
    if not all(np.isfinite(array).all() for array in (points, opacity, scales)):
        raise ValueError("Nonfinite saved geometry; no filtering is allowed in density evaluation.")
    rows = []
    support = None
    if (run_dir / "surface_support.npz").exists():
        with np.load(run_dir / "surface_support.npz", allow_pickle=False) as data:
            support = dict(data)
        with open(run_dir / "train_config.json", encoding="utf-8") as f:
            config = json.load(f)
        if source_indices is None or source_indices.min() < 0 or source_indices.max() >= len(support["points"]):
            raise ValueError("Invalid saved surface source mapping.")
        if not np.array_equal(support["frame_indices"][source_indices], frames):
            raise ValueError("Surface support crossed frame boundaries.")
    for frame in np.unique(frames):
        mask = frames == frame
        xyz = points[mask]
        with np.load(Path(reference_clouds) / f"frame_{int(frame):06d}.npz", allow_pickle=False) as data:
            reference = data["points"]
            normals = data["normals"]
        if not np.isfinite(reference).all() or not np.isfinite(normals).all():
            raise ValueError("Nonfinite reference cloud/normals.")
        normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        if (normal_lengths <= 0).any() or len(reference) < 2:
            raise ValueError("Reference geometry requires valid normals and at least two points.")
        normals = normals / normal_lengths
        reference_tree = cKDTree(reference)
        neighbors = reference_tree.query(reference, k=min(7, len(reference)))[0]
        reference_spacing = neighbors[:, 1:].mean(axis=1)
        if (reference_spacing <= 0).any():
            raise ValueError("Zero reference spacing; normalized geometry metrics are undefined.")
        tree = cKDTree(xyz)
        spacing = tree.query(xyz, k=2)[0][:, 1] if len(xyz) > 1 else np.zeros(1)
        to_reference, nearest = reference_tree.query(xyz, k=1)
        from_reference = tree.query(reference, k=1)[0]
        residual = xyz.astype(np.float64) - reference[nearest]
        plane_error = np.abs(np.sum(residual * normals[nearest], axis=1))
        relative_distance = to_reference / reference_spacing[nearest]
        row = {"frame": int(frame), "gaussians": len(xyz),
                     "unique_centers": len(np.unique(xyz, axis=0)),
                     "opacity_ge_0_01": int((opacity[mask] >= 0.01).sum()),
                     "split_descendants": int((generation[mask] > 0).sum()),
                     "median_nn_spacing": float(np.median(spacing)),
                     "p90_nn_spacing": float(np.quantile(spacing, 0.9)),
                     "median_max_scale": float(np.median(scales[mask].max(axis=1))),
                     "model_to_training_cloud_mean": float(to_reference.mean()),
                     "model_to_training_cloud_p95": float(np.quantile(to_reference, 0.95)),
                     "model_to_training_cloud_p99": float(np.quantile(to_reference, 0.99)),
                     "model_to_training_cloud_max": float(to_reference.max()),
                     "nearest_plane_mean": float(plane_error.mean()),
                     "nearest_plane_p95": float(np.quantile(plane_error, 0.95)),
                     "distance_over_spacing_p95": float(np.quantile(relative_distance, 0.95)),
                     "fraction_farther_than_one_spacing": float(np.mean(relative_distance > 1.0)),
                     "training_cloud_to_model_mean": float(from_reference.mean()),
                     "training_cloud_to_model_p95": float(np.quantile(from_reference, 0.95)),
                     "coverage_within_half_spacing": float(np.mean(from_reference <= 0.5 * reference_spacing)),
                     "symmetric_mean_distance": float((to_reference.mean() + from_reference.mean()) / 2)}
        if support is not None:
            source = source_indices[mask]
            origin = support["points"][source].astype(np.float64)
            normal = support["normals"][source].astype(np.float64)
            local_spacing = support["spacing"][source].reshape(-1)
            displacement = xyz.astype(np.float64) - origin
            height = np.sum(displacement * normal, axis=1)
            tangent = displacement - height[:, None] * normal
            tangent_length = np.linalg.norm(tangent, axis=1)
            normal_excess = np.abs(height) - config["surface_normal_band"] * local_spacing
            tangent_excess = tangent_length - config["surface_tangent_radius"] * local_spacing
            # Absolute float32 coordinate rounding, not a geometric acceptance threshold.
            tolerance = 8 * np.finfo(np.float32).eps * np.maximum(np.abs(origin).max(axis=1), 1.0)
            violations = (normal_excess > tolerance) | (tangent_excess > tolerance)
            row.update({"support_violations": int(violations.sum()),
                        "support_normal_ratio_max": float((np.abs(height) / local_spacing).max()),
                        "support_tangent_ratio_max": float((tangent_length / local_spacing).max()),
                        "support_excess_max": float(max(0, normal_excess.max(), tangent_excess.max()))})
        rows.append(row)
    summary = {"reference": str(Path(reference_clouds).resolve()),
               "interpretation": "Density and training-cloud coverage diagnostics, not independent geometry accuracy.",
               "units": "Released SfM coordinates; physical scale unknown.",
               "opacity_threshold": 0.01,
               "spacing_definition": "Mean distance to up to six other training points; no physical unit conversion.",
               "coverage_threshold_spacing": 0.5, "far_threshold_spacing": 1.0,
               "total_gaussians": len(points), "frame_count": len(rows),
               "min_gaussians_per_frame": min(row["gaussians"] for row in rows),
               "max_gaussians_per_frame": max(row["gaussians"] for row in rows),
               "total_unique_centers_per_frame": sum(row["unique_centers"] for row in rows),
               "total_opacity_ge_0_01": sum(row["opacity_ge_0_01"] for row in rows),
               "total_split_descendants": sum(row["split_descendants"] for row in rows)}
    for field in ("median_nn_spacing", "median_max_scale", "model_to_training_cloud_mean",
                  "model_to_training_cloud_p95", "model_to_training_cloud_p99", "nearest_plane_mean",
                  "nearest_plane_p95", "distance_over_spacing_p95", "fraction_farther_than_one_spacing",
                  "training_cloud_to_model_mean", "training_cloud_to_model_p95",
                  "coverage_within_half_spacing", "symmetric_mean_distance"):
        summary["mean_frame_" + field] = float(np.mean([row[field] for row in rows]))
    summary["worst_model_to_training_cloud_distance"] = max(row["model_to_training_cloud_max"] for row in rows)
    if support is not None:
        summary["total_support_violations"] = sum(row["support_violations"] for row in rows)
        summary["max_support_excess_before_rounding_tolerance"] = max(row["support_excess_max"] for row in rows)
    output = run_dir / "density_evaluation" if output_dir is None else Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "per_frame_density.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(output / "density_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    return summary


if __name__ == "__main__":
    print(json.dumps(evaluate_density(RUN_DIR, REFERENCE_CLOUDS), indent=2))
