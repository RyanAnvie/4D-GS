"""Mesh-reference geometry and held-out camera/time evaluation in meters."""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree
import torch

from calibration_eval import reliability_statistics
from gs_uncertainty.rendering import render_scalar_exposure_integrated
from losses import ssim_loss
from renderer import render_exposure_integrated
from model import save_ply_at_time


def evaluate_controlled(model, cameras, settings, truth_root, output, held_out_frames, held_out_cameras):
    output = Path(output)
    output.mkdir(parents=True)
    frame_rows = []
    image_rows = []
    concentration = model.get_opacity_concentration().detach().cpu().numpy().reshape(-1)
    for frame_idx in sorted({cam.frame_idx for cam in cameras}):
        frame_cameras = [cam for cam in cameras if cam.frame_idx == frame_idx]
        left = next(cam for cam in frame_cameras if cam.camera_id == 0)
        timestamp = left.timestamp
        with np.load(Path(truth_root) / f"frame_{frame_idx:06d}.npz") as data:
            vertices = data["vertices"]
            triangles = data["triangles"]
            reference_depth = data["depth_m"]
            visible = data["visible"]
        scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(triangles))
        with torch.no_grad():
            indices = model.active_indices_at(timestamp, settings["temporal_window"])
            points = model.position_at(timestamp)[indices].cpu().numpy()
        if not np.isfinite(points).all() or len(points) == 0:
            raise ValueError("Invalid active geometry; no silent filtering.")
        distances = scene.compute_distance(o3d.core.Tensor(points), nthreads=1).numpy()
        rows, columns = np.indices(reference_depth[0].shape)
        z = reference_depth[0][visible[0]]
        x = (columns[visible[0]] - left.cx) * z / left.fx
        y = (rows[visible[0]] - left.cy) * z / left.fy
        reference = np.column_stack((x, y, z))
        completeness = cKDTree(points).query(reference)[0]
        confidence = np.log(concentration[indices.cpu().numpy()])
        stats = reliability_statistics(confidence, distances)
        np.savez_compressed(output / f"geometry_{frame_idx:06d}.npz", indices=indices.cpu().numpy(),
                            points=points, mesh_distance_m=distances, log_kappa=confidence)
        save_ply_at_time(model, str(output / f"sequence_frame_{frame_idx:06d}.ply"), timestamp,
                         temporal_window=settings["temporal_window"])
        frame_rows.append({"frame": frame_idx, "timestamp_s": timestamp,
                           "held_out_time": frame_idx in held_out_frames,
                           "active_gaussians": len(points), "unique_centers": len(np.unique(points, axis=0)),
                           "mesh_distance_mean_m": float(distances.mean()),
                           "mesh_distance_p95_m": float(np.quantile(distances, 0.95)),
                           "visible_reference_completeness_mean_m": float(completeness.mean()),
                           "confidence_reliability": stats})
        with torch.no_grad():
            for cam in frame_cameras:
                raw = render_exposure_integrated(model, cam, use_exposure=True, **settings)["image"]
                if not torch.isfinite(raw).all():
                    raise ValueError("Nonfinite dynamic rendering.")
                prediction = raw.clamp(0., 1.)
                mse = float((prediction - cam.image).square().mean())
                ones = torch.ones_like(model.get_opacity_mean())
                alpha = render_scalar_exposure_integrated(model, cam, ones,
                            use_exposure=True, **settings)["map"].squeeze().cpu().numpy()
                held_time = frame_idx in held_out_frames
                held_camera = cam.camera_id in held_out_cameras
                group = ("unseen_time" if held_time else "seen_time") + ("_novel_view" if held_camera else "_train_view")
                image_rows.append({"frame": frame_idx, "camera": cam.camera_id, "group": group,
                                   "psnr": float(-10 * np.log10(max(mse, 1e-12))),
                                   "ssim": float(1 - ssim_loss(prediction, cam.image)),
                                   "mae": float((prediction - cam.image).abs().mean()),
                                   "alpha_ge_0_5_fraction": float(np.mean(alpha >= 0.5)),
                                   "rgb_clipped_fraction": float(((raw < 0) | (raw > 1)).float().mean())})
                rgb = prediction.permute(1, 2, 0).cpu().numpy()
                Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(output / f"frame_{frame_idx:06d}_cam{cam.camera_id}.png")
        print(f"Geometry t={timestamp:.3f}: {len(points)} active, mesh mean={distances.mean():.6f} m", flush=True)
    summary = {"units": "meters", "geometry_reference": "Independent triangle mesh, not initialization cloud",
               "image_domain": "All image pixels, black outside designed surface; no foreground masking",
               "frame_metrics": frame_rows, "image_metrics": image_rows, "image_groups": {}, "geometry_groups": {}}
    for group in sorted({row["group"] for row in image_rows}):
        selected = [row for row in image_rows if row["group"] == group]
        summary["image_groups"][group] = {"images": len(selected)}
        for field in ("psnr", "ssim", "mae", "alpha_ge_0_5_fraction"):
            summary["image_groups"][group][field] = float(np.mean([row[field] for row in selected]))
    for held, name in ((False, "seen_time"), (True, "unseen_time")):
        selected = [row for row in frame_rows if row["held_out_time"] == held]
        summary["geometry_groups"][name] = {}
        for field in ("active_gaussians", "mesh_distance_mean_m", "mesh_distance_p95_m", "visible_reference_completeness_mean_m"):
            summary["geometry_groups"][name][field] = float(np.mean([row[field] for row in selected]))
    with open(output / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    sequence_manifest = {"units": "meters", "frames": []}
    for row in frame_rows:
        sequence_manifest["frames"].append({"frame_idx": row["frame"], "timestamp": row["timestamp_s"],
            "file": f"sequence_frame_{row['frame']:06d}.ply", "held_out_time": row["held_out_time"]})
    with open(output / "sequence_manifest.json", "w", encoding="utf-8") as f:
        json.dump(sequence_manifest, f, indent=2)
    return summary
