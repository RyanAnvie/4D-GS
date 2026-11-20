"""Frozen RGB coverage and alpha-weighted center-depth diagnosis, in meters."""

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import open3d as o3d
import torch

from calibration_eval import load_saved_model
from data_io import load_openmvg_sequence
from gs_uncertainty.rendering import render_scalar_exposure_integrated
from renderer import render_exposure_integrated


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "benchmark_data/controlled_multibaseline_sequence_dev_v1"
AUDIT = ROOT / "benchmark_results/controlled_projection_moments_audit_v1/summary.json"
OUTPUT = ROOT / "benchmark_results/controlled_spatial_diagnosis_dev_v1"
ALPHA_EDGES = np.array([0., .25, .5, .75, 1.], dtype=float)
VARIANTS = {
    "adaptive": "controlled_matched_adaptive_dev_v1",
    "frozen": "controlled_tangent_prior_ablation_dev_v1",
    "tangent": "controlled_tangent_prior_ablation_dev_v1",
    "projection_only": "controlled_projection_only_dev_v1",
    "retain_moments": "controlled_projection_moments_dev_v1",
}


def render_center_depth(model, cam, settings):
    """Return sum(w*camera-Z)/sum(w), with NaN at exactly zero rendered alpha."""
    points = model.position_at(cam.timestamp)
    rotation = cam.w2c[:3, :3]
    translation = cam.w2c[:3, 3].reshape(1, 3)
    camera_points = points @ rotation.T + translation
    numerator = render_scalar_exposure_integrated(model, cam, camera_points[:, 2],
        use_exposure=False, **settings)["map"].squeeze()
    alpha = render_scalar_exposure_integrated(model, cam, torch.ones_like(camera_points[:, 2]),
        use_exposure=False, **settings)["map"].squeeze()
    if not torch.isfinite(numerator).all() or not torch.isfinite(alpha).all():
        raise ValueError("Nonfinite scalar render; no pixel filtering allowed.")
    if torch.any(alpha < 0) or torch.any(alpha > 1):
        raise ValueError("Rendered alpha outside [0,1].")
    depth = torch.full_like(alpha, float("nan"))
    supported = alpha > 0
    depth[supported] = numerator[supported] / alpha[supported]
    if not torch.isfinite(depth[supported]).all():
        raise ValueError("Invalid depth on supported pixels.")
    return {"depth_m": depth, "alpha": alpha, "numerator_m": numerator}


def mesh_camera_depth(vertices, triangles, cam):
    """Ray parameter equals camera-Z because the camera ray's Z component is 1."""
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(triangles))
    rotation = cam.w2c[:3, :3].cpu().numpy()
    translation = cam.w2c[:3, 3].cpu().numpy()
    center = cam.camera_center.cpu().numpy()
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0)
    np.testing.assert_allclose(rotation @ center + translation, 0., atol=1e-6, rtol=0)
    rows, columns = np.indices((cam.height, cam.width), dtype=np.float32)
    directions = np.stack(((columns-cam.cx)/cam.fx, (rows-cam.cy)/cam.fy, np.ones_like(rows)), axis=-1)
    rays = np.empty((cam.height, cam.width, 6), dtype=np.float32)
    rays[..., :3] = center
    rays[..., 3:] = directions @ rotation
    return scene.cast_rays(o3d.core.Tensor(rays), nthreads=1)["t_hit"].numpy()


def image_diagnostics(residual, alpha, depth, center_alpha, reference_depth, visible):
    squared = np.mean(residual.astype(float)**2, axis=0)
    assert squared.shape == alpha.shape == depth.shape == reference_depth.shape == visible.shape
    assert np.isfinite(squared).all() and np.isfinite(alpha).all()
    assert np.all((alpha >= 0) & (alpha <= 1))
    assert np.array_equal(np.isfinite(reference_depth), visible)
    assert np.array_equal(np.isfinite(depth), center_alpha > 0)
    squared_total = float(squared.sum())
    regions = []
    for region, mask in (("visible_at_center", visible), ("outside_at_center", ~visible)):
        for index in range(len(ALPHA_EDGES)-1):
            selected = mask & (alpha >= ALPHA_EDGES[index])
            if index < len(ALPHA_EDGES)-2:
                selected &= alpha < ALPHA_EDGES[index+1]
            else:
                selected &= alpha <= ALPHA_EDGES[index+1]
            count = int(selected.sum())
            error_sum = float(squared[selected].sum())
            regions.append({"region": region, "alpha_low": float(ALPHA_EDGES[index]),
                "alpha_high": float(ALPHA_EDGES[index+1]), "pixels": count,
                "squared_rgb_error_sum": error_sum,
                "contribution_to_full_mse": error_sum/squared.size,
                "conditional_mse": error_sum/count if count else None})
    assert sum(row["pixels"] for row in regions) == squared.size
    np.testing.assert_allclose(sum(row["squared_rgb_error_sum"] for row in regions), squared_total, rtol=1e-12)
    supported = visible & (center_alpha > 0)
    signed = depth[supported].astype(float)-reference_depth[supported].astype(float)
    absolute = np.abs(signed)
    depth_regions = []
    for index in range(len(ALPHA_EDGES)-1):
        selected = supported & (center_alpha >= ALPHA_EDGES[index])
        if index < len(ALPHA_EDGES)-2:
            selected &= center_alpha < ALPHA_EDGES[index+1]
        else:
            selected &= center_alpha <= ALPHA_EDGES[index+1]
        error = depth[selected].astype(float)-reference_depth[selected].astype(float)
        depth_regions.append({"alpha_low": float(ALPHA_EDGES[index]),
            "alpha_high": float(ALPHA_EDGES[index+1]), "pixels": len(error),
            "signed_error_sum_m": float(error.sum()), "absolute_error_sum_m": float(np.abs(error).sum()),
            "absolute_median_m": float(np.median(np.abs(error))) if len(error) else None})
    assert sum(row["pixels"] for row in depth_regions) == len(signed)
    return {"pixels": squared.size, "mse": squared_total/squared.size,
        "rgb_regions": regions, "depth_regions": depth_regions, "reference_visible_pixels": int(visible.sum()),
        "reference_visible_without_depth": int((visible & (center_alpha == 0)).sum()),
        "rendered_without_reference": int((~visible & (center_alpha > 0)).sum()),
        "center_alpha_visible_sum": float(center_alpha[visible].astype(float).sum()),
        "exposure_alpha_visible_sum": float(alpha[visible].astype(float).sum()),
        "depth_supported_pixels": len(signed),
        "depth_signed_error_sum_m": float(signed.sum()),
        "depth_absolute_error_sum_m": float(absolute.sum()),
        "depth_absolute_median_m": float(np.median(absolute)) if len(signed) else None,
        "depth_absolute_p95_m": float(np.quantile(absolute, .95)) if len(signed) else None}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with open(AUDIT, encoding="utf-8") as file:
        audit = json.load(file)
    assert audit["source_input_split_budget_lineage_density_checks"]
    with open(ROOT / "benchmark_results/controlled_projection_moments_dev_v1/protocol.json", encoding="utf-8") as file:
        original = json.load(file)
    hashes = {str(AUDIT.relative_to(ROOT)): hashlib.sha256(AUDIT.read_bytes()).hexdigest()}
    sources = [Path(__file__).name, "tests/test_spatial_rendering.py", "model.py", "renderer.py", "data_io.py",
        "calibration_eval.py", "ply_utils.py", "EXPERIMENT_PLAN.md"]
    sources.extend(str(path.relative_to(ROOT)) for path in sorted((ROOT / "gs_uncertainty").glob("*.py")))
    for name in sources:
        digest = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        # Historical protocols recorded tests in the project root.
        original_name = name.removeprefix("tests/")
        if original_name in original["source_sha256"] and name != "EXPERIMENT_PLAN.md":
            assert digest == original["source_sha256"][original_name], name
        hashes[name] = digest
    for name, digest in original["data_sha256"].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == digest
        hashes[str((DATA/name).relative_to(ROOT))] = digest
    OUTPUT.mkdir(parents=True)
    for name in sources:
        destination = OUTPUT / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    with open(OUTPUT / "protocol.json", "w", encoding="utf-8") as file:
        json.dump({"phase": "development", "changes": "none; frozen-model diagnosis",
            "alpha_bins": ALPHA_EDGES.tolist(), "source_and_input_sha256": hashes}, file, indent=2)
    summaries, images = [], []
    for kind in ("translation", "deformation"):
        reference = {}
        for variant, folder in VARIANTS.items():
            run = ROOT / "benchmark_results" / folder / kind / variant
            destination = OUTPUT / kind / variant
            destination.mkdir(parents=True)
            for name in ("dynamic_gaussians_final.npz", "train_config.json", "render_settings.json", "independent_evaluation/summary.json"):
                path = run/name
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            model_name = str((run/"dynamic_gaussians_final.npz").relative_to(ROOT))
            assert hashes[model_name] == audit["model_sha256"][model_name]
            with open(run/"train_config.json", encoding="utf-8") as file:
                config = json.load(file)
            with open(run/"render_settings.json", encoding="utf-8") as file:
                settings = json.load(file)
            with open(run/"independent_evaluation/summary.json", encoding="utf-8") as file:
                previous = json.load(file)
            cameras = load_openmvg_sequence(config["sfm_json"], views_manifest=config["views_manifest"],
                target_width=config["target_width"], device="cuda")
            assert len(cameras) == 42
            for path in [Path(config["sfm_json"]), Path(config["views_manifest"])] + [Path(cam.image_path) for cam in cameras]:
                assert str(path.relative_to(ROOT)) in hashes, path
            model = load_saved_model(run/"dynamic_gaussians_final.npz")
            saved = {name: value.detach().clone() for name, value in model.named_parameters()}
            records = []
            for cam, old in zip(cameras, previous["image_metrics"]):
                key = (cam.frame_idx, cam.camera_id)
                assert key == (old["frame"], old["camera"])
                if key not in reference:
                    path = DATA/kind/"evaluation_only"/f"frame_{cam.frame_idx:06d}.npz"
                    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                    with np.load(path) as data:
                        depth = data["depth_m"][cam.camera_id].copy()
                        visible = data["visible"][cam.camera_id].copy()
                        cast = mesh_camera_depth(data["vertices"], data["triangles"], cam)
                    np.testing.assert_array_equal(np.isfinite(cast), visible)
                    np.testing.assert_allclose(cast[visible], depth[visible], rtol=0, atol=2e-6)
                    reference[key] = (depth, visible)
                reference_depth, visible = reference[key]
                with torch.no_grad():
                    raw = render_exposure_integrated(model, cam, use_exposure=True, **settings)["image"]
                    assert torch.isfinite(raw).all()
                    residual = (raw.clamp(0., 1.)-cam.image).cpu().numpy()
                    alpha = render_scalar_exposure_integrated(model, cam, torch.ones_like(model.get_opacity_mean()),
                        use_exposure=True, **settings)["map"].squeeze().cpu().numpy()
                    center = render_center_depth(model, cam, settings)
                depth = center["depth_m"].cpu().numpy()
                center_alpha = center["alpha"].cpu().numpy()
                row = image_diagnostics(residual, alpha, depth, center_alpha, reference_depth, visible)
                row.update({"kind": kind, "variant": variant, "frame": cam.frame_idx,
                    "camera": cam.camera_id, "group": old["group"], "timestamp_s": cam.timestamp})
                psnr = -10*np.log10(max(row["mse"], 1e-12))
                np.testing.assert_allclose(psnr, old["psnr"], rtol=0, atol=1e-5)
                depth_error = np.full(depth.shape, np.nan, dtype=np.float32)
                comparable = visible & (center_alpha > 0)
                depth_error[comparable] = depth[comparable]-reference_depth[comparable]
                np.savez_compressed(destination/f"frame_{cam.frame_idx:06d}_cam{cam.camera_id}.npz",
                    rgb_residual=residual, exposure_alpha=alpha, center_alpha=center_alpha,
                    center_depth_m=depth, center_depth_numerator_m=center["numerator_m"].cpu().numpy(),
                    reference_depth_m=reference_depth, reference_visible=visible,
                    center_depth_signed_error_m=depth_error, center_depth_absolute_error_m=np.abs(depth_error))
                records.append(row)
            for name, value in model.named_parameters():
                torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
            for group in ("all", "held_out", "seen_time_train_view", "seen_time_novel_view", "unseen_time_train_view", "unseen_time_novel_view"):
                selected = []
                for row in records:
                    if group == "all" or row["group"] == group or (group == "held_out" and row["group"] != "seen_time_train_view"):
                        selected.append(row)
                totals = {}
                for field in ("pixels", "reference_visible_pixels", "reference_visible_without_depth", "rendered_without_reference",
                    "center_alpha_visible_sum", "exposure_alpha_visible_sum", "depth_supported_pixels",
                    "depth_signed_error_sum_m", "depth_absolute_error_sum_m"):
                    totals[field] = sum(row[field] for row in selected)
                count = totals["depth_supported_pixels"]
                totals["depth_mae_m"] = totals["depth_absolute_error_sum_m"]/count if count else None
                totals["depth_bias_m"] = totals["depth_signed_error_sum_m"]/count if count else None
                totals["rgb_regions"] = []
                for index in range(8):
                    parts = [row["rgb_regions"][index] for row in selected]
                    error_sum = sum(part["squared_rgb_error_sum"] for part in parts)
                    pixel_count = sum(part["pixels"] for part in parts)
                    totals["rgb_regions"].append({"region": parts[0]["region"], "alpha_low": parts[0]["alpha_low"],
                        "alpha_high": parts[0]["alpha_high"], "pixels": pixel_count,
                        "squared_rgb_error_sum": error_sum, "contribution_to_full_mse": error_sum/totals["pixels"],
                        "conditional_mse": error_sum/pixel_count if pixel_count else None})
                totals["depth_regions"] = []
                for index in range(4):
                    parts = [row["depth_regions"][index] for row in selected]
                    pixel_count = sum(part["pixels"] for part in parts)
                    absolute_sum = sum(part["absolute_error_sum_m"] for part in parts)
                    signed_sum = sum(part["signed_error_sum_m"] for part in parts)
                    totals["depth_regions"].append({"alpha_low": parts[0]["alpha_low"],
                        "alpha_high": parts[0]["alpha_high"], "pixels": pixel_count,
                        "absolute_error_sum_m": absolute_sum, "signed_error_sum_m": signed_sum,
                        "mae_m": absolute_sum/pixel_count if pixel_count else None})
                summaries.append({"kind": kind, "variant": variant, "group": group, "images": len(selected), **totals})
            images.extend(records)
            print(f"PASS {kind}/{variant}:42 images, replayed PSNR, mesh camera-Z, immutable model", flush=True)
            del model, saved, cameras
            torch.cuda.empty_cache()
    for name, digest in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    with open(OUTPUT/"summary.json", "w", encoding="utf-8") as file:
        json.dump({"scope": "Frozen DEV diagnosis; alpha-weighted center depth is not a surface or uncertainty",
            "source_and_input_sha256": hashes, "summary": summaries, "images": images}, file, indent=2, allow_nan=False)
    lines = ["# Frozen Spatial Diagnosis", "", "Ten unchanged models,420 images, DEV1201 only; no training or new acceptance criterion.",
        "Camera-Z is sum(w_i*Z_i)/sum(w_i) at exposure center, not first-hit mesh depth or coordinate uncertainty.",
        "RGB/alpha use the original exposure integration. Mesh visibility is at center time, not shutter integrated.",
        "Zero-support depth remains NaN in maps and is counted below; all positive-alpha reference pixels enter depth MAE.",
        "All model/data/source hashes, replayed full-image PSNR, mesh rays and frozen parameters pass.", "",
        "| Case | Model | Held depth MAE (mm) | Bias (mm) | No depth / reference pixels | Mean visible alpha | Visible alpha<.75 MSE contribution |",
        "|---|---|---:|---:|---|---:|---:|"]
    for row in summaries:
        if row["group"] != "held_out":
            continue
        contribution = sum(part["contribution_to_full_mse"] for part in row["rgb_regions"] if part["region"] == "visible_at_center" and part["alpha_high"] <= .75)
        visible_alpha = row["exposure_alpha_visible_sum"]/row["reference_visible_pixels"]
        mae = "undefined" if row["depth_mae_m"] is None else f"{1000*row['depth_mae_m']:.4f}"
        bias = "undefined" if row["depth_bias_m"] is None else f"{1000*row['depth_bias_m']:.4f}"
        lines.append(f"| {row['kind']} | {row['variant']} | {mae} | {bias} | {row['reference_visible_without_depth']}/{row['reference_visible_pixels']} | {visible_alpha:.6f} | {contribution:.8f} |")
    lines += ["", "Full counts, conditional errors and additive full-image MSE contributions are retained for every region and image.",
        "Alpha-bin changes combine pixels changing bins and changing errors; they are not a causal decomposition.",
        "Per-image NPZ maps preserve all pixels, invalid reference infinity and zero-support depth NaN.",
        "No model promotion, three-seed validation, XYZ uncertainty calibration or real granular claim follows from this diagnosis.", ""]
    (OUTPUT/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summaries


if __name__ == "__main__":
    main()
