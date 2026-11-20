"""Evaluate saved public-benchmark models. Run directly in Spyder."""

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
import torch

from data_io import load_openmvg_sequence
from losses import ssim_loss
from model import DynamicGaussianModel
from renderer import render_exposure_integrated
from gs_uncertainty.rendering import render_scalar_exposure_integrated


RUN_DIR = ROOT / "examples/results/controlled_demo/translation/sensitivity/fixed_on"
RESERVED_MEASUREMENT_CAMERA_IDS = (2,)
ALPHA_THRESHOLD = 0.5
FIGURE_WIDTH_CM = 16.0
FIGURE_HEIGHT_CM = 11.0
EXPORT_DPI = 600
TITLE_SIZE = 13
LABEL_SIZE = 10
LEGEND_SIZE = 9


def load_saved_model(path, device="cuda"):
    with np.load(path) as data:
        required = ["features_dc", "features_rest", "active_sh_degree", "max_sh_degree",
                    "opacity_logit", "opacity_logconc", "scaling_parameter", "rotation_parameter"]
        for field in required:
            if field not in data:
                raise ValueError(f"{path} lacks {field}; this legacy export cannot reproduce the trained renderer.")
        n = len(data["mu_ref"])
        init = {"mu_ref": data["mu_ref"], "velocity": data["velocity"], "t_ref": data["t_ref"],
                "colors": data["colors"], "scales": data["scales"], "rotations": data["rotations"],
                "track_ids": data["track_ids"], "frame_indices": data["frame_indices"],
                "mode": str(data["mode"].item()), "track_times": data["t_ref"].reshape(-1, 1),
                "track_points": data["mu_ref"].reshape(n, 1, 3),
                "track_mask": np.ones((n, 1), dtype=np.float32)}
        if "temporal_mode" in data:
            init["temporal_mode"] = str(data["temporal_mode"].item())
        for field in ("source_indices", "parent_ids", "generation"):
            if field in data:
                init[field] = data[field]
        model = DynamicGaussianModel(init, max_sh_degree=int(data["max_sh_degree"]), device=device)
        with torch.no_grad():
            model.features_dc.copy_(torch.as_tensor(data["features_dc"], device=device))
            model.features_rest.copy_(torch.as_tensor(data["features_rest"], device=device))
            model.opacity_logit.copy_(torch.as_tensor(data["opacity_logit"], device=device))
            model.opacity_logconc.copy_(torch.as_tensor(data["opacity_logconc"], device=device))
            model.scaling.copy_(torch.as_tensor(data["scaling_parameter"], device=device))
            model.rotation.copy_(torch.as_tensor(data["rotation_parameter"], device=device))
        model.active_sh_degree = int(data["active_sh_degree"])
    model.eval()
    return model


def reliability_statistics(confidence, error):
    if not np.isfinite(confidence).all() or not np.isfinite(error).all():
        raise ValueError("Nonfinite evaluation values; refusing to silently discard pixels.")
    if len(error) < 2:
        return {"count": len(error), "spearman_confidence_negative_error": None, "bins": [], "risk_coverage": []}
    correlation = None
    # Numerical noise in an alpha-normalized constant field is not meaningful confidence.
    constant = np.ptp(confidence) <= 1e-5
    if not constant and np.ptp(error) > 0:
        correlation = float(spearmanr(confidence, -error).statistic)
    order = np.argsort(confidence, kind="stable")
    bins = []
    if constant:
        bins.append({"confidence": float(confidence.mean()), "mae": float(error.mean()), "count": len(error)})
    else:
        for indices in np.array_split(order, min(10, len(error))):
            bins.append({"confidence": float(confidence[indices].mean()),
                         "mae": float(error[indices].mean()), "count": len(indices)})
    descending = order[::-1]
    oracle = np.argsort(error, kind="stable")
    curves = []
    for coverage in np.linspace(0.1, 1.0, 10):
        count = max(1, int(round(len(error) * coverage)))
        # Include the entire boundary tie group instead of choosing spatially ordered pixels.
        cutoff = confidence[descending[count - 1]]
        selected = confidence >= cutoff
        risk = float(error.mean()) if constant else float(error[selected].mean())
        curves.append({"requested_coverage": float(coverage),
                       "actual_coverage": 1.0 if constant else float(selected.mean()),
                       "mae": risk, "random_expected_mae": float(error.mean()),
                       "oracle_mae": float(error[oracle[:count]].mean())})
    return {"count": len(error), "constant_confidence": bool(constant),
            "spearman_confidence_negative_error": correlation, "bins": bins, "risk_coverage": curves}


def evaluate(model, cameras, settings, output, use_exposure=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not cameras:
        raise ValueError("No held-out cameras selected.")
    plt.rcParams.update({"font.size": LABEL_SIZE, "axes.titlesize": TITLE_SIZE,
                         "axes.labelsize": LABEL_SIZE, "xtick.labelsize": LABEL_SIZE,
                         "ytick.labelsize": LABEL_SIZE, "legend.fontsize": LEGEND_SIZE,
                         "svg.fonttype": "none"})
    rows = []
    all_confidence = []
    all_variance = []
    all_error = []
    all_alpha = []
    example = None
    with torch.no_grad():
        scalar_values = [torch.ones_like(model.get_opacity_mean()),
                         torch.log(model.get_opacity_concentration()), model.get_opacity_variance()]
        for cam in cameras:
            rendered = render_exposure_integrated(model, cam, use_exposure=use_exposure, **settings)["image"]
            if not torch.isfinite(rendered).all():
                raise ValueError("Nonfinite rendered RGB.")
            prediction = rendered.clamp(0.0, 1.0)
            target = cam.image
            mse = float(torch.mean((prediction - target) ** 2))
            psnr = float(-10.0 * np.log10(max(mse, 1e-12)))
            ssim = float(1.0 - ssim_loss(prediction, target))
            error = torch.mean(torch.abs(prediction - target), dim=0).cpu().numpy()
            maps = []
            for values in scalar_values:
                result = render_scalar_exposure_integrated(model, cam, values, use_exposure=use_exposure, **settings)
                maps.append(result["map"].squeeze(0).cpu().numpy())
            alpha = maps[0]
            confidence = np.divide(maps[1], alpha, out=np.zeros_like(alpha), where=alpha > 1e-6)
            variance = np.divide(maps[2], alpha, out=np.zeros_like(alpha), where=alpha > 1e-6)
            supported = alpha >= ALPHA_THRESHOLD
            for array in (alpha, confidence, variance, error):
                if not np.isfinite(array).all():
                    raise ValueError("Nonfinite map; evaluation aborted.")
            stats = reliability_statistics(confidence[supported], error[supported])
            rows.append({"frame": cam.frame_idx, "camera": cam.camera_id, "psnr": psnr,
                         "ssim": ssim, "mae": float(error.mean()), "supported_fraction": float(supported.mean()),
                         "spearman": stats["spearman_confidence_negative_error"],
                         "rgb_clipped_fraction": float(((rendered < 0) | (rendered > 1)).float().mean())})
            all_confidence.append(confidence[supported])
            all_variance.append(variance[supported])
            all_error.append(error[supported])
            all_alpha.append(alpha[supported])
            pred_np = prediction.permute(1, 2, 0).cpu().numpy()
            target_np = target.permute(1, 2, 0).cpu().numpy()
            stem = f"cam{cam.camera_id:02d}_frame_{cam.frame_idx:06d}"
            Image.fromarray(np.rint(pred_np * 255).astype(np.uint8)).save(output / f"{stem}_render.png")
            Image.fromarray(np.rint(target_np * 255).astype(np.uint8)).save(output / f"{stem}_reference.png")
            np.savez_compressed(output / f"{stem}_maps.npz", rgb_raw=rendered.cpu().numpy(),
                                confidence=confidence, opacity_variance_score=variance,
                                alpha=alpha, absolute_rgb_error=error, supported=supported)
            if example is None:
                example = (target_np, pred_np, error, confidence, supported)
            print(f"Held-out {stem}: PSNR={psnr:.3f}, SSIM={ssim:.4f}, supported={supported.mean():.3f}", flush=True)
    confidence = np.concatenate(all_confidence)
    error = np.concatenate(all_error)
    variance = np.concatenate(all_variance)
    alpha = np.concatenate(all_alpha)
    reliability = reliability_statistics(confidence, error)
    summary = {
        "evaluation_type": "held-out cameras and/or timestamps",
        "evaluated_camera_ids": sorted({cam.camera_id for cam in cameras}),
        "evaluated_frame_ids": sorted({cam.frame_idx for cam in cameras}),
        "exposure_integration": bool(use_exposure),
        "n_images": len(rows), "mean_psnr": float(np.mean([r["psnr"] for r in rows])),
        "mean_ssim": float(np.mean([r["ssim"] for r in rows])),
        "mean_mae": float(np.mean([r["mae"] for r in rows])),
        "mean_supported_fraction": float(np.mean([r["supported_fraction"] for r in rows])),
        "confidence_definition": "alpha-normalized composite of per-Gaussian log(kappa)",
        "variance_definition": "alpha-normalized opacity-variance score, not predictive RGB/XYZ variance",
        "reliability_mask": f"rendered alpha >= {ALPHA_THRESHOLD}; RGB metrics use all pixels",
        "confidence_reliability": reliability,
        "negative_opacity_variance_reliability": reliability_statistics(-variance, error),
        "alpha_only_reliability": reliability_statistics(alpha, error),
        "geometry_metrics": None, "interval_coverage": None,
        "evaluation_notes": ["Camera calibration is read from the input dataset",
                             "SSIM uses the project's zero-padded 11x11 implementation",
                             "Risk curves retain all boundary ties"],
    }
    with open(output / "calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    with open(output / "per_view_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_CM / 2.54, FIGURE_HEIGHT_CM / 2.54), layout="constrained")
    bins = reliability["bins"]
    curves = reliability["risk_coverage"]
    axes[0].plot([b["confidence"] for b in bins], [b["mae"] for b in bins], "o-", color="#0072B2")
    axes[0].set(xlabel="Mean log concentration", ylabel="Mean absolute RGB error", title="Confidence vs error")
    axes[0].set_ylim(bottom=0)
    axes[1].plot([r["actual_coverage"] for r in curves], [r["mae"] for r in curves], "o-", label="Confidence")
    axes[1].plot([r["requested_coverage"] for r in curves], [r["random_expected_mae"] for r in curves], "--", label="Random expectation")
    axes[1].plot([r["requested_coverage"] for r in curves], [r["oracle_mae"] for r in curves], ":", label="Error oracle")
    axes[1].set(xlabel="Retained pixel fraction", ylabel="Mean absolute RGB error", title="Risk vs coverage", ylim=(0, None))
    axes[1].legend(loc="best")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"confidence_error_curve.{suffix}", dpi=EXPORT_DPI)
    plt.close(fig)
    target_np, pred_np, error_image, conf_image, supported = example
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_CM / 2.54, 13 / 2.54), layout="constrained")
    axes[0, 0].imshow(target_np)
    axes[0, 0].set_title("Held-out reference")
    axes[0, 1].imshow(pred_np)
    axes[0, 1].set_title("Rendered RGB")
    error_plot = axes[1, 0].imshow(error_image, cmap="magma", vmin=0, vmax=1)
    axes[1, 0].set_title("Absolute RGB error")
    fig.colorbar(error_plot, ax=axes[1, 0], shrink=0.7)
    # A shared 0..log(31) scale covers the baseline initialization and the 1..21 evidence target.
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    confidence_plot = axes[1, 1].imshow(np.ma.array(conf_image, mask=~supported), cmap=cmap, vmin=0, vmax=np.log(31))
    axes[1, 1].set_title("Confidence: log kappa")
    fig.colorbar(confidence_plot, ax=axes[1, 1], shrink=0.7)
    for ax in axes.flat:
        ax.set_axis_off()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"held_out_example.{suffix}", dpi=EXPORT_DPI)
    plt.close(fig)
    return summary


def main(run_dir=RUN_DIR):
    run_dir = Path(run_dir)
    with open(run_dir / "train_config.json", encoding="utf-8") as f:
        config = json.load(f)
    with open(run_dir / "render_settings.json", encoding="utf-8") as f:
        settings = json.load(f)
    sfm_path = Path(config["sfm_json"])
    if not sfm_path.is_absolute():
        sfm_path = ROOT / sfm_path
    manifest_path = Path(config["views_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if (sfm_path.parent / "evaluation/sfm.json").is_file():
        sfm_path = sfm_path.parent / "evaluation/sfm.json"
        manifest_path = sfm_path.parent / "views_manifest.json"
    cameras = load_openmvg_sequence(str(sfm_path), views_manifest=str(manifest_path),
                                   target_width=config["target_width"], device="cuda")
    held_cameras = set(config["held_out_camera_ids"]) - set(RESERVED_MEASUREMENT_CAMERA_IDS)
    held_frames = set(config["held_out_frame_ids"])
    test_cameras = [cam for cam in cameras
                    if cam.camera_id in held_cameras or cam.frame_idx in held_frames]
    model = load_saved_model(run_dir / "dynamic_gaussians_final.npz")
    from demo_config import new_output
    return evaluate(model, test_cameras, settings, new_output("image_evaluation"), use_exposure=True)


if __name__ == "__main__":
    main()
