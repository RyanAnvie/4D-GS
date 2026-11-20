"""Spyder F5: rerender and measure both bundled models against mesh references."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from calibration_eval import load_saved_model
from controlled_dynamics_eval import evaluate_controlled
from data_io import load_openmvg_sequence
from demo_config import RESULTS, new_output


EXPERIMENTS = ("translation", "deformation")


def main():
    output = new_output("evaluation")
    results = {}
    for experiment in EXPERIMENTS:
        run = RESULTS / experiment / "sensitivity/fixed_on"
        data = ROOT / "examples/data" / experiment / "evaluation"
        model = load_saved_model(run / "dynamic_gaussians_final.npz")
        with open(run / "render_settings.json", encoding="utf-8") as file:
            settings = json.load(file)
        cameras = load_openmvg_sequence(str(data / "sfm.json"),
            views_manifest=str(data / "views_manifest.json"), target_width=448, device="cuda")
        metrics = evaluate_controlled(model, cameras, settings, data / "geometry",
            output / experiment / "independent_evaluation", [1, 3, 5], [5])
        selected = []
        for row in metrics["image_metrics"]:
            time_group = "unseen_time" if row["frame"] in (1, 3, 5) else "seen_time"
            if row["camera"] == 2:
                view_group = "reserved_measurement_view"
            elif row["camera"] == 5:
                view_group = "novel_view"
            else:
                view_group = "train_view"
            row["group"] = time_group + "_" + view_group
            if row["frame"] in (1, 3, 5) or row["camera"] == 5:
                row["role"] = "evaluation"
                selected.append(row)
            elif row["camera"] == 2:
                row["role"] = "reserved_measurement"
            else:
                row["role"] = "fit"
        assert len(selected) == 22
        metrics["image_groups"] = {}
        for group in sorted({row["group"] for row in metrics["image_metrics"]}):
            group_rows = [row for row in metrics["image_metrics"] if row["group"] == group]
            metrics["image_groups"][group] = {"images": len(group_rows)}
            for name in ("psnr", "ssim", "mae", "alpha_ge_0_5_fraction"):
                metrics["image_groups"][group][name] = float(np.mean([row[name] for row in group_rows]))
        metrics["evaluation_images"] = {"images": len(selected)}
        for name in ("psnr", "ssim", "mae", "alpha_ge_0_5_fraction"):
            metrics["evaluation_images"][name] = float(np.mean([row[name] for row in selected]))
        with open(output / experiment / "independent_evaluation/summary.json", "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, allow_nan=False)
        results[experiment] = metrics
        del model, cameras
        torch.cuda.empty_cache()
    with open(output / "summary.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=False)
    print("Evaluation saved:", output)
    return output


if __name__ == "__main__":
    OUTPUT = main()
