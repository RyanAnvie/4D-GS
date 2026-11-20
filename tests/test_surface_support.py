"""Synthetic tilted-surface tests; directly runnable in envtorch/Spyder."""

import json
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from calibration_eval import load_saved_model
from densification import split_surface_gaussians
from density_eval import evaluate_density
from model import DynamicGaussianModel
from surface_support import build_surface_support, project_surface_centers, surface_support_loss
from train_4dgs import TrainConfig


def run_checks():
    angle = np.pi / 4
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float32)
    xy = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    xyz = np.concatenate((xy @ rotation.T, xy @ rotation.T + [0, 0, 2])).astype(np.float32)
    quaternion = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
    init = {"mu_ref": xyz, "velocity": np.zeros_like(xyz), "t_ref": np.repeat([0., 0.04], 4),
            "colors": np.full_like(xyz, 0.6), "scales": np.tile([1., 1., 0.08], (8, 1)),
            "rotations": np.tile(quaternion, (8, 1)), "track_ids": np.arange(8),
            "track_times": np.repeat([0., 0.04], 4)[:, None], "track_points": xyz[:, None, :],
            "track_mask": np.ones((8, 1)), "frame_indices": np.repeat([0, 1], 4), "mode": "flow_sand"}
    config = TrainConfig(surface_constraint=True)
    for device in ("cpu", "cuda"):
        # Copy arrays because CPU torch.as_tensor can share NumPy storage.
        copied = {name: value.copy() if isinstance(value, np.ndarray) else value for name, value in init.items()}
        model = DynamicGaussianModel(copied, max_sh_degree=1, device=device)
        support = build_surface_support(init, device)
        torch.testing.assert_close(support["normals"], torch.tensor(rotation[:, 2], device=device).expand(8, 3))
        original_support = {name: value.clone() for name, value in support.items()}
        model.set_train_stage(2)
        optimizer = model.optimizer(1, 100)
        model.mu_ref.sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        normal = support["normals"]
        tangent_axis = torch.tensor(rotation[:, 0], device=device)
        with torch.no_grad():
            model.mu_ref.copy_(support["points"])
            model.mu_ref[0] += 2 * normal[0] + 2 * tangent_axis
            model.mu_ref[4] -= 2 * normal[4] + 2 * tangent_axis
        old_momentum = optimizer.state[model.mu_ref]["exp_avg"].clone()
        old_variance = optimizer.state[model.mu_ref]["exp_avg_sq"].clone()
        unprojected = model.mu_ref.detach().clone()
        config.surface_projection_reset_moments = False
        retained_report = project_surface_centers(model, optimizer, support, config, report=True)
        assert retained_report["projected_centers"] == 2
        retained_projection = model.mu_ref.detach().clone()
        torch.testing.assert_close(optimizer.state[model.mu_ref]["exp_avg"], old_momentum, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[model.mu_ref]["exp_avg_sq"], old_variance, rtol=0, atol=0)
        with torch.no_grad():
            model.mu_ref.copy_(unprojected)
        config.surface_projection_reset_moments = True
        other_parameters = {name: value.clone() for name, value in model.named_parameters() if name != "mu_ref"}
        report = project_surface_centers(model, optimizer, support, config, report=True)
        assert report["projected_centers"] == 2
        torch.testing.assert_close(model.mu_ref, retained_projection, rtol=0, atol=0)
        retained = torch.tensor([1, 2, 3, 5, 6, 7], device=device)
        torch.testing.assert_close(optimizer.state[model.mu_ref]["exp_avg"][retained], old_momentum[retained])
        assert not optimizer.state[model.mu_ref]["exp_avg"][[0, 4]].any()
        for name, old in other_parameters.items():
            torch.testing.assert_close(getattr(model, name), old, atol=0, rtol=0)
        expected = support["points"][0] + 0.5 * tangent_axis + 0.08 * normal[0]
        torch.testing.assert_close(model.mu_ref[0], expected, atol=1e-6, rtol=0)

        # Test parents on the disk rim and subsequent generations, in both frames.
        split_surface_gaussians(model, optimizer, torch.tensor([0, 4], device=device), support, config)
        split_surface_gaussians(model, optimizer, torch.tensor([6, 8], device=device), support, config)
        assert model.n_gaussians == 12 and model.track_ids.unique().numel() == 12
        assert model.frame_indices.bincount().tolist() == [6, 6]
        assert torch.unique(model.mu_ref, dim=0).shape[0] == 12
        for name in support:
            torch.testing.assert_close(support[name], original_support[name], atol=0, rtol=0)
        indices = torch.arange(model.n_gaussians, device=device)
        loss = surface_support_loss(model, support, indices)
        loss.backward()
        assert torch.isfinite(model.mu_ref.grad).all() and model.mu_ref.grad.abs().sum() > 0
        optimizer.step()
        project_surface_centers(model, optimizer, support, config)
        source = model.source_indices
        displacement = model.mu_ref - support["points"][source]
        height = (displacement * support["normals"][source]).sum(dim=1, keepdim=True)
        tangent = displacement - height * support["normals"][source]
        assert (height.abs() <= 0.080001).all()
        assert (torch.linalg.vector_norm(tangent, dim=1) <= 0.500001).all()
        torch.testing.assert_close(model.frame_indices, support["frame_indices"][source])
        torch.testing.assert_close(model.t_ref.reshape(-1), support["t_ref"][source])

        with tempfile.TemporaryDirectory(prefix="surface_support_test_") as directory:
            run_dir = Path(directory)
            model.save_npz(str(run_dir / "dynamic_gaussians_final.npz"))
            loaded = load_saved_model(run_dir / "dynamic_gaussians_final.npz", device=device)
            torch.testing.assert_close(model.mu_ref, loaded.mu_ref, atol=0, rtol=0)
            torch.testing.assert_close(model.source_indices, loaded.source_indices, atol=0, rtol=0)
            np.savez_compressed(run_dir / "surface_support.npz",
                                **{name: value.cpu().numpy() for name, value in support.items()})
            with open(run_dir / "train_config.json", "w", encoding="utf-8") as f:
                json.dump(vars(config), f)
            reference_dir = run_dir / "reference"
            reference_dir.mkdir()
            for frame in (0, 1):
                mask = init["frame_indices"] == frame
                np.savez(reference_dir / f"frame_{frame:06d}.npz", points=xyz[mask],
                         normals=np.tile(rotation[:, 2], (4, 1)))
            metrics = evaluate_density(run_dir, reference_dir)
            assert metrics["total_support_violations"] == 0
            assert metrics["total_unique_centers_per_frame"] == 12
            assert metrics["mean_frame_model_to_training_cloud_mean"] < np.hypot(0.5, 0.08)
        print(f"PASS {device}: tilted planes, identical projection with retained/reset Adam first/second moments, distinct repeated splits, immutable evidence roots, save/reload, geometry audit.")


if __name__ == "__main__":
    run_checks()
