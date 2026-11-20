"""Numerical and CUDA tests for real Gaussian insertion and optimizer resizing."""

import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from calibration_eval import load_saved_model
from data_io import CameraInfo
from densification import (accumulate_screen_gradients, new_densification_stats,
                           select_split_parents, split_surface_gaussians)
from gs_uncertainty.losses import kappa_evidence_loss
from model import DynamicGaussianModel
from renderer import render_exposure_integrated
from train_4dgs import TrainConfig, estimate_clip_planes
from surface_support import build_surface_support


def run_checks():
    torch.manual_seed(42)
    xyz = np.array([[-0.2, -0.2, 3], [0.2, -0.2, 3], [0, 0.2, 3],
                    [-0.2, -0.1, 3], [0.2, -0.1, 3], [0, 0.3, 3]], dtype=np.float32)
    init = {"mu_ref": xyz, "velocity": np.zeros_like(xyz), "t_ref": np.repeat([0., 0.04], 3),
            "colors": np.full_like(xyz, 0.6), "scales": np.tile([0.12, 0.08, 0.01], (6, 1)),
            "rotations": np.tile([1., 0., 0., 0.], (6, 1)), "track_ids": np.arange(6),
            "track_times": np.repeat([0., 0.04], 3)[:, None], "track_points": xyz[:, None, :],
            "track_mask": np.ones((6, 1)), "frame_indices": np.repeat([0, 1], 3), "mode": "flow_sand"}
    init["rotations"][0] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    input_arrays = {name: value.copy() for name, value in init.items() if isinstance(value, np.ndarray)}
    for device in ("cpu", "cuda"):
        model = DynamicGaussianModel(init, max_sh_degree=1, device=device)
        model.set_train_stage(2)
        optimizer = model.optimizer(1, 100)
        loss = sum(parameter.square().sum() for parameter in model.parameters() if parameter.requires_grad)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        old = {name: value.detach().clone() for name, value in model.named_parameters()}
        old_mean = optimizer.state[model.mu_ref]["exp_avg"].clone()
        old_step = optimizer.state[model.mu_ref]["step"].clone()
        parents = torch.tensor([0, 3], device=device)
        retained = torch.tensor([1, 2, 4, 5], device=device)
        split_surface_gaussians(model, optimizer, parents)
        assert model.n_gaussians == 8
        assert model.track_ids.unique().numel() == 8
        assert model.frame_indices.tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
        assert model.source_indices.tolist() == [1, 2, 4, 5, 0, 0, 3, 3]
        assert model.parent_ids.tolist() == [-1, -1, -1, -1, 0, 0, 3, 3]
        assert model.generation.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
        torch.testing.assert_close(model.mu_ref[:4], old["mu_ref"][retained])
        for j, parent in enumerate(parents):
            children = model.mu_ref[4 + 2*j:6 + 2*j]
            torch.testing.assert_close(children.mean(dim=0), old["mu_ref"][parent])
            assert not torch.equal(children[0], children[1])
            offset = children - old["mu_ref"][parent]
            rotation = torch.eye(3, device=device)
            if int(parent) == 0:
                rotation = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1]], device=device)
            covariance = rotation @ torch.diag(model.scaling[4 + 2*j].exp().square()) @ rotation.T
            covariance += offset.T @ offset / 2
            expected = rotation @ torch.diag(old["scaling"][parent].exp().square()) @ rotation.T
            torch.testing.assert_close(covariance, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(model.opacity_logconc[4:], old["opacity_logconc"][parents.repeat_interleave(2)])
        torch.testing.assert_close(optimizer.state[model.mu_ref]["exp_avg"][:4], old_mean[retained])
        assert not optimizer.state[model.mu_ref]["exp_avg"][4:].any()
        torch.testing.assert_close(optimizer.state[model.mu_ref]["step"], old_step)
        assert model.mu_ref.requires_grad and not model.velocity.requires_grad and not model.scaling.requires_grad
        for group in optimizer.param_groups:
            assert group["params"][0] is getattr(model, group["name"])
        for value in model.buffers():
            assert value.shape[0] == 8

        stats = new_densification_stats(model)
        config = TrainConfig(densify_fraction=1.0, densify_max_per_frame=5)
        selected, _ = select_split_parents(model, stats, config)
        assert selected.numel() == 0
        stats["observations"].fill_(3)
        stats["gradient_sum"][:] = torch.arange(1, 9, device=device)
        stats["max_radius"].fill_(5)
        selected, rows = select_split_parents(model, stats, config)
        assert selected.tolist() == [5, 7] and all(row["after"] == 5 for row in rows)
        split_surface_gaussians(model, optimizer, selected)
        assert model.n_gaussians == 10
        targets = torch.arange(6, device=device).reshape(-1, 1).float() + 5
        model.set_train_stage(5)
        loss = kappa_evidence_loss(model, targets[model.source_indices])
        loss = loss + sum(parameter.square().mean() for parameter in model.parameters())
        loss.backward()
        optimizer.step()
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        print(f"PASS {device}: distinct children, covariance, IDs, frame/evidence alignment, Adam state, caps, subsequent update.")

        selection_model = DynamicGaussianModel(init, device=device)
        support = build_surface_support(init, device)
        selection_stats = new_densification_stats(selection_model)
        selection_stats["observations"].fill_(3)
        selection_stats["max_radius"].fill_(5)
        selection_stats["gradient_sum"][:] = torch.tensor([100., 10., 1., 8., 2., 1.], device=device)
        with torch.no_grad():
            selection_model.mu_ref[0, 2] += .05
            selection_model.mu_ref[1, 0] += .05
        selection_config = TrainConfig(densify_fraction=1., densify_max_per_frame=4)
        selected, _ = select_split_parents(selection_model, selection_stats, selection_config)
        assert selected.tolist() == [0, 3]
        selection_config.densify_geometry_filter = True
        selected, rows = select_split_parents(selection_model, selection_stats, selection_config, support)
        assert selected.tolist() == [1, 3]
        assert rows[0]["geometry_rejected"] == 1 and rows[1]["geometry_rejected"] == 0
        torch.testing.assert_close(support["points"][0], torch.as_tensor(xyz[0], device=device))
        for name, original in input_arrays.items():
            np.testing.assert_array_equal(init[name], original)
        print(f"PASS {device}: normal-drift gate rejects high-gradient off-plane parent; tangent displacement remains eligible.")
        print(f"PASS {device}: optimization, splitting and explicit center changes preserve caller input arrays.")

    model = DynamicGaussianModel(init, max_sh_degree=1, device="cuda")
    cam = CameraInfo(image=torch.zeros((3, 64, 64), device="cuda"), image_path="synthetic",
                     w2c=torch.eye(4, device="cuda"), camera_center=torch.zeros(3, device="cuda"),
                     fx=50., fy=50., cx=32., cy=32., width=64, height=64,
                     fovx=2*np.arctan(64/100), fovy=2*np.arctan(64/100),
                     frame_idx=0, timestamp=0.02, camera_id=0, exposure=0.04)
    output = render_exposure_integrated(model, cam, exposure_samples=3,
                                        retain_screen_grad=True, temporal_window=0.015)
    assert [len(sample["indices"]) for sample in output["screen_samples"]] == [3, 1, 3]
    output["image"][:, :, :32].sum().backward()
    stats = new_densification_stats(model)
    accumulate_screen_gradients(stats, output["screen_samples"])
    assert stats["gradient_sum"].sum() > 0 and torch.isfinite(stats["gradient_sum"]).all()
    assert (stats["observations"] > 0).all()
    assert output["radii"].shape == (3,)
    optimizer = model.optimizer(1, 100)
    split_surface_gaussians(model, optimizer, torch.tensor([0], device="cuda"))
    with tempfile.TemporaryDirectory(prefix="densification_test_") as directory:
        path = Path(directory) / "model.npz"
        model.save_npz(str(path))
        loaded = load_saved_model(path)
        with torch.no_grad():
            before = render_exposure_integrated(model, cam, temporal_window=0.015)["image"]
            after = render_exposure_integrated(loaded, cam, temporal_window=0.015)["image"]
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(model.source_indices, loaded.source_indices)
    print("PASS CUDA: shutter samples with different active IDs, screen gradients, split model save/reload.")

    # Reproduce the large-array clipping failure without downloading or changing data.
    points = np.zeros(((1 << 24) + 1, 3), dtype=np.float32)
    points[:, 2] = np.linspace(200, 400, len(points), dtype=np.float32)
    near, far = estimate_clip_planes([SimpleNamespace(frame_idx=0, points=points)], [cam], 1.0)
    np.testing.assert_allclose([near, far], [39.98, 599.7], atol=1e-4, rtol=0)
    print("PASS: clip planes with more than 2**24 depth samples, no data subsampling.")


if __name__ == "__main__":
    run_checks()
