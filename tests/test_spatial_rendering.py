"""Analytic scalar-depth, camera-Z ray and all-pixel partition checks."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data_io import CameraInfo
from diagnose_spatial_rendering import image_diagnostics, mesh_camera_depth, render_center_depth
from gs_uncertainty.rendering import render_scalar_exposure_integrated
from model import DynamicGaussianModel


def run_checks():
    angle = .3
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]], dtype=np.float32)
    center = np.array([.4, -.2, .1], dtype=np.float32)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = -rotation @ center
    camera = CameraInfo(torch.zeros((3, 64, 64), device="cuda"), "analytic",
        torch.tensor(w2c, device="cuda"), torch.tensor(center, device="cuda"),
        50., 50., 32., 32., 64, 64, 2*np.arctan(64/100), 2*np.arctan(64/100),
        0, .2, 0, .04, raster_pixel_offset=.5)
    settings = {"exposure_samples": 5, "exposure_rule": "midpoint", "znear": .01, "zfar": 100.}
    camera_points = np.array([[0., 0., 2.], [0., 0., 4.]], dtype=np.float32)
    velocity = np.tile(np.array([[0., 0., .5]], dtype=np.float32) @ rotation, (2, 1))
    points = camera_points @ rotation + center
    init = dict(mu_ref=points, velocity=velocity, t_ref=np.zeros(2), colors=np.ones((2, 3)),
        scales=np.full((2, 3), .05), rotations=np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
        track_ids=np.arange(2), track_times=np.zeros((2, 1)), track_points=points[:, None, :],
        track_mask=np.ones((2, 1)), frame_indices=np.zeros(2, dtype=int), mode="flow_sand")
    model = DynamicGaussianModel(init, device="cuda")
    with torch.no_grad():
        result = render_center_depth(model, camera, settings)
        near = render_scalar_exposure_integrated(model, camera, torch.tensor([1., 0.], device="cuda"),
            use_exposure=False, **settings)["map"].squeeze()
        far = render_scalar_exposure_integrated(model, camera, torch.tensor([0., 1.], device="cuda"),
            use_exposure=False, **settings)["map"].squeeze()
        torch.testing.assert_close(result["alpha"], near+far, rtol=2e-6, atol=1e-7)
        # Motion at t=.2 adds .1 m camera-Z; opacity/compositing weights are unchanged.
        expected = 2.1*near+4.1*far
        torch.testing.assert_close(result["numerator_m"], expected, rtol=2e-6, atol=1e-7)
        supported = result["alpha"] > 0
        assert supported.any() and (~supported).any()
        assert torch.isnan(result["depth_m"][~supported]).all()
        assert torch.all(result["depth_m"][supported] >= 2.1-2e-6)
        assert torch.all(result["depth_m"][supported] <= 4.1+2e-6)
        for value in init:
            if isinstance(init[value], np.ndarray):
                init[value] = init[value][:1].copy()
        single = DynamicGaussianModel(init, device="cuda")
        one = render_center_depth(single, camera, settings)
        torch.testing.assert_close(one["depth_m"][one["alpha"] > 0], torch.full_like(one["depth_m"][one["alpha"] > 0], 2.1), rtol=0, atol=2e-6)
    vertices_camera = np.array([[-4., -4., 2.], [4., -4., 2.], [-4., 4., 2.], [4., 4., 2.]], dtype=np.float32)
    vertices_world = vertices_camera @ rotation + center
    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
    plane = mesh_camera_depth(vertices_world, triangles, camera)
    np.testing.assert_allclose(plane, 2., rtol=0, atol=1e-6)
    residual = np.arange(24, dtype=np.float32).reshape(3, 2, 4)/24
    alpha = np.array([[0., .25, .5, .75], [1., .125, .375, .625]], dtype=np.float32)
    depth = np.full((2, 4), 2., dtype=np.float32)
    depth[0, 0] = np.nan
    visible = np.array([[True, True, True, True], [False, False, False, False]])
    truth = np.where(visible, 2.5, np.inf).astype(np.float32)
    row = image_diagnostics(residual, alpha, depth, alpha, truth, visible)
    assert row["reference_visible_without_depth"] == 1 and row["depth_supported_pixels"] == 3
    assert row["depth_signed_error_sum_m"] == -1.5 and row["depth_absolute_error_sum_m"] == 1.5
    assert row["rendered_without_reference"] == 4
    np.testing.assert_allclose(sum(part["contribution_to_full_mse"] for part in row["rgb_regions"]), np.mean(residual.astype(float)**2), rtol=1e-12)
    assert sum(part["pixels"] for part in row["rgb_regions"]) == 8
    print("PASS: CUDA one/two-layer scalar depth, moving centers, nonidentity camera, analytic plane, zero support and all-pixel partitions.", flush=True)


if __name__ == "__main__":
    run_checks()
