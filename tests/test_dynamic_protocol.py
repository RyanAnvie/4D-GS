"""Shutter integration and time-split leakage checks. Direct Spyder execution."""

from types import SimpleNamespace
from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data_io import CameraInfo, SequenceFrame, initialize_sand_surface
from renderer import exposure_sample_offsets, render_exposure_integrated
from train_4dgs import select_training_observations
from prepare_controlled_dynamics import render_mesh_views, make_surface
from model import DynamicGaussianModel
from calibration_eval import load_saved_model


def run_checks():
    exposure = 0.04
    expected = exposure ** 2 / 12.0  # Exact uniform-shutter mean of t**2.
    errors = []
    for count in (4, 8, 16, 32):
        times = exposure_sample_offsets(exposure, count, "cpu", rule="midpoint")
        assert abs(float(times.mean())) < 1e-8
        errors.append(abs(float(times.square().mean()) - expected))
    assert all(errors[i + 1] < errors[i] / 3.9 for i in range(3))
    old = exposure_sample_offsets(exposure, 5, "cpu")
    assert torch.equal(old, torch.linspace(-0.02, 0.02, 5))
    assert exposure_sample_offsets(0., 5, "cpu", rule="midpoint").tolist() == [0.]
    print("PASS: midpoint shutter converges quadratically; legacy offsets unchanged.")

    frames = []
    cameras = []
    for frame_idx in range(3):
        xyz = np.array([[0., 0., 1.], [0.01, 0., 1.], [0., 0.01, 1.], [0.01, 0.01, 1.]], dtype=np.float32)
        xyz[:, 2] += 0.001 * frame_idx
        if frame_idx == 1:
            xyz += 100.0  # Poisoned held-out cloud must never reach the initializer.
        frames.append(SequenceFrame(frame_idx, frame_idx * 0.05, xyz, np.ones_like(xyz) * 0.5,
                                    np.tile([0., 0., -1.], (4, 1)), np.arange(4), np.ones(4)))
        for camera_id in range(3):
            cameras.append(SimpleNamespace(frame_idx=frame_idx, camera_id=camera_id,
                           timestamp=frame_idx * 0.05, image_path=f"{frame_idx}_{camera_id}.png"))
    selected_cameras, selected_frames, split = select_training_observations(cameras, frames, (2,), (1,))
    assert len(selected_cameras) == 4
    assert split["training_cloud_frames"] == [0, 2]
    init = initialize_sand_surface(selected_frames)
    assert set(init["frame_indices"]) == {0, 2}
    assert np.max(init["mu_ref"]) < 2.
    assert np.max(np.abs(init["velocity"])) < 0.1
    print("PASS: held-out image and poisoned cloud excluded before flow initialization.")
    legacy = DynamicGaussianModel(init, device="cpu")
    assert len(legacy.active_indices_at(0.05, 0.0375)) == 1
    init["temporal_mode"] = "nearest_frame"
    nearest = DynamicGaussianModel(init, device="cpu")
    active = nearest.active_indices_at(0.05, 0.06)
    assert len(active) == 4
    assert torch.equal(nearest.temporal_weights(0.05, active, 0.01), torch.ones(4, 1))
    rejected = False
    try:
        nearest.active_indices_at(-1., 0.06)
    except ValueError:
        rejected = True
    assert rejected
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "nearest.npz"
        nearest.save_npz(str(path))
        repeated = load_saved_model(path, device="cpu")
        assert repeated.temporal_mode == "nearest_frame"
        assert torch.equal(active, repeated.active_indices_at(0.05, 0.06))
    print("PASS: legacy gap reproduced; nearest-frame mode retains a full surface and rejects unsupported extrapolation.")

    init["velocity"] = np.zeros_like(init["velocity"])
    init["colors"][init["frame_indices"] == 2] = 0.95
    static = DynamicGaussianModel(init, device="cuda")
    cam = CameraInfo(image=torch.zeros((3, 64, 64), device="cuda"), image_path="fixture",
                     w2c=torch.eye(4, device="cuda"), camera_center=torch.zeros(3, device="cuda"),
                     fx=50., fy=50., cx=32., cy=32., width=64, height=64,
                     fovx=2*np.arctan(64/100), fovy=2*np.arctan(64/100),
                     frame_idx=1, timestamp=0.05, camera_id=0, exposure=0.04, raster_pixel_offset=0.5)
    with torch.no_grad():
        center = render_exposure_integrated(static, cam, use_exposure=False, temporal_window=0.071)["image"]
        integrated = render_exposure_integrated(static, cam, exposure_samples=5, temporal_window=0.071,
                        exposure_rule="midpoint", shutter_reference_mode="center_nearest")["image"]
        switching = render_exposure_integrated(static, cam, exposure_samples=5, temporal_window=0.071,
                        exposure_rule="midpoint", shutter_reference_mode="sample_nearest")["image"]
    assert torch.allclose(integrated, center, atol=1e-7, rtol=1e-6)
    assert float((switching - center).abs().max()) > 1e-3
    print("PASS: fixed-reference shutter leaves zero-motion images unchanged; frame-switching artifact reproduced.")

    vertices = np.array([[-2., -2., 1.25], [2., -2., 1.25],
                         [-2., 2., 1.25], [2., 2., 1.25]], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
    uv = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=np.float32)
    texture = np.full((16, 16, 3), 0.4, dtype=np.float32)
    centers = np.array([[0., 0., 0.], [0.1, 0., 0.]], dtype=np.float32)
    images, depth, valid = render_mesh_views(vertices, triangles, uv, texture, centers,
                                            width=64, height=48, focal=50.)
    assert valid.all()
    assert np.allclose(depth, 1.25, atol=1e-6)
    assert np.allclose(images, 0.4, atol=1e-6)
    first = make_surface("translation", 0.)[0]
    later = make_surface("translation", 0.1)[0]
    assert np.allclose(later - first, [0.035, 0.003, 0.02], atol=1e-6)
    print("PASS: independent ray caster recovers metric plane depth and designed motion.")


if __name__ == "__main__":
    run_checks()
