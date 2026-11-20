"""Pixel, projection, disparity, physical-scale and PFM orientation checks."""

import json
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from data_io import CameraInfo, load_openmvg_sequence
from model import DynamicGaussianModel
from prepare_eth3d_geometry import reconstruct_rectified_pair
from renderer import render_exposure_integrated


def run_checks():
    xyz = np.array([[0., 0., 3.]], dtype=np.float32)
    init = dict(mu_ref=xyz, velocity=np.zeros_like(xyz), t_ref=np.array([0.]),
                colors=np.ones_like(xyz), scales=np.array([[.06, .06, .01]]),
                rotations=np.array([[1., 0, 0, 0]]), track_ids=np.array([0]),
                track_times=np.array([[0.]]), track_points=xyz[:, None, :],
                track_mask=np.ones((1, 1)), frame_indices=np.array([0]), mode="flow_sand")
    cam = CameraInfo(torch.zeros((3, 64, 64), device="cuda"), "test",
                     torch.eye(4, device="cuda"), torch.zeros(3, device="cuda"),
                     50., 55., 32., 32., 64, 64, 2 * np.arctan(64 / 100),
                     2 * np.arctan(64 / 110), 0, 0., 0, 0.)
    model = DynamicGaussianModel(init, device="cuda")
    with torch.no_grad():
        y, x = torch.meshgrid(torch.arange(64, device="cuda"), torch.arange(64, device="cuda"), indexing="ij")
        for offset, expected in ((0.0, 31.5), (0.5, 32.0)):
            cam.raster_pixel_offset = offset
            values = render_exposure_integrated(model, cam)["image"][0]
            centroid = [float((values * x).sum() / values.sum()), float((values * y).sum() / values.sum())]
            np.testing.assert_allclose(centroid, [expected, expected], atol=1e-5, rtol=0)
    print("PASS CUDA: legacy half-pixel replay and explicit OpenCV integer-center projection.")

    with tempfile.TemporaryDirectory(prefix="measurement_conventions_") as directory:
        root = Path(directory)
        expected = np.array([[1, 2, np.inf], [3, 4, 5]], dtype=np.float32)
        with open(root / "depth.pfm", "wb") as f:
            f.write(b"Pf\n3 2\n-1\n")
            f.write(expected[::-1].astype("<f4").tobytes())
        loaded = cv2.imread(str(root / "depth.pfm"), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(loaded, expected)
        cv2.imwrite(str(root / "im0.png"), np.zeros((48, 64, 3), dtype=np.uint8))
        sfm = {"root_path": str(root), "pixel_coordinates": "opencv_integer_centers",
               "views": [{"key": 0, "value": {"ptr_wrapper": {"data": {
                   "filename": "im0.png", "id_pose": 0, "id_intrinsic": 0}}}}],
               "intrinsics": [{"key": 0, "value": {"ptr_wrapper": {"data": {
                   "focal_length": 50., "focal_length_y": 55., "principal_point": [32., 24.]}}}}],
               "extrinsics": [{"key": 0, "value": {"rotation": np.eye(3).tolist(), "center": [0, 0, 0]}}]}
        with open(root / "sfm.json", "w", encoding="utf-8") as f:
            json.dump(sfm, f)
        camera = load_openmvg_sequence(str(root / "sfm.json"), target_width=32, device="cpu")[0]
        np.testing.assert_allclose([camera.fx, camera.fy, camera.cx, camera.cy], [25, 27.5, 15.75, 11.75])
        assert camera.raster_pixel_offset == 0.5
    print("PASS: little-endian PFM, bottom-up rows, invalid infinity, pixel-center-aware image resizing.")

    rng = np.random.default_rng(42)
    left = rng.integers(0, 256, (128, 320, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-16] = left[:, 16:]
    K = np.array([[200., 0, 160.], [0, 200., 64.], [0, 0, 1.]])
    calibration = {"cam0": K, "cam1": K, "baseline_m": 0.1, "doffs": 0.}
    points, _, disparity, valid, _ = reconstruct_rectified_pair(left, right, calibration)
    assert len(points) > 5000
    assert np.median(disparity[valid]) == 16
    np.testing.assert_allclose(np.median(points[:, 2]), 1.25, atol=1e-6, rtol=0)
    assert np.isfinite(points).all()
    print("PASS: independent 16-pixel stereo fixture reconstructs 1.25 m depth with 0.1 m baseline.")


if __name__ == "__main__":
    run_checks()
