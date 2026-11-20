"""Verify the public-result video assets and the orbit camera convention."""

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from data_io import CameraInfo
from export_public_video import orbit_camera


def run_checks():
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    center = np.array([1., 2., 3.])
    w2c = np.eye(4)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = -rotation @ center
    for offset in (0., 0.5):
        base = CameraInfo(torch.zeros((3, 360, 480), device="cuda"), "unit_test",
            torch.tensor(w2c, dtype=torch.float32, device="cuda"),
            torch.tensor(center, dtype=torch.float32, device="cuda"),
            400., 400., 240., 180., 480, 360, 1., 1., 0, 0., 0, 0.,
            raster_pixel_offset=offset)
        unchanged = orbit_camera(base, 10., 0., 0., width=480, height=360)
        np.testing.assert_allclose(unchanged.w2c.cpu(), base.w2c.cpu(), atol=1e-6)
        np.testing.assert_allclose(unchanged.camera_center.cpu(), center, atol=1e-6)
        assert unchanged.cx == base.cx and unchanged.cy == base.cy
        for angle in (-10., 10.):
            camera = orbit_camera(base, 10., math.radians(angle), 0.)
            transformed = camera.w2c.cpu().numpy()
            R = transformed[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
            np.testing.assert_allclose(np.linalg.det(R), 1., atol=1e-6)
            target = center + rotation.T @ np.array([0., 0., 10.])
            np.testing.assert_allclose((transformed @ np.append(target, 1))[:3], [0, 0, 10], atol=1e-5)
            expected_cx = (base.cx + offset) * camera.width / base.width
            np.testing.assert_allclose(camera.cx + offset, expected_cx, rtol=0, atol=1e-12)

    root = ROOT / "media/cook_spinach"
    with open(root / "video_manifest.json", encoding="utf-8") as file:
        manifest = json.load(file)
    assert manifest["source_frame_ids"] == list(range(30))
    assert manifest["opacity_filter"] is None
    assert manifest["scale_modifier"] == 1.0
    assert not manifest["preview_only"]
    for row in manifest["models"].values():
        assert row["iterations"] == 6000 and row["sh_degree"] == 1
        assert row["active_gaussians_per_timestamp"] == [100000] * 30
        assert row["native_render_max_uint8_difference"] <= 1
    for name, size in (("cook_spinach_comparison.mp4", (1920, 1080)),
                       ("cook_spinach_3d.mp4", (1920, 900))):
        capture = cv2.VideoCapture(str(root / name))
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FPS) == 30
        count = 0
        samples = {}
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame.shape == (size[1], size[0], 3)
            assert float(frame[150:600].std()) > 10.
            if count in (0, 116, 150, 210, 240, 356):
                samples[count] = frame[150:600].copy()
            count += 1
        capture.release()
        assert count == 360
        assert not np.array_equal(samples[0], samples[116])
        if name == "cook_spinach_3d.mp4":
            assert not np.array_equal(samples[150], samples[210])
            assert not np.array_equal(samples[240], samples[356])
    print("PASS: orbit transforms, source records, 60 replay checks, 720 decoded video frames and time/view motion.")


if __name__ == "__main__":
    run_checks()
