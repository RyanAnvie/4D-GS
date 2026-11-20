from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from multibaseline_geometry import fuse_inverse_depth
from prepare_eth3d_geometry import reconstruct_rectified_pair


def test_fusion():
    baselines = np.array([0.06, 0.12, 0.18, 0.24])
    rays = np.array([[0., 0., 1.], [0.2, -0.1, 1.]])
    depth = np.array([1.3, 2.])
    disparity = 400. * baselines[:, None] / depth[None, :]
    estimated, scale = fuse_inverse_depth(disparity, baselines, 400., rays)
    for value in estimated.values():
        np.testing.assert_allclose(value, depth, atol=1e-12)
    np.testing.assert_allclose(scale["disagreement"], scale["camera_quantization"])
    biased = disparity + 0.4
    changed, biased_scale = fuse_inverse_depth(biased, baselines, 400., rays)
    assert np.all(changed["least_squares"] < depth)
    assert np.all(biased_scale["disagreement"] > biased_scale["camera_quantization"])
    millimeters, scales_mm = fuse_inverse_depth(biased, baselines * 1000., 400., rays)
    for name in changed:
        np.testing.assert_allclose(millimeters[name], changed[name] * 1000.)
    for name in biased_scale:
        np.testing.assert_allclose(scales_mm[name], biased_scale[name] * 1000.)
    # A shared inverse-depth bias is invisible to cross-baseline disagreement.
    shared_bias = disparity + 400. * baselines[:, None] * 0.02
    _, shared_scale = fuse_inverse_depth(shared_bias, baselines, 400., rays)
    np.testing.assert_allclose(shared_scale["disagreement"], shared_scale["camera_quantization"])
    print("PASS: exact depth, signed bias, metric unit equivariance, shared-bias limitation.")


def test_padding_coordinates():
    rng = np.random.default_rng(42)
    left = rng.integers(0, 256, (96, 256, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-8] = left[:, 8:]
    calibration = {"cam0": np.array([[200., 0., 128.], [0., 200., 48.], [0., 0., 1.]]),
                   "baseline_m": 0.12, "doffs": 0.}
    outputs = []
    for padding in (0, 64):
        points, colors, disparity, valid, _ = reconstruct_rectified_pair(
            left, right, calibration, num_disparities=64, horizontal_padding=padding)
        assert disparity.shape == (96, 256)
        rows, columns = np.nonzero(valid)
        np.testing.assert_allclose(np.median(disparity[valid]), 8., atol=1e-6)
        np.testing.assert_allclose(np.median(points[:, 2]), 3., atol=1e-6)
        np.testing.assert_allclose(points[:, 0] / points[:, 2], (columns - 127.5) / 200., atol=1e-7)
        np.testing.assert_allclose(points[:, 1] / points[:, 2], (rows - 47.5) / 200., atol=1e-7)
        np.testing.assert_allclose(colors, left[valid][:, ::-1] / 255., atol=1e-7)
        assert np.all(np.rint(columns - disparity[valid]) >= 0)
        assert np.all(np.rint(columns - disparity[valid]) < 256)
        outputs.append(int(valid.sum()))
    assert outputs[1] > outputs[0]
    print(f"PASS: padded stereo original coordinates/colors/depth/validity; support {outputs}.")


if __name__ == "__main__":
    test_fusion()
    test_padding_coordinates()
