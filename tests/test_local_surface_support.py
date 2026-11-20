"""Analytic geometry, weighting, frame isolation and unit-equivariance checks."""

import numpy as np
from pathlib import Path
import tempfile
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_surface_support import neighbor_plane_origins
from surface_support import build_surface_support


def run_checks():
    xy = np.array([[0., 0.], [1., 0.], [0., 2.], [-3., 0.], [0., -4.]])
    points = np.column_stack((xy, np.zeros(5)))
    points = np.concatenate((points, points+[0., 0., 100.]))
    normals = np.tile([0., 0., 1.], (10, 1))
    frames = np.repeat([0, 2], 5)
    jacobian = np.tile([.2, .3, 1.], (10, 1))*np.arange(1, 11)[:, None]
    original = points.copy()
    for sensitivity in (None, jacobian):
        origins, offsets, ids, weights = neighbor_plane_origins(points, normals, frames, neighbors=4, ray_sensitivity=sensitivity)
        np.testing.assert_array_equal(origins, points)
        np.testing.assert_array_equal(offsets, np.zeros(10))
        np.testing.assert_array_equal(frames[ids], np.broadcast_to(frames[:, None], ids.shape))
        assert np.all(ids != np.arange(10)[:, None])
        np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-15, rtol=0)
    noisy = points.copy()
    noisy[0, 2] += .2
    noisy[5, 2] -= .4
    origins, offsets, ids, weights = neighbor_plane_origins(noisy, normals, frames, neighbors=4, ray_sensitivity=jacobian)
    np.testing.assert_allclose(origins[[0, 5]], points[[0, 5]], atol=1e-14, rtol=0)
    for row in range(len(points)):
        expected_weights = 1./jacobian[ids[row], 2]**2
        expected_weights /= expected_weights.sum()
        np.testing.assert_allclose(weights[row], expected_weights, atol=1e-15, rtol=0)
        expected_height = np.sum(expected_weights*noisy[ids[row], 2])
        np.testing.assert_allclose(origins[row, 2], expected_height, atol=2e-14, rtol=0)
    angle = .37
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    shift = np.array([.3, -.2, 4.])
    transformed = neighbor_plane_origins(noisy@rotation.T+shift, normals@rotation.T, frames, neighbors=4, ray_sensitivity=jacobian@rotation.T)
    np.testing.assert_allclose(transformed[0], origins@rotation.T+shift, atol=5e-14, rtol=0)
    np.testing.assert_array_equal(transformed[2], ids)
    scaled = neighbor_plane_origins(noisy*1000, normals, frames, neighbors=4, ray_sensitivity=jacobian*1000)
    np.testing.assert_allclose(scaled[0], origins*1000, atol=3e-11, rtol=0)
    np.testing.assert_allclose(scaled[3], weights, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(points, original)
    zero = jacobian.copy()
    zero[:, 2] = 0.
    try:
        neighbor_plane_origins(points, normals, frames, neighbors=4, ray_sensitivity=zero)
    except ValueError as error:
        assert "Zero projected sensitivity" in str(error)
    else:
        raise AssertionError("Unsupported zero sensitivity was accepted.")
    print("PASS: exact planes/offsets/weights, source exclusion, frame isolation, rigid/unit equivariance and immutable raw points.")


def run_import_checks():
    points = np.array([[0., 0., 1.], [.1, 0., 1.], [0., .1, 1.], [.1, .1, 1.]], dtype=np.float32)
    init = {"mu_ref": points, "rotations": np.tile([1., 0., 0., 0.], (4, 1)),
            "scales": np.tile([.1, .1, .008], (4, 1)), "frame_indices": np.array([0, 0, 2, 2]),
            "t_ref": np.array([0., 0., .1, .1])}
    original = {name: value.copy() for name, value in init.items()}
    with tempfile.TemporaryDirectory(prefix="derived_surface_") as folder:
        path = Path(folder)/"origins.npz"
        for device in ("cpu", "cuda"):
            default = build_surface_support(init, device)
            explicit = build_surface_support(init, device, None)
            for name in default:
                torch.testing.assert_close(default[name], explicit[name], atol=0, rtol=0)
            normals = default["normals"].cpu().numpy()
            values = {"origins": points+.02*normals, "raw_points": points.copy(),
                      "frames": init["frame_indices"].copy(), "normals": normals.copy()}
            np.savez(path, **values)
            derived = build_surface_support(init, device, path)
            np.testing.assert_array_equal(derived["points"].cpu().numpy(), values["origins"])
            torch.testing.assert_close(derived["raw_points"], default["points"], atol=0, rtol=0)
            for name in ("normals", "spacing", "frame_indices", "t_ref"):
                torch.testing.assert_close(derived[name], default[name], atol=0, rtol=0)
            for field in ("raw_points", "frames", "normals", "origins"):
                invalid = {name: value.copy() for name, value in values.items()}
                invalid[field].flat[0] += 1
                np.savez(path, **invalid)
                try:
                    build_surface_support(init, device, path)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"Invalid {field} was accepted.")
            invalid = {name: value.copy() for name, value in values.items()}
            invalid["origins"][0, 2] = np.nan
            np.savez(path, **invalid)
            try:
                build_surface_support(init, device, path)
            except ValueError as error:
                assert "finite" in str(error)
            else:
                raise AssertionError("NaN derived origin was accepted.")
            for name, value in original.items():
                np.testing.assert_array_equal(init[name], value)
            print(f"PASS {device}: default unchanged; normal-only support import; raw/frame/normal checks; immutable initialization.")


if __name__ == "__main__":
    run_checks()
    run_import_checks()
