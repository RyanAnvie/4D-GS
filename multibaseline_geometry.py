"""Rectified same-intrinsic stereo fusion; scales are not calibrated covariances."""

import numpy as np


def fuse_supported_inverse_depth(disparities, valid, baselines, focal):
    """Fuse only explicitly valid pairs; zero-support depth remains NaN."""
    disparities = np.asarray(disparities, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    baselines = np.asarray(baselines, dtype=np.float64)
    if disparities.ndim != 2 or valid.shape != disparities.shape or len(baselines) != disparities.shape[0]:
        raise ValueError("Expected disparities/mask [pairs,pixels] and baselines [pairs].")
    if not np.isfinite(focal) or focal <= 0 or not np.isfinite(baselines).all() or np.any(baselines <= 0):
        raise ValueError("Finite positive focal length and baselines are required.")
    if not np.isfinite(disparities[valid]).all() or np.any(disparities[valid] <= 0):
        raise ValueError("A valid stereo observation must have finite positive disparity.")
    observed = np.where(valid, disparities, 0.)
    baseline_square_sum = np.sum(valid*baselines[:, None]**2, axis=0)
    numerator = np.sum(baselines[:, None]*observed, axis=0)
    pair_count = np.sum(valid, axis=0)
    supported = pair_count > 0
    depth = np.full(disparities.shape[1], np.nan, dtype=np.float64)
    inverse_depth = numerator[supported]/(focal*baseline_square_sum[supported])
    depth[supported] = 1./inverse_depth
    return depth, pair_count, baseline_square_sum


def fuse_inverse_depth(disparities, baselines, focal, rays):
    """Inputs are positive disparities [views,N], baselines [views] in meters."""
    disparities = np.asarray(disparities, dtype=np.float64)
    baselines = np.asarray(baselines, dtype=np.float64)
    rays = np.asarray(rays, dtype=np.float64)
    if disparities.ndim != 2 or disparities.shape[0] != len(baselines):
        raise ValueError("Disparities must be views by points.")
    if rays.shape != (disparities.shape[1], 3):
        raise ValueError("Rays must be points by three.")
    if not np.isfinite(disparities).all() or np.any(disparities <= 0):
        raise ValueError("Only explicit common-valid, positive disparities are supported.")
    if not np.isfinite(baselines).all() or np.any(baselines <= 0) or focal <= 0:
        raise ValueError("Positive finite baselines and focal length are required.")
    inverse_depths = disparities / (focal * baselines[:, None])
    denominator = focal * np.sum(baselines ** 2)
    least_squares = np.sum(baselines[:, None] * disparities, axis=0) / denominator
    median = np.median(inverse_depths, axis=0)
    residual = inverse_depths - least_squares[None, :]
    disagreement = np.sqrt(np.mean(residual ** 2, axis=0))
    # Numerical disparity resolution, not a measured sensor noise standard deviation.
    quantization_floor = (1. / 16.) / (focal * np.sqrt(np.sum(baselines ** 2)))
    ray_norm = np.linalg.norm(rays, axis=1)
    sensitivity = ray_norm / least_squares ** 2
    scales = {
        "camera_quantization": sensitivity * quantization_floor,
        "disagreement": sensitivity * np.sqrt(disagreement ** 2 + quantization_floor ** 2),
    }
    return {"least_squares": 1. / least_squares, "median": 1. / median}, scales
