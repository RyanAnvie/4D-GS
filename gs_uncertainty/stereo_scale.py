"""Image-derived stereo error scales; not calibrated positional covariance."""

import cv2
import numpy as np


def stereo_matching_scale(left, right, disparity, input_valid, window=5):
    """Gray intensities are in 8-bit levels; disparity and gradient use pixels."""
    if left.ndim != 2 or left.shape != right.shape or disparity.shape != left.shape:
        raise ValueError("Aligned rectified gray images and disparity are required.")
    if input_valid.shape != left.shape or window < 1 or window % 2 != 1:
        raise ValueError("Aligned validity mask and an odd positive window are required.")
    if not np.isfinite(disparity).all():
        raise ValueError("Disparity must be finite; invalid measurements use input_valid.")
    rows, columns = np.indices(left.shape, dtype=np.float32)
    right_columns = columns - disparity.astype(np.float32)
    right_float = right.astype(np.float32)
    # Central finite difference, with intensity/pixel units; no implicit Sobel gain.
    derivative = cv2.Sobel(right_float, cv2.CV_32F, 1, 0, ksize=1, scale=0.5)
    matched = cv2.remap(right_float, right_columns, rows, cv2.INTER_LINEAR)
    matched_derivative = cv2.remap(derivative, right_columns, rows, cv2.INTER_LINEAR)
    valid = input_valid & (right_columns >= 1) & (right_columns <= left.shape[1] - 2)
    residual = left.astype(np.float32) - matched
    weights = valid.astype(np.float64)
    # Independent unit-step uniform quantizers: Var(Q_left-Q_right) = 2/12.
    residual_energy = (residual.astype(np.float64) ** 2 + 1. / 6.) * weights
    gradient_energy = matched_derivative.astype(np.float64) ** 2 * weights
    size = (window, window)
    count = cv2.boxFilter(weights, -1, size, normalize=False, borderType=cv2.BORDER_CONSTANT)
    numerator = cv2.boxFilter(residual_energy, -1, size, normalize=False, borderType=cv2.BORDER_CONSTANT)
    denominator = cv2.boxFilter(gradient_energy, -1, size, normalize=False, borderType=cv2.BORDER_CONSTANT)
    observable = valid & (count > 0) & (denominator > 0)
    scale = np.full(left.shape, np.inf)
    scale[observable] = np.sqrt(numerator[observable] / denominator[observable])
    return {"disparity_scale_px": scale, "observable": observable,
            "patch_count": count, "residual_dn": residual, "gradient_dn_per_px": matched_derivative}


def disparity_position_sensitivity(points, K, baseline):
    """Norm of dX/dd on a fixed rectified left-camera ray, world units/pixel."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Finite N by 3 left-camera points required.")
    if baseline <= 0 or K[0, 0] <= 0 or np.any(points[:, 2] <= 0):
        raise ValueError("Positive baseline, focal length and depth required.")
    disparity_plus_offset = K[0, 0] * baseline / points[:, 2]
    return np.linalg.norm(points, axis=1) / disparity_plus_offset
