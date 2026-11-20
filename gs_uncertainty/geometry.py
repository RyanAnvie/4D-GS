"""Conditional triangulation sensitivity; not calibrated measurement covariance."""

import numpy as np


def triangulation_sensitivity(points, cameras, normals=None):
    """Return (J.T J)^-1 in world-unit^2 / pixel^2 where triangulation is observable.

    This assumes fixed camera parameters and independent, equal-variance image
    coordinate measurements. No pixel variance is supplied or inferred here.
    Frustum inclusion is NOT an occlusion or correspondence-validity test.
    Invalid/degenerate estimates remain infinite and have an explicit valid mask.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite N by 3 world points.")
    information = np.zeros((len(points), 3, 3), dtype=np.float64)
    support = np.zeros(len(points), dtype=np.int32)
    for cam in cameras:
        w2c = cam.w2c.detach().cpu().numpy().astype(np.float64)
        R = w2c[:3, :3]
        xyz = points @ R.T + w2c[:3, 3]
        positive = xyz[:, 2] > 0
        z = np.where(positive, xyz[:, 2], 1.)
        u = cam.fx * xyz[:, 0] / z + cam.cx
        v = cam.fy * xyz[:, 1] / z + cam.cy
        inside = positive & (u >= 0) & (u <= cam.width - 1) & (v >= 0) & (v <= cam.height - 1)
        idx = np.flatnonzero(inside)
        jacobian = np.zeros((len(idx), 2, 3), dtype=np.float64)
        jacobian[:, 0, 0] = cam.fx / z[idx]
        jacobian[:, 1, 1] = cam.fy / z[idx]
        jacobian[:, 0, 2] = -cam.fx * xyz[idx, 0] / z[idx] ** 2
        jacobian[:, 1, 2] = -cam.fy * xyz[idx, 1] / z[idx] ** 2
        jacobian = jacobian @ R
        information[idx] += np.transpose(jacobian, (0, 2, 1)) @ jacobian
        support[idx] += 1
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    tolerance = 3 * np.finfo(np.float64).eps * np.maximum(eigenvalues[:, -1], 1.)
    valid = (support >= 2) & (eigenvalues[:, 0] > tolerance)
    covariance_per_pixel2 = np.full_like(information, np.inf)
    basis = eigenvectors[valid]
    inverse_values = 1. / eigenvalues[valid]
    covariance_per_pixel2[valid] = (basis * inverse_values[:, None, :]) @ np.transpose(basis, (0, 2, 1))
    trace_scale = np.full(len(points), np.inf)
    trace_scale[valid] = np.sqrt(np.trace(covariance_per_pixel2[valid], axis1=1, axis2=2))
    result = {"valid": valid, "frustum_support": support, "covariance_per_pixel2": covariance_per_pixel2,
              "position_scale_per_pixel": trace_scale}
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float64)
        if normals.shape != points.shape or not np.isfinite(normals).all():
            raise ValueError("Normals must be finite N by 3.")
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 0):
            raise ValueError("Normal-direction sensitivity requires nonzero normals.")
        unit_normals = normals / lengths[:, None]
        normal_scale = np.full(len(points), np.inf)
        projected = covariance_per_pixel2[valid] @ unit_normals[valid, :, None]
        normal_variance = (unit_normals[valid] * projected[:, :, 0]).sum(axis=1)
        normal_scale[valid] = np.sqrt(np.maximum(normal_variance, 0.))
        result["normal_scale_per_pixel"] = normal_scale
    return result
