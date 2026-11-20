"""Experimental local-plane origins; raw measurements and normals are retained."""

import numpy as np
from scipy.spatial import cKDTree


def neighbor_plane_origins(points, normals, frame_indices, neighbors=16, ray_sensitivity=None):
    """Return normal-only offsets, with other same-frame sources as support.

    ray_sensitivity, when given, is dXYZ/d(disparity pixel), not measured sigma.
    Inverse projected-sensitivity weights assume equal pixel error scale and
    fixed camera/normal estimates; they are not calibrated precision weights.
    """
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    frames = np.asarray(frame_indices)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape or frames.shape != (len(points),):
        raise ValueError("Expected points/normals [N,3] and frame indices [N].")
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError("Nonfinite measurements must not be silently filtered.")
    if not np.allclose(np.linalg.norm(normals, axis=1), 1., rtol=0, atol=1e-5):
        raise ValueError("Unit normals are required.")
    if not isinstance(neighbors, int) or neighbors < 1:
        raise ValueError("neighbors must be a positive integer.")
    if ray_sensitivity is not None:
        ray_sensitivity = np.asarray(ray_sensitivity, dtype=np.float64)
        if ray_sensitivity.shape != points.shape or not np.isfinite(ray_sensitivity).all():
            raise ValueError("Finite ray sensitivity [N,3] is required.")
    neighbor_ids = np.empty((len(points), neighbors), dtype=np.int64)
    weights = np.empty((len(points), neighbors), dtype=np.float64)
    offsets = np.empty(len(points), dtype=np.float64)
    for frame in np.unique(frames):
        selected = np.flatnonzero(frames == frame)
        if len(selected) <= neighbors:
            raise ValueError(f"Frame {frame} needs at least {neighbors+1} sources.")
        _, local_ids = cKDTree(points[selected]).query(points[selected], k=neighbors+1)
        for row in range(len(selected)):
            other = local_ids[row][local_ids[row] != row][:neighbors]
            neighbor_ids[selected[row]] = selected[other]
        ids = neighbor_ids[selected]
        relative = points[ids]-points[selected, None, :]
        heights = np.sum(relative*normals[selected, None, :], axis=2)
        if ray_sensitivity is None:
            local_weights = np.ones(heights.shape, dtype=np.float64)
        else:
            projected = np.sum(ray_sensitivity[ids]*normals[selected, None, :], axis=2)
            if np.any(projected == 0):
                raise ValueError("Zero projected sensitivity cannot define these experimental weights.")
            local_weights = 1./projected**2
            if not np.isfinite(local_weights).all():
                raise ValueError("Nonfinite local-plane weights.")
        local_weights = local_weights/local_weights.sum(axis=1, keepdims=True)
        weights[selected] = local_weights
        offsets[selected] = np.sum(local_weights*heights, axis=1)
    origins = points+offsets[:, None]*normals
    return origins, offsets, neighbor_ids, weights
