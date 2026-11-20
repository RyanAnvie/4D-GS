"""OpenCV rectification/stereo and explicit world-frame depth fusion."""

import cv2
import numpy as np


def reconstruct_calibrated_pair(image_a, image_b, pose_a, pose_b, pixels_a, pixels_b,
                                max_disparities=512, disparity_margin=16, rectification_alpha=-1):
    """Return rectified disparity diagnostics and measured world points, no GT."""
    # Different undistorted crops keep their own intrinsics and valid domains.
    # A common output canvas does not resize or stretch either source image.
    height = max(image_a.shape[0], image_b.shape[0])
    width = max(image_a.shape[1], image_b.shape[1])
    relative_R = pose_b["R"] @ pose_a["R"].T
    relative_t = pose_b["t"] - relative_R @ pose_a["t"]
    if np.linalg.norm(relative_t) <= 1e-10:
        raise ValueError("Coincident camera centers cannot supply stereo depth.")
    R_a, R_b, P_a, P_b, Q, _, _ = cv2.stereoRectify(
        pose_a["K"], None, pose_b["K"], None, (width, height), relative_R, relative_t,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=rectification_alpha)
    if not all(np.isfinite(value).all() for value in (R_a, R_b, P_a, P_b, Q)) or P_a[0, 0] <= 0:
        return {"status": "invalid_rectification"}
    axis = 1 if abs(P_b[1, 3]) > abs(P_b[0, 3]) else 0
    sparse_a = cv2.undistortPoints(pixels_a.reshape(-1, 1, 2), pose_a["K"], None, R=R_a, P=P_a).reshape(-1, 2)
    sparse_b = cv2.undistortPoints(pixels_b.reshape(-1, 1, 2), pose_b["K"], None, R=R_b, P=P_b).reshape(-1, 2)
    sparse_disparity = sparse_a[:, axis] - sparse_b[:, axis]
    if not np.isfinite(sparse_disparity).all() or not len(sparse_disparity):
        raise ValueError("Finite accepted feature disparities are required.")
    length = width if axis == 0 else height
    lower = max(-(length - 1), int(np.floor(sparse_disparity.min())) - disparity_margin)
    upper = min(length - 1, int(np.ceil(sparse_disparity.max())) + disparity_margin)
    count = max(16, int(np.ceil((upper - lower + 1) / 16)) * 16)
    metadata = {"axis": axis, "min_disparity": lower, "num_disparities": count,
                "rectified_width": width, "rectified_height": height, "alpha": rectification_alpha,
                "sparse_disparity_min": float(sparse_disparity.min()),
                "sparse_disparity_max": float(sparse_disparity.max())}
    if count > max_disparities:
        return {"status": "range_exceeds_declared_budget", **metadata}
    maps = []
    rectified = []
    masks = []
    for image, pose, R, P in ((image_a, pose_a, R_a, P_a), (image_b, pose_b, R_b, P_b)):
        map_x, map_y = cv2.initUndistortRectifyMap(pose["K"], None, R, P, (width, height), cv2.CV_32FC1)
        maps.append((map_x, map_y))
        rectified.append(cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
        masks.append((map_x >= 0) & (map_x < image.shape[1] - 1) & (map_y >= 0) & (map_y < image.shape[0] - 1))
    gray_a = cv2.cvtColor(rectified[0], cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(rectified[1], cv2.COLOR_BGR2GRAY)
    if axis == 1:
        gray_a, gray_b = gray_a.T.copy(), gray_b.T.copy()
        mask_a, mask_b = masks[0].T, masks[1].T
    else:
        mask_a, mask_b = masks
    padding = max(abs(lower), abs(lower + count)) + 8
    padded_a = cv2.copyMakeBorder(gray_a, 0, 0, padding, padding, cv2.BORDER_REPLICATE)
    padded_b = cv2.copyMakeBorder(gray_b, 0, 0, padding, padding, cv2.BORDER_REPLICATE)
    parameters = dict(numDisparities=count, blockSize=5, P1=8 * 25, P2=32 * 25,
                      uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                      disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    reverse_lower = -lower - count + 1
    forward = cv2.StereoSGBM_create(minDisparity=lower, **parameters)
    backward = cv2.StereoSGBM_create(minDisparity=reverse_lower, **parameters)
    disparity = forward.compute(padded_a, padded_b).astype(np.float32)[:, padding:-padding] / 16.
    reverse = backward.compute(padded_b, padded_a).astype(np.float32)[:, padding:-padding] / 16.
    rows, columns = np.indices(disparity.shape)
    matched_columns = np.rint(columns - disparity).astype(np.int32)
    inside = (matched_columns >= 0) & (matched_columns < gray_a.shape[1])
    clipped = np.clip(matched_columns, 0, gray_a.shape[1] - 1)
    reverse_match = reverse[rows, clipped]
    lr_error = np.abs(disparity + reverse_match)
    valid = inside & mask_a & mask_b[rows, clipped]
    valid &= (disparity > lower - 1) & (reverse_match > reverse_lower - 1) & (lr_error <= 1.)
    if axis == 1:
        disparity, lr_error, valid = disparity.T, lr_error.T, valid.T
    rows, columns = np.nonzero(valid)
    homogeneous = np.column_stack((columns, rows, disparity[valid], np.ones(len(rows)))) @ Q.T
    nonzero = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 3]) > 1e-12)
    rect_points = np.full((len(rows), 3), np.nan)
    rect_points[nonzero] = homogeneous[nonzero, :3] / homogeneous[nonzero, 3:4]
    camera_a = rect_points @ R_a
    world = (camera_a - pose_a["t"]) @ pose_a["R"]
    camera_b = world @ pose_b["R"].T + pose_b["t"]
    positive = nonzero & (rect_points[:, 2] > 0) & (camera_a[:, 2] > 0) & (camera_b[:, 2] > 0)
    valid[rows[~positive], columns[~positive]] = False
    return {"status": "ok", **metadata, "disparity": disparity, "lr_error_px": lr_error,
        "valid": valid, "points": world[positive], "colors": rectified[0][valid][:, ::-1] / 255.,
        "R_a": R_a, "R_b": R_b, "P_a": P_a, "P_b": P_b, "Q": Q,
        "rectified_valid_a": masks[0], "rectified_valid_b": masks[1]}


def world_points_to_depth(points, pose):
    """Nearest projected-pixel z-buffer; absent pixels are explicit NaN."""
    camera = points @ pose["R"].T + pose["t"]
    projection = camera @ pose["K"].T
    valid = np.isfinite(projection).all(axis=1) & (camera[:, 2] > 0)
    pixels = np.full((len(points), 2), -1, dtype=np.int64)
    pixels[valid] = np.rint(projection[valid, :2] / projection[valid, 2:3]).astype(np.int64)
    valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < pose["width"])
    valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] < pose["height"])
    depth = np.full(pose["height"] * pose["width"], np.inf, dtype=np.float64)
    index = pixels[valid, 1] * pose["width"] + pixels[valid, 0]
    np.minimum.at(depth, index, camera[valid, 2])
    depth[np.isinf(depth)] = np.nan
    return depth.reshape(pose["height"], pose["width"]).astype(np.float32)


def fuse_view_depths(depths, relative_tolerance=.01, minimum_support=2):
    """Median depth with explicitly recorded pair support, not calibrated sigma."""
    depths = np.asarray(depths)
    if depths.ndim != 3 or len(depths) == 0:
        raise ValueError("Depth stack must be partners by height by width.")
    finite = np.isfinite(depths)
    if np.any(finite & (depths <= 0)) or np.isinf(depths).any():
        raise ValueError("Depths must be positive or explicit missing NaN.")
    available = finite.sum(axis=0)
    median = np.full(depths.shape[1:], np.nan, dtype=np.float32)
    has_measurement = available > 0
    median[has_measurement] = np.nanmedian(depths[:, has_measurement], axis=0)
    consistent = finite & (np.abs(depths - median[None]) <= relative_tolerance * median[None])
    support = consistent.sum(axis=0)
    accepted = support >= minimum_support
    fused = np.full_like(median, np.nan)
    selected = np.where(consistent[:, accepted], depths[:, accepted], np.nan)
    fused[accepted] = np.nanmedian(selected, axis=0)
    # Spread is descriptive pair disagreement; shared-image bias is not independent noise.
    spread = np.full_like(median, np.nan)
    spread[accepted] = np.nanmax(selected, axis=0) - np.nanmin(selected, axis=0)
    return fused, available.astype(np.uint8), support.astype(np.uint8), spread
