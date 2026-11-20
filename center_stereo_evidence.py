"""Fixed horizontal-rig coordinate discrepancies, not calibrated XYZ uncertainty."""

import numpy as np
from scipy.stats import spearmanr


def query_stereo_coordinates(points, intrinsic, center, baselines, disparity, valid, lr_error):
    """Query one reference's pair fields. XYZ/center/baselines share length units."""
    points = np.asarray(points, dtype=float)
    K = np.asarray(intrinsic, dtype=float)
    center = np.asarray(center, dtype=float)
    baselines = np.asarray(baselines, dtype=float)
    disparity = np.asarray(disparity, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    lr_error = np.asarray(lr_error, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or center.shape != (3,):
        raise ValueError("Expected N by3 points and a3-vector camera center.")
    if K.shape != (3, 3) or not np.isfinite(K).all() or not np.isfinite(center).all():
        raise ValueError("Finite pinhole calibration and center required.")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or K[0, 1] != 0 or K[1, 0] != 0 or not np.array_equal(K[2], [0, 0, 1]):
        raise ValueError("Positive focal lengths and zero-skew pinhole calibration required.")
    if disparity.ndim != 3 or valid.shape != disparity.shape or lr_error.shape != disparity.shape:
        raise ValueError("Pair fields must have equal [pairs,height,width] shapes.")
    pairs, height, width = disparity.shape
    if baselines.shape != (pairs,) or not np.isfinite(baselines).all() or np.any(baselines == 0) or min(height, width) < 1:
        raise ValueError("Finite nonzero signed horizontal baselines required.")
    local = points - center.reshape(1, 3)
    depth_valid = np.isfinite(local).all(axis=1) & (local[:, 2] > 0)
    pixels = np.full((len(points), 2), np.nan)
    pixels[depth_valid, 0] = K[0, 0] * local[depth_valid, 0] / local[depth_valid, 2] + K[0, 2]
    pixels[depth_valid, 1] = K[1, 1] * local[depth_valid, 1] / local[depth_valid, 2] + K[1, 2]
    inside = depth_valid & np.isfinite(pixels).all(axis=1)
    inside &= (pixels[:, 0] >= 0) & (pixels[:, 0] <= width - 1)
    inside &= (pixels[:, 1] >= 0) & (pixels[:, 1] <= height - 1)
    columns = np.zeros((len(points), 4), dtype=np.int64)
    rows = np.zeros_like(columns)
    weights = np.zeros((len(points), 4))
    x, y = pixels[inside, 0], pixels[inside, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = x - x0, y - y0
    columns[inside] = np.column_stack((x0, x1, x0, x1))
    rows[inside] = np.column_stack((y0, y0, y1, y1))
    weights[inside] = np.column_stack(((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy))
    available = np.broadcast_to(inside, (pairs, len(points))).copy()
    sampled_disparity = np.zeros(available.shape)
    sampled_lr = np.zeros(available.shape)
    for corner in range(4):
        d = disparity[:, rows[:, corner], columns[:, corner]]
        lr = lr_error[:, rows[:, corner], columns[:, corner]]
        finite_valid = valid[:, rows[:, corner], columns[:, corner]] & np.isfinite(d) & np.isfinite(lr)
        required = weights[:, corner] > 0
        available &= ~required[None, :] | finite_valid
        # Invalid contributions stay unavailable, never renormalized to surviving corners.
        sampled_disparity += np.where(finite_valid, d, 0.) * weights[None, :, corner]
        sampled_lr += np.where(finite_valid, lr, 0.) * weights[None, :, corner]
    reason = np.zeros(available.shape, dtype=np.uint8)
    reason[:, inside] = np.where(available[:, inside], 0, 3)
    reason[:, depth_valid & ~inside] = 2
    reason[:, ~depth_valid] = 1
    sampled_disparity[~available] = np.nan
    sampled_lr[~available] = np.nan
    predicted = np.full(available.shape, np.nan)
    axial = np.full(available.shape, np.nan)
    predicted[:, depth_valid] = K[0, 0] * baselines[:, None] / local[None, depth_valid, 2]
    axial[:, depth_valid] = K[0, 0] * baselines[:, None] / local[None, depth_valid, 2]**2
    return {"pixels": pixels, "corner_rows": rows, "corner_columns": columns, "weights": weights,
        "available": available, "reason": reason, "predicted_disparity_px": predicted,
        "sampled_disparity_px": sampled_disparity, "lr_error_px": sampled_lr,
        "residual_px": predicted - sampled_disparity, "axial_px_per_length": axial}


def summarize_coordinate_residuals(residual, lr_error, axial, available):
    """Combine pair rows without claiming independent observations or a covariance."""
    if not (residual.shape == lr_error.shape == axial.shape == available.shape):
        raise ValueError("All pair arrays must have the same shape.")
    for values in (residual, lr_error, axial):
        if not np.isfinite(values[available]).all():
            raise ValueError("An available measurement must be finite.")
    count = available.sum(axis=0)
    scored = count > 0
    squared_error = np.sum(np.where(available, residual**2, 0.), axis=0)
    squared_lr = np.sum(np.where(available, lr_error**2, 0.), axis=0)
    sensitivity = np.sum(np.where(available, axial**2, 0.), axis=0)
    if np.any(sensitivity[scored] <= 0):
        raise ValueError("Available pairs require nonzero axial sensitivity.")
    rms, lr_rms, equivalent = np.full(len(count), np.nan), np.full(len(count), np.nan), np.full(len(count), np.nan)
    rms[scored] = np.sqrt(squared_error[scored] / count[scored])
    lr_rms[scored] = np.sqrt(squared_lr[scored] / count[scored])
    equivalent[scored] = np.sqrt(squared_error[scored] / sensitivity[scored])
    return {"pair_count": count, "pixel_rms": rms, "lr_rms": lr_rms, "depth_equivalent": equivalent}


def exact_risk_statistics(confidence, error):
    """Exact ranks and expected selective risk under uniform boundary-tie selection."""
    confidence = np.asarray(confidence, dtype=float)
    error = np.asarray(error, dtype=float)
    if confidence.ndim != 1 or error.shape != confidence.shape or not np.isfinite(confidence).all() or not np.isfinite(error).all():
        raise ValueError("Equal finite score/error vectors required; missingness must be explicit outside this function.")
    correlation = None
    if len(error) > 1 and np.ptp(confidence) > 0 and np.ptp(error) > 0:
        correlation = float(spearmanr(confidence, -error).statistic)
    risks = []
    ordered = np.sort(confidence)[::-1]
    for coverage in np.linspace(.1, 1., 10):
        if not len(error):
            break
        count = max(1, int(round(len(error) * coverage)))
        cutoff = ordered[count - 1]
        above = confidence > cutoff
        tied = confidence == cutoff
        remainder = count - int(above.sum())
        expected_sum = error[above].sum() + remainder * error[tied].mean()
        risks.append({"requested_coverage": float(coverage), "retained_count": count,
            "actual_coverage": count / len(error), "expected_mae": float(expected_sum / count),
            "strictly_above": int(above.sum()), "boundary_ties": int(tied.sum()), "selected_from_ties": remainder})
    return {"count": len(error), "spearman": correlation, "expected_risk": risks,
        "mean_expected_risk": float(np.mean([r["expected_mae"] for r in risks])) if risks else None}
