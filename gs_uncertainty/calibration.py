"""Empirical one-sided bounds on reference-surface error, not XYZ covariance."""

import numpy as np


def group_balanced_quantile(values, groups, probability):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    if len(values) == 0 or len(values) != len(groups) or not np.isfinite(values).all():
        raise ValueError("Quantile inputs must be finite, nonempty and aligned.")
    if not 0 < probability <= 1:
        raise ValueError("Quantile probability must lie in (0,1].")
    names, group_indices, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1. / (len(names) * counts[group_indices])
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, probability, side="left")), len(values) - 1)
    return float(values[order[index]])


def fit_error_bounds(scale, error, groups, coverages=(0.5, 0.8, 0.9, 0.95)):
    """Each scene has equal total weight; samples within a scene remain dependent."""
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    if scale.shape != error.shape or not np.isfinite(scale).all() or not np.isfinite(error).all():
        raise ValueError("Finite aligned scales/errors are required; no silent filtering.")
    if np.any(scale <= 0) or np.any(error < 0):
        raise ValueError("Scales must be positive and unsigned errors nonnegative.")
    ratios = error / scale
    quantiles = []
    for coverage in coverages:
        quantiles.append(group_balanced_quantile(ratios, groups, coverage))
    return {"coverages": list(coverages), "multipliers": quantiles,
            "calibration_groups": np.unique(groups).tolist(),
            "interval": "[0, multiplier * scale], unsigned reference-surface error",
            "guarantee": "Empirical only; no exchangeability or conformal coverage guarantee."}


def evaluate_error_bounds(scale, error, groups, fitted):
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    if len(scale) == 0 or len(scale) != len(error) or len(scale) != len(groups):
        raise ValueError("Evaluation arrays must be nonempty and aligned.")
    if not np.isfinite(scale).all() or not np.isfinite(error).all() or np.any(scale <= 0) or np.any(error < 0):
        raise ValueError("Invalid scale/error; no silent sample removal.")
    rows = []
    for coverage, multiplier in zip(fitted["coverages"], fitted["multipliers"]):
        bounds = multiplier * scale
        per_group = []
        for group in np.unique(groups):
            selected = groups == group
            per_group.append({"group": str(group), "count": int(selected.sum()),
                "coverage": float(np.mean(error[selected] <= bounds[selected])),
                "mean_bound": float(bounds[selected].mean()),
                "mean_excess_error": float(np.maximum(error[selected] - bounds[selected], 0.).mean())})
        observed = float(np.mean([row["coverage"] for row in per_group]))
        rows.append({"nominal_coverage": coverage, "group_balanced_coverage": observed,
                     "absolute_coverage_error": abs(observed - coverage),
                     "group_balanced_mean_bound": float(np.mean([row["mean_bound"] for row in per_group])),
                     "per_group": per_group})
    return rows
