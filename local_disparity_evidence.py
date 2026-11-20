"""Explicit local disparity fitting; raw validity and spatial coordinates stay fixed."""

import numpy as np

from center_stereo_evidence import summarize_coordinate_residuals


WINDOWS = (5, 9)
GROUP_ROWS = {"reserved": [1, 6], "fit1": [0, 5], "fit3": [2, 7], "fit_all": [0, 2, 3, 4, 5, 7]}


def fit_local_disparity(pixels, disparity, valid, original_available, window):
    pixels = np.asarray(pixels, dtype=float)
    disparity = np.asarray(disparity, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    original_available = np.asarray(original_available, dtype=bool)
    if window not in WINDOWS:
        raise ValueError("Only the registered5/9 pixel windows are supported.")
    if pixels.ndim != 2 or pixels.shape[1] != 2 or original_available.shape != (len(pixels),):
        raise ValueError("Expected N by2 pixels and N original availability flags.")
    if disparity.ndim != 2 or valid.shape != disparity.shape:
        raise ValueError("Equal two-dimensional disparity/validity maps required.")
    height, width = disparity.shape
    selected = np.flatnonzero(original_available)
    uv = pixels[selected]
    if not np.isfinite(uv).all() or np.any(uv < 0) or np.any(uv[:, 0] > width-1) or np.any(uv[:, 1] > height-1):
        raise ValueError("Originally available coordinates must be finite and inside the image.")
    n = len(pixels)
    result = {"fitted_disparity_px": np.full(n, np.nan), "coefficients": np.full((n, 3), np.nan),
        "singular_values": np.full((n, 3), np.nan), "fit_squared_residual_px2": np.full(n, np.nan),
        "count": np.zeros(n, dtype=np.int16), "candidate_count": np.zeros(n, dtype=np.int16),
        "nonfinite_count": np.zeros(n, dtype=np.int16), "valid_nonfinite_count": np.zeros(n, dtype=np.int16),
        "rank": np.zeros(n, dtype=np.uint8), "reason": np.ones(n, dtype=np.uint8),
        "patch_columns": np.full(n, -1, dtype=np.int32), "patch_rows": np.full(n, -1, dtype=np.int32),
        "available": np.zeros(n, dtype=bool)}
    if not len(selected):
        return result
    origins, inverse = np.unique(np.floor(uv).astype(int), axis=0, return_inverse=True)
    radius = window//2
    # Centers sharing an integer patch have identical least-squares systems.
    patches = {name: np.empty((len(origins), *values.shape[1:]), dtype=values.dtype) for name, values in result.items()}
    for index, (column, row) in enumerate(origins):
        yy, xx = np.mgrid[max(0, row-radius):min(height, row+radius+1), max(0, column-radius):min(width, column+radius+1)]
        values = disparity[yy, xx]
        finite = np.isfinite(values)
        mask = valid[yy, xx] & finite
        count = int(mask.sum())
        patches["candidate_count"][index] = values.size
        patches["nonfinite_count"][index] = int((~finite).sum())
        patches["valid_nonfinite_count"][index] = int((valid[yy, xx] & ~finite).sum())
        patches["count"][index] = count
        patches["rank"][index] = 0
        patches["reason"][index] = 2
        patches["available"][index] = False
        patches["coefficients"][index] = np.nan
        patches["singular_values"][index] = np.nan
        patches["fit_squared_residual_px2"][index] = np.nan
        patches["fitted_disparity_px"][index] = np.nan
        patches["patch_columns"][index], patches["patch_rows"][index] = column, row
        if count < 3:
            continue
        design = np.column_stack((np.ones(count), xx[mask]-column, yy[mask]-row))
        observed = values[mask]
        coefficients, _, rank, singular = np.linalg.lstsq(design, observed, rcond=None)
        residual = observed-design@coefficients
        patches["coefficients"][index] = coefficients
        patches["singular_values"][index] = singular
        patches["fit_squared_residual_px2"][index] = np.sum(residual*residual)
        patches["rank"][index] = rank
        patches["reason"][index] = 0 if rank == 3 else 3
        patches["available"][index] = rank == 3
    for name in result:
        result[name][selected] = patches[name][inverse]
    available = result["available"]
    fraction = pixels[available]-np.column_stack((result["patch_columns"][available], result["patch_rows"][available]))
    coefficients = result["coefficients"][available]
    result["fitted_disparity_px"][available] = coefficients[:, 0]+coefficients[:, 1]*fraction[:, 0]+coefficients[:, 2]*fraction[:, 1]
    return result


def local_scores(arrays, window):
    prefix = f"local{window}"
    fitted = np.vstack((arrays[prefix+"_ref0_fitted_disparity_px"], arrays[prefix+"_ref4_fitted_disparity_px"]))
    available = np.vstack((arrays[prefix+"_ref0_available"], arrays[prefix+"_ref4_available"]))
    count = np.vstack((arrays[prefix+"_ref0_count"], arrays[prefix+"_ref4_count"]))
    squared = np.vstack((arrays[prefix+"_ref0_fit_squared_residual_px2"], arrays[prefix+"_ref4_fit_squared_residual_px2"]))
    predicted = np.vstack((arrays["ref0_predicted_disparity_px"], arrays["ref4_predicted_disparity_px"]))
    axial = np.vstack((arrays["ref0_axial_px_per_length"], arrays["ref4_axial_px_per_length"]))
    rms = np.full(count.shape, np.nan)
    rms[available] = np.sqrt(squared[available]/count[available])
    scores = {}
    for group, rows in GROUP_ROWS.items():
        values = summarize_coordinate_residuals((predicted-fitted)[rows], rms[rows], axial[rows], available[rows])
        scores[f"negative_{prefix}_{group}_depth"] = -values["depth_equivalent"]
        if group == "reserved":
            scores[prefix+"_reserved_pair_count"] = values["pair_count"].astype(float)
            scores["negative_"+prefix+"_reserved_fit_rms"] = -values["lr_rms"]
    return scores


def local_comparators(window):
    return ("negative_reserved_depth", f"negative_local{window}_fit1_depth", f"negative_local{window}_fit3_depth",
        "full_evidence", "support_ratio", "negative_spacing", "constant", "opacity")
