"""Fixed-support RGB offset refinement of saved signed disparity planes."""

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import elementwise

from center_stereo_evidence import summarize_coordinate_residuals
from local_disparity_evidence import GROUP_ROWS


OFFSETS = np.linspace(-1., 1., 33)
XATOL = 1e-5
SOLVER_XATOL = 1e-6  # Library termination permits a wider bracket; acceptance stays1e-5.
MAXITER = 100
BATCH_SIZE = 1024
PRIMARY = "negative_photo_reserved_depth"
COMPARATORS = ("negative_reserved_depth", "negative_local5_reserved_depth",
    "negative_photo_fit1_depth", "negative_photo_fit3_depth", "full_evidence",
    "support_ratio", "negative_spacing", "constant", "opacity")
# Query reasons1/2 are assigned before entering this patch-only estimator.
REASONS = {0: "available", 1: "original_unavailable", 2: "local_plane_unavailable",
    3: "no_fixed_support", 4: "boundary_minimum", 5: "flat_or_invalid_bracket",
    6: "solver_failed", 7: "solver_width_unresolved"}


def fixed_patch_samples(reference, partner, disparity, valid, origins, coefficients):
    """Pixels are integer centers; target x = reference x - signed disparity."""
    height, width, channels = reference.shape
    if channels != 3 or partner.shape != reference.shape or disparity.shape != (height, width):
        raise ValueError("Equal RGB images and matching disparity map required.")
    if valid.shape != disparity.shape or coefficients.shape != (len(origins), 3):
        raise ValueError("Patch coefficients/validity dimensions do not match.")
    if not np.isfinite(coefficients).all():
        raise ValueError("Only finite, rank3 local planes may be refined.")
    dy, dx = np.mgrid[-2:3, -2:3]
    x = origins[:, :1]+dx.ravel()
    y = origins[:, 1:2]+dy.ravel()
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    safe_x, safe_y = np.clip(x, 0, width-1), np.clip(y, 0, height-1)
    measured = disparity[safe_y, safe_x]
    raw = inside & valid[safe_y, safe_x] & np.isfinite(measured)
    plane = coefficients[:, :1]+coefficients[:, 1:2]*dx.ravel()+coefficients[:, 2:3]*dy.ravel()
    target_x = x-plane
    fixed = raw & (target_x-1. >= 0.) & (target_x+1. <= width-1.)
    rgb = reference[safe_y, safe_x]
    return {"x": x, "y": y, "target_x": target_x, "rgb": rgb, "fixed": fixed,
        "candidate_count": inside.sum(axis=1), "raw_count": raw.sum(axis=1),
        "count": fixed.sum(axis=1), "border_excluded_count": (raw & ~fixed).sum(axis=1)}


def image_cost(offset, indices, samples, partner):
    indices = np.asarray(indices, dtype=int)
    offset = np.broadcast_to(offset, indices.shape)
    fixed = samples["fixed"][indices]
    x = samples["target_x"][indices]-offset[:, None]
    y = samples["y"][indices].astype(float)
    height, width = partner.shape[:2]
    if np.any(x[fixed] < 0.) or np.any(x[fixed] > width-1.) or np.any(y[fixed] < 0.) or np.any(y[fixed] > height-1.):
        raise ValueError("Photometric support moved outside the original image.")
    # Masked slots are computational padding, never additional observations.
    x = np.where(fixed, x, 0.)
    y = np.where(fixed, y, 0.)
    squared = np.zeros(fixed.shape)
    for channel in range(3):
        target = map_coordinates(partner[:, :, channel], [y, x], order=1, prefilter=False, mode="constant", cval=np.nan)
        difference = samples["rgb"][indices, :, channel]-target
        squared += difference*difference
    squared[~fixed] = 0.
    count = samples["count"][indices]
    cost = np.full(len(indices), np.nan)
    observed = count > 0
    cost[observed] = squared[observed].sum(axis=1)/(3*count[observed])
    return cost


def refine_patches(reference, partner, disparity, valid, origins, coefficients, batch_size=BATCH_SIZE):
    n = len(origins)
    result = {"origins": origins.copy(), "coefficients": coefficients.copy(),
        "coarse_cost": np.full((n, 33), np.nan), "coarse_index": np.full(n, -1, dtype=np.int16),
        "initial_cost": np.full(n, np.nan), "final_cost": np.full(n, np.nan),
        "candidate_offset_px": np.full(n, np.nan), "offset_px": np.full(n, np.nan),
        "initial_bracket": np.full((n, 3), np.nan), "final_bracket": np.full((n, 3), np.nan),
        "final_bracket_cost": np.full((n, 3), np.nan), "solver_status": np.full(n, -99, dtype=np.int16),
        "nit": np.zeros(n, dtype=np.int16), "nfev": np.zeros(n, dtype=np.int16),
        "reason": np.full(n, 3, dtype=np.uint8), "available": np.zeros(n, dtype=bool)}
    for name in ("candidate_count", "raw_count", "count", "border_excluded_count"):
        result[name] = np.zeros(n, dtype=np.int16)
    priority = np.lexsort((OFFSETS, np.abs(OFFSETS)))
    for start in range(0, n, batch_size):
        stop = min(n, start+batch_size)
        samples = fixed_patch_samples(reference, partner, disparity, valid, origins[start:stop], coefficients[start:stop])
        for name in ("candidate_count", "raw_count", "count", "border_excluded_count"):
            result[name][start:stop] = samples[name]
        ids = np.flatnonzero(samples["count"] > 0)
        if not len(ids):
            continue
        coarse = np.column_stack([image_cost(delta, ids, samples, partner) for delta in OFFSETS])
        assert np.isfinite(coarse).all()
        selected = priority[np.argmin(coarse[:, priority], axis=1)]
        output_ids = start+ids
        result["coarse_cost"][output_ids] = coarse
        result["coarse_index"][output_ids] = selected
        result["initial_cost"][output_ids] = coarse[:, 16]
        result["candidate_offset_px"][output_ids] = OFFSETS[selected]
        result["final_cost"][output_ids] = coarse[np.arange(len(ids)), selected]
        boundary = (selected == 0) | (selected == 32)
        result["reason"][output_ids[boundary]] = 4
        interior_ids = ids[~boundary]
        chosen = selected[~boundary]
        interior_costs = coarse[~boundary]
        rows = np.arange(len(chosen))
        low, mid, high = interior_costs[rows, chosen-1], interior_costs[rows, chosen], interior_costs[rows, chosen+1]
        bracketed = (mid <= low) & (mid <= high) & ((mid < low) | (mid < high))
        result["reason"][start+interior_ids] = 5
        solve_ids = interior_ids[bracketed]
        chosen = chosen[bracketed]
        if not len(solve_ids):
            continue
        initial = (OFFSETS[chosen-1], OFFSETS[chosen], OFFSETS[chosen+1])

        def cost(delta, patch_ids):
            return image_cost(delta, patch_ids, samples, partner)

        solved = elementwise.find_minimum(cost, initial, args=(solve_ids,), maxiter=MAXITER,
            tolerances={"xatol": SOLVER_XATOL, "xrtol": 0., "fatol": 0., "frtol": 0.})
        output_ids = start+solve_ids
        bracket = np.column_stack(solved.bracket)
        result["initial_bracket"][output_ids] = np.column_stack(initial)
        result["final_bracket"][output_ids] = bracket
        result["final_bracket_cost"][output_ids] = np.column_stack(solved.f_bracket)
        result["candidate_offset_px"][output_ids] = solved.x
        result["final_cost"][output_ids] = solved.f_x
        for name in ("nit", "nfev"):
            result[name][output_ids] = solved[name]
        result["solver_status"][output_ids] = solved.status
        width_ok = bracket[:, 2]-bracket[:, 0] <= XATOL
        accepted = solved.success & width_ok
        reason = np.full(len(output_ids), 6, dtype=np.uint8)
        reason[solved.success & ~width_ok] = 7
        reason[accepted] = 0
        result["reason"][output_ids] = reason
        result["available"][output_ids] = accepted
        result["offset_px"][output_ids[accepted]] = solved.x[accepted]
    return result


def photometric_scores(arrays):
    fitted = np.vstack((arrays["photo_ref0_fitted_disparity_px"], arrays["photo_ref4_fitted_disparity_px"]))
    available = np.vstack((arrays["photo_ref0_available"], arrays["photo_ref4_available"]))
    cost = np.vstack((arrays["photo_ref0_final_cost"], arrays["photo_ref4_final_cost"]))
    offset = np.vstack((arrays["photo_ref0_offset_px"], arrays["photo_ref4_offset_px"]))
    predicted = np.vstack((arrays["ref0_predicted_disparity_px"], arrays["ref4_predicted_disparity_px"]))
    axial = np.vstack((arrays["ref0_axial_px_per_length"], arrays["ref4_axial_px_per_length"]))
    scores = {}
    for group, rows in GROUP_ROWS.items():
        values = summarize_coordinate_residuals((predicted-fitted)[rows], np.sqrt(cost[rows]), axial[rows], available[rows])
        scores[f"negative_photo_{group}_depth"] = -values["depth_equivalent"]
        if group == "reserved":
            scores["photo_reserved_pair_count"] = values["pair_count"].astype(float)
            scores["negative_photo_reserved_rms"] = -values["lr_rms"]
            count = values["pair_count"]
            square_sum = np.where(available[rows], offset[rows]**2, 0.).sum(axis=0)
            rms = np.full(len(count), np.nan)
            rms[count > 0] = np.sqrt(square_sum[count > 0]/count[count > 0])
            scores["negative_photo_reserved_offset_rms"] = -rms
    return scores
