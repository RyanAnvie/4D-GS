"""Finite-candidate minimum of fixed-support RGB cost in saved coarse brackets."""

import numpy as np
from scipy.ndimage import map_coordinates

from photometric_disparity import fixed_patch_samples, image_cost


BATCH_SIZE = 512


def refine_piecewise(reference, partner, disparity, valid, prior, batch_size=BATCH_SIZE):
    n = len(prior["origins"])
    result = {"edges_px": np.full((n, 27), np.nan), "edge_cost": np.full((n, 27), np.nan),
        "edge_count": np.zeros(n, dtype=np.int16), "interior_representable": np.zeros((n, 26), dtype=bool),
        "interval_A": np.full((n, 26), np.nan), "interval_B": np.full((n, 26), np.nan),
        "vertex_px": np.full((n, 26), np.nan), "vertex_cost": np.full((n, 26), np.nan),
        "offset_px": np.full(n, np.nan), "final_cost": np.full(n, np.nan),
        "candidate_index": np.full(n, -1, dtype=np.int16), "available": np.zeros(n, dtype=bool),
        "reason": prior["reason"].copy()}
    eligible = np.isfinite(prior["initial_bracket"]).all(axis=1)
    assert np.all(np.isin(prior["reason"][~eligible], (3, 4, 5)))
    for start in range(0, n, batch_size):
        stop = min(n, start+batch_size)
        ids = np.flatnonzero(eligible[start:stop])
        if not len(ids):
            continue
        samples = fixed_patch_samples(reference, partner, disparity, valid, prior["origins"][start:stop], prior["coefficients"][start:stop])
        assert np.all(samples["count"][ids] > 0)
        for name in ("candidate_count", "raw_count", "count", "border_excluded_count"):
            np.testing.assert_array_equal(samples[name], prior[name][start:stop])
        output_ids = start+ids
        lower, upper = prior["initial_bracket"][output_ids, 0], prior["initial_bracket"][output_ids, 2]
        assert np.all(upper-lower == .125)
        x0 = samples["target_x"][ids]
        crossings = x0-np.floor(x0-lower[:, None])
        inside = samples["fixed"][ids] & (crossings > lower[:, None]) & (crossings < upper[:, None])
        padded = np.column_stack((lower, np.where(inside, crossings, upper[:, None]), upper))
        sorted_edges = np.sort(padded, axis=1)
        unique = np.column_stack((np.ones(len(ids), dtype=bool), np.diff(sorted_edges, axis=1) > 0.))
        columns = np.cumsum(unique, axis=1)-1
        rows = np.broadcast_to(np.arange(len(ids))[:, None], unique.shape)
        edges = np.full(sorted_edges.shape, np.nan)
        edges[rows[unique], columns[unique]] = sorted_edges[unique]
        counts = unique.sum(axis=1)
        result["edges_px"][output_ids] = edges
        result["edge_count"][output_ids] = counts
        for column in range(int(counts.max())):
            chosen = np.flatnonzero(column < counts)
            result["edge_cost"][output_ids[chosen], column] = image_cost(edges[chosen, column], ids[chosen], samples, partner)
        for interval in range(int(counts.max())-1):
            chosen = np.flatnonzero(interval+1 < counts)
            left, right = edges[chosen, interval], edges[chosen, interval+1]
            mid = (left+right)/2.
            representable = (mid > left) & (mid < right)
            result["interior_representable"][output_ids[chosen], interval] = representable
            chosen = chosen[representable]
            if not len(chosen):
                continue
            left, right, mid = left[representable], right[representable], mid[representable]
            patch_ids = ids[chosen]
            fixed = samples["fixed"][patch_ids]
            y = np.where(fixed, samples["y"][patch_ids], 0)
            target_left = np.where(fixed, samples["target_x"][patch_ids]-left[:, None], 0.)
            target_mid = np.where(fixed, samples["target_x"][patch_ids]-mid[:, None], 0.)
            pixel_left = np.floor(target_mid).astype(int)
            pixel_right = np.minimum(pixel_left+1, partner.shape[1]-1)
            A, B = np.zeros(len(chosen)), np.zeros(len(chosen))
            for channel in range(3):
                intensity = map_coordinates(partner[:, :, channel], [y, target_left], order=1, prefilter=False, mode="constant", cval=np.nan)
                residual = samples["rgb"][patch_ids, :, channel]-intensity
                slope = partner[y, pixel_right, channel]-partner[y, pixel_left, channel]
                residual[~fixed], slope[~fixed] = 0., 0.
                A += np.sum(slope*slope, axis=1)
                B += np.sum(residual*slope, axis=1)
            divisor = 3*samples["count"][patch_ids]
            A, B = A/divisor, B/divisor
            assert np.isfinite(A).all() and np.isfinite(B).all() and np.all(A >= 0.)
            result["interval_A"][output_ids[chosen], interval] = A
            result["interval_B"][output_ids[chosen], interval] = B
            curved = A > 0.
            vertex = np.clip(left[curved]-B[curved]/A[curved], left[curved], right[curved])
            result["vertex_px"][output_ids[chosen[curved]], interval] = vertex
            result["vertex_cost"][output_ids[chosen[curved]], interval] = image_cost(vertex, patch_ids[curved], samples, partner)
        offsets = np.column_stack((result["edges_px"][output_ids], result["vertex_px"][output_ids]))
        costs = np.column_stack((result["edge_cost"][output_ids], result["vertex_cost"][output_ids]))
        best_offset, best_cost = offsets[:, 0].copy(), costs[:, 0].copy()
        best_index = np.zeros(len(ids), dtype=np.int16)
        for index in range(1, offsets.shape[1]):
            candidate, cost = offsets[:, index], costs[:, index]
            closer = (np.abs(candidate) < np.abs(best_offset)) | ((np.abs(candidate) == np.abs(best_offset)) & (candidate < best_offset))
            better = (cost < best_cost) | ((cost == best_cost) & closer)
            best_offset[better], best_cost[better], best_index[better] = candidate[better], cost[better], index
        assert np.isfinite(best_offset).all() and np.isfinite(best_cost).all()
        result["offset_px"][output_ids], result["final_cost"][output_ids] = best_offset, best_cost
        result["candidate_index"][output_ids] = best_index
        result["available"][output_ids], result["reason"][output_ids] = True, 0
    return result
