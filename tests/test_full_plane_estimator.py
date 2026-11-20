"""Spyder F5 numerical tests, including an independent bounded linear solution."""

from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from full_plane_estimator import fit_full_plane, plane_objective, plane_residual


def reference_values(q, partner, y, target_x, rgb, design):
    residual, rows = [], []
    for p in range(len(y)):
        x = float(target_x[p] - np.dot(design[p], q))
        column = max(0, min(int(np.floor(x)), partner.shape[1] - 2))
        for c in range(3):
            slope = float(partner[y[p], column + 1, c] - partner[y[p], column, c])
            intensity = float(partner[y[p], column, c]) + (x - column) * slope
            residual.append(float(rgb[p, c]) - intensity)
            rows.append(design[p] * slope)
    return np.array(residual), np.array(rows)


def independent_bounded_linear(B, residual, design):
    # Enumerate active sets for a three-variable convex quadratic fixture only.
    planes = np.unique(np.vstack((design, -design)), axis=0)
    gram, gradient = B.T @ B, B.T @ residual
    best, best_cost = None, np.inf
    for count in range(4):
        for active in combinations(range(len(planes)), count):
            C = planes[list(active)]
            system = np.block([[gram, C.T], [C, np.zeros((count, count))]])
            rhs = np.concatenate((-gradient, np.ones(count)))
            solution, _, rank, _ = np.linalg.lstsq(system, rhs, rcond=None)
            if rank != len(system):
                continue
            q = solution[:3]
            if np.any(planes @ q > 1 + 1e-10) or np.any(solution[3:] < -1e-10):
                continue
            cost = float(np.sum((residual + B @ q) ** 2))
            if cost < best_cost:
                best, best_cost = q, cost
    assert best is not None
    return best, best_cost


def main():
    height, width = 12, 20
    yy, xx = np.mgrid[:height, :width]
    partner = np.stack((.02 * xx + .01 * yy, .03 * xx - .01 * yy, .01 * xx + .02 * yy), axis=2)
    dy, dx = np.mgrid[-2:3, -2:3]
    design = np.column_stack((np.ones(25), dx.ravel(), dy.ravel()))
    y = 5 + dy.ravel()
    target_x = 9.35 + dx.ravel()
    tolerance = 128 * np.finfo(float).eps * (1 + width + 5)
    dummy = np.zeros((25, 3))
    cases = 0
    for true_q in (np.array([.21, .04, -.07]), np.array([-.31, -.08, .03]), np.zeros(3)):
        negative_rgb, _, _, _ = plane_residual(true_q, partner, y, target_x, dummy, design)
        rgb = -negative_rgb.reshape(25, 3)
        r, B = reference_values(np.zeros(3), partner, y, target_x, rgb, design)
        H = float(B[:, 0] @ B[:, 0])
        fitted = fit_full_plane(partner, y, target_x, rgb, design, 0., H, True, tolerance)
        assert fitted["accepted"]
        np.testing.assert_allclose(fitted["used_q"], true_q, atol=2e-7, rtol=0)
        again = fit_full_plane(partner, y, target_x, rgb, design, 0., H, True, tolerance)
        np.testing.assert_array_equal(fitted["candidate_q"], again["candidate_q"])
        cases += 1
    # A boundary optimum has a nonzero gradient; a success flag is not enough.
    corners = np.array([0, 4, 20, 24])
    D, ys, xs = design[corners], y[corners], target_x[corners]
    truth = np.array([.8, .45, -.3])
    r, _, _, _ = plane_residual(truth, partner, ys, xs, np.zeros((4, 3)), D)
    rgb = -r.reshape(4, 3) + np.array([.002, -.001, -.001])
    r0, B0 = reference_values(np.zeros(3), partner, ys, xs, rgb, D)
    H = float(B0[:, 0] @ B0[:, 0])
    expected, cost = independent_bounded_linear(B0, r0, D)
    fitted = fit_full_plane(partner, ys, xs, rgb, D, 0., H, True, tolerance)
    assert fitted["accepted"] and np.any(abs(D @ fitted["used_q"]) > 1 - 1e-8)
    np.testing.assert_allclose(fitted["used_q"], expected, atol=2e-7, rtol=0)
    np.testing.assert_allclose(fitted["candidate_sse"], cost, atol=1e-12, rtol=1e-10)
    cases += 1
    # Nonlinear image segments: costs, signs, derivative and branch crossings.
    partner = partner + .001 * xx[:, :, None] ** 2
    true_q = np.array([.7, .025, -.02])
    r, _, _, _ = plane_residual(true_q, partner, y, target_x, dummy, design)
    rgb = -r.reshape(25, 3)
    q = np.array([.1, .02, .01])
    r_ref, B_ref = reference_values(q, partner, y, target_x, rgb, design)
    r, B, _, _ = plane_residual(q, partner, y, target_x, rgb, design)
    np.testing.assert_allclose(r, r_ref, atol=1e-15, rtol=0)
    np.testing.assert_allclose(B, B_ref, atol=1e-15, rtol=0)
    H = float(B[:, 0] @ B[:, 0])
    _, gradient = plane_objective(q, partner, y, target_x, rgb, design, H)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = 1e-6
        plus = plane_objective(q + step, partner, y, target_x, rgb, design, H)[0]
        minus = plane_objective(q - step, partner, y, target_x, rgb, design, H)[0]
        np.testing.assert_allclose(gradient[axis], (plus - minus) / 2e-6, atol=1e-8, rtol=1e-7)
    fitted = fit_full_plane(partner, y, target_x, rgb, design, 0., H, True, tolerance)
    assert fitted["accepted"] and fitted["branch_changes"] > 0
    np.testing.assert_allclose(fitted["used_q"], true_q, atol=2e-6, rtol=0)
    cases += 1
    # Flat/rank-deficient and missing observations retain the original estimate.
    flat = np.zeros_like(partner)
    fitted = fit_full_plane(flat, y, target_x, dummy, design, .2, 1., True, tolerance)
    assert fitted["reason"] == 2 and not fitted["accepted"]
    np.testing.assert_array_equal(fitted["used_q"], [.2, 0, 0])
    one_row = dy.ravel() == 0
    fitted = fit_full_plane(partner, y[one_row], target_x[one_row], rgb[one_row], design[one_row], 0., H, True, tolerance)
    assert fitted["reason"] == 2
    fitted = fit_full_plane(partner, y[:0], target_x[:0], rgb[:0], design[:0], np.nan, np.nan, False, tolerance)
    assert fitted["reason"] == 1 and np.isnan(fitted["used_q"][0])
    cases += 3
    # Border trials use linear extension. At integer coordinates use right branch.
    xs = np.array([-1., 0., width - 1., width + 1.])
    D = np.tile([1., 0., 0.], (4, 1))
    ys = np.zeros(4, dtype=int)
    rgb = np.zeros((4, 3))
    r, B, left, x = plane_residual(np.zeros(3), partner, ys, xs, rgb, D)
    rr, BB = reference_values(np.zeros(3), partner, ys, xs, rgb, D)
    np.testing.assert_allclose(r, rr, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(left, [0, 0, width - 2, width - 2])
    np.testing.assert_array_equal(x, xs)
    np.testing.assert_allclose(B, BB, atol=1e-15, rtol=0)
    cases += 1
    print(f"PASS: {cases} fixture groups; analytic/independent/finite-difference/active-set checks.")


if __name__ == "__main__":
    main()
