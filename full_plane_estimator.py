"""Constrained RGB plane refinement in signed disparity pixels, not covariance."""

import numpy as np
from scipy.optimize import LinearConstraint, minimize


MAXITER = 100
FTOL = 1e-12
REASONS = {0: "accepted", 1: "parent_ineligible", 2: "initial_rank_deficient",
           3: "solver_failed", 4: "nonfinite_candidate", 5: "infeasible_candidate",
           6: "increased_cost"}


def plane_residual(q, partner, y, target_x, rgb, design):
    # Integer x uses the segment to its right, except the last image column.
    # Outside-image trial points use a linear extension, never clipped RGB.
    x = target_x - design @ q
    left = np.clip(np.floor(x).astype(int), 0, partner.shape[1] - 2)
    slope = partner[y, left + 1] - partner[y, left]
    sampled = partner[y, left] + (x - left)[:, None] * slope
    residual = rgb - sampled
    B = (slope[:, :, None] * design[:, None, :]).reshape(-1, 3)
    return residual.ravel(), B, left, x


def plane_objective(q, partner, y, target_x, rgb, design, H):
    residual, B, _, _ = plane_residual(q, partner, y, target_x, rgb, design)
    sse = float(residual @ residual)
    gradient = 2 * B.T @ residual
    return sse / H, gradient / H


def fit_full_plane(partner, y, target_x, rgb, design, offset, H, eligible, tolerance):
    initial = np.array([offset, 0., 0.])
    result = {"initial_q": initial, "candidate_q": np.full(3, np.nan),
              "used_q": initial.copy(), "accepted": False, "reason": 1,
              "solver_success": False, "solver_status": -99, "message": "not_run",
              "nit": 0, "nfev": 0, "njev": 0, "H": float(H),
              "initial_sse": np.nan, "candidate_sse": np.nan, "cost_tolerance": np.nan,
              "initial_rank": -1, "candidate_rank": -1,
              "initial_singular": np.full(3, np.nan), "candidate_singular": np.full(3, np.nan),
              "initial_gradient": np.full(3, np.nan), "candidate_gradient": np.full(3, np.nan),
              "strict_feasible": False, "tolerance_feasible": False,
              "max_violation": np.nan, "coordinate_tolerance": float(tolerance),
              "branch_changes": -1, "boundary_hits": -1}
    if not eligible:
        return result
    if not len(y) or not np.isfinite(initial).all() or not np.isfinite(H) or H <= 0:
        raise ValueError("Parent-eligible patches need support, finite offset and positive H.")
    if np.any(abs(design @ initial) > 1) or np.any(target_x - 1 < 0) or np.any(target_x + 1 > partner.shape[1] - 1):
        raise ValueError("Initial measurement and fixed support must be in the frozen domain.")
    r0, B0, left0, _ = plane_residual(initial, partner, y, target_x, rgb, design)
    singular = np.linalg.svd(B0, compute_uv=False)
    rank = int(np.sum(singular > np.finfo(float).eps * max(B0.shape) * singular[0]))
    result.update(initial_sse=float(r0 @ r0), initial_rank=rank,
                  initial_singular=singular, initial_gradient=2 * B0.T @ r0)
    if rank != 3:
        result["reason"] = 2
        return result
    constraint = LinearConstraint(design, -1., 1.)
    solved = minimize(plane_objective, initial, args=(partner, y, target_x, rgb, design, H),
                      method="SLSQP", jac=True, constraints=[constraint],
                      options={"maxiter": MAXITER, "ftol": FTOL})
    q = solved.x.copy()
    result.update(candidate_q=q, solver_success=bool(solved.success), solver_status=int(solved.status),
                  message=str(solved.message), nit=int(solved.nit), nfev=int(solved.nfev), njev=int(solved.njev))
    if not np.isfinite(q).all():
        result["reason"] = 4
        return result
    r, B, left, x = plane_residual(q, partner, y, target_x, rgb, design)
    sse = float(r @ r)
    singular = np.linalg.svd(B, compute_uv=False)
    rank = int(np.sum(singular > np.finfo(float).eps * max(B.shape) * singular[0]))
    offsets = design @ q
    violation = max(0., float(np.max(abs(offsets) - 1)), float(np.max(-x)),
                    float(np.max(x - (partner.shape[1] - 1))))
    # Bound arithmetic error using the actual interpolation slopes and residuals.
    slope = B[:, 0]
    residual_bound = abs(slope) * tolerance + 128 * np.finfo(float).eps * (1 + abs(r))
    cost_tolerance = float(np.sum((2 * abs(r) + residual_bound) * residual_bound))
    slope0 = B0[:, 0]
    residual_bound0 = abs(slope0) * tolerance + 128 * np.finfo(float).eps * (1 + abs(r0))
    cost_tolerance += float(np.sum((2 * abs(r0) + residual_bound0) * residual_bound0))
    result.update(candidate_sse=sse, candidate_rank=rank, candidate_singular=singular,
                  candidate_gradient=2 * B.T @ r, strict_feasible=violation == 0,
                  tolerance_feasible=violation <= tolerance, max_violation=violation,
                  cost_tolerance=cost_tolerance, branch_changes=int(np.sum(left != left0)),
                  boundary_hits=int(np.sum(abs(x - np.rint(x)) <= tolerance)))
    if not np.isfinite(sse) or not np.isfinite(B).all():
        result["reason"] = 4
    elif not solved.success:
        result["reason"] = 3
    elif violation > tolerance:
        result["reason"] = 5
    elif sse > result["initial_sse"] + cost_tolerance:
        result["reason"] = 6
    else:
        result.update(accepted=True, reason=0, used_q=q.copy())
    return result
