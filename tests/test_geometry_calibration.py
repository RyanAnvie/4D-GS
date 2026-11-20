"""Group weighting, disjoint calibration/test simulation, and interval units."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from gs_uncertainty.calibration import group_balanced_quantile, fit_error_bounds, evaluate_error_bounds


def run_checks():
    values = np.concatenate((np.ones(10), np.full(1000, 10.)))
    groups = np.array(["small"] * 10 + ["large"] * 1000)
    assert group_balanced_quantile(values, groups, 0.4) == 1.
    assert group_balanced_quantile(values, groups, 0.6) == 10.
    assert group_balanced_quantile(values, groups, 1.) == 10.
    rng = np.random.default_rng(42)
    scale = np.concatenate((np.ones(20000), np.full(20000, 5.)))
    group = np.array(["low"] * 20000 + ["high"] * 20000)
    calibration_error = np.abs(rng.standard_normal(len(scale))) * scale
    fitted = fit_error_bounds(scale, calibration_error, group)
    constant = fit_error_bounds(np.ones_like(scale), calibration_error, group)
    test_error = np.abs(rng.standard_normal(len(scale))) * scale
    results = evaluate_error_bounds(scale, test_error, group, fitted)
    baseline = evaluate_error_bounds(np.ones_like(scale), test_error, group, constant)
    for row in results:
        assert row["absolute_coverage_error"] < 0.01
    assert results[2]["group_balanced_mean_bound"] < baseline[2]["group_balanced_mean_bound"]
    # A change of length units must change bounds but not coverage.
    rescaled_fit = fit_error_bounds(scale * 1000, calibration_error * 1000, group)
    rescaled = evaluate_error_bounds(scale * 1000, test_error * 1000, group, rescaled_fit)
    for before, after in zip(results, rescaled):
        assert before["group_balanced_coverage"] == after["group_balanced_coverage"]
        np.testing.assert_allclose(after["group_balanced_mean_bound"], before["group_balanced_mean_bound"] * 1000)
    print("PASS: equal scene weighting, disjoint heteroscedastic simulation, coverage/width and unit equivariance.")


if __name__ == "__main__":
    run_checks()
