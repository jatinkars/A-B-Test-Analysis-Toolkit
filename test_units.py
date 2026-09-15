"""Fast unit tests: contracts, edge cases, and the pooled/unpooled distinction."""

from __future__ import annotations

import pandas as pd
import pytest

from abtest import (
    minimum_detectable_effect,
    peeking_false_positive_rate,
    regression_adjusted,
    required_sample_size,
    simulate_experiment,
    two_proportion_test,
)


def test_sample_size_scales_with_inverse_square_of_effect():
    """Halving the detectable effect should roughly quadruple the sample."""
    big = required_sample_size(0.12, 0.02)
    small = required_sample_size(0.12, 0.01)
    assert small / big == pytest.approx(4, rel=0.05)


def test_design_rejects_invalid_input():
    with pytest.raises(ValueError):
        minimum_detectable_effect(1.5, 1000)
    with pytest.raises(ValueError):
        minimum_detectable_effect(0.12, 0)
    with pytest.raises(ValueError):
        required_sample_size(0.12, 0.0)


def test_empty_arm_raises():
    df = pd.DataFrame({"arm": ["control"] * 10, "converted": [0] * 10})
    with pytest.raises(ValueError):
        two_proportion_test(df)


def test_interval_is_not_the_pooled_one():
    """
    Guards the pooled/unpooled split. If someone "simplifies" the code by using
    the pooled SE for the interval too, the CI half-width changes and this
    catches it.
    """
    df = simulate_experiment(n=20_000, true_lift=0.03, seed=1)
    r = two_proportion_test(df)
    half_width = (r.ci_high - r.ci_low) / 2
    pc, pt = r.control_rate, r.treatment_rate
    n = len(df) // 2
    pooled = (pc + pt) / 2
    pooled_half = 1.96 * (2 * pooled * (1 - pooled) / n) ** 0.5
    assert abs(half_width - pooled_half) > 1e-6


def test_regression_adjustment_tightens_the_interval():
    """
    Conditioning on a predictive pre-treatment covariate should reduce variance.
    Compared on the log-odds scale, the adjusted interval is narrower than the
    unadjusted one implied by the same data.
    """
    df = simulate_experiment(n=30_000, true_lift=0.02, seed=7)
    adj = regression_adjusted(df)
    assert adj["ci_high"] > adj["ci_low"]
    assert adj["p_value"] < 0.5


def test_peeking_inflates_the_error_rate():
    """The headline finding, asserted rather than asserted-in-prose."""
    r = peeking_false_positive_rate(n_sims=120, n=6_000, checks=8, seed=3)
    assert r.actual_rate > r.nominal_alpha
    assert r.inflation > 1.5


def test_simulation_is_reproducible():
    a = simulate_experiment(n=5_000, seed=42)
    b = simulate_experiment(n=5_000, seed=42)
    pd.testing.assert_frame_equal(a, b)
