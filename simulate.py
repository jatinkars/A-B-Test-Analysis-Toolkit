"""Synthetic experiment data with a known true effect."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["simulate_experiment"]


def simulate_experiment(
    n: int = 40_000,
    baseline: float = 0.12,
    true_lift: float = 0.015,
    seed: int | None = None,
    split: float = 0.5,
) -> pd.DataFrame:
    """
    Binary outcome with one pre-experiment covariate.

    `prior_activity` is measured BEFORE assignment, so the treatment cannot
    have influenced it. That is what makes it legitimate for variance
    reduction later; adjusting for a post-treatment variable would reintroduce
    exactly the bias randomisation removes.
    """
    rng = np.random.default_rng(seed)
    arm = np.where(rng.random(n) < split, "treatment", "control")
    prior_activity = rng.gamma(shape=2.0, scale=1.5, size=n)

    logit = np.log(baseline / (1 - baseline)) + 0.18 * (
        prior_activity - prior_activity.mean()
    )
    p = 1 / (1 + np.exp(-logit))
    p = np.where(arm == "treatment", np.clip(p + true_lift, 0, 1), p)

    return pd.DataFrame(
        {
            "arm": arm,
            "prior_activity": prior_activity,
            "converted": rng.binomial(1, p),
        }
    )
