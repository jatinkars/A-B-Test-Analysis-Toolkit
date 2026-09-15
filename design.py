"""Pre-experiment design: what can this test actually detect?"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["minimum_detectable_effect", "required_sample_size", "power_at"]


def minimum_detectable_effect(
    baseline: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.80
) -> float:
    """
    Smallest absolute lift detectable at this sample size.

    Computing this *before* running is the difference between a test and a
    guess. An underpowered test returns "no significant difference" whether or
    not an effect exists, and that null routinely gets misread as evidence of
    no effect.
    """
    if not 0 < baseline < 1:
        raise ValueError("baseline must be a proportion in (0, 1)")
    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    var = 2 * baseline * (1 - baseline)
    return float((z_alpha + z_power) * np.sqrt(var / n_per_arm))


def required_sample_size(
    baseline: float, mde: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """
    Users per arm needed to detect `mde` with the given power.

    Scales with the inverse square of the effect: halving the effect you want
    to catch quadruples the sample you need.
    """
    if mde <= 0:
        raise ValueError("mde must be positive")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    var = 2 * baseline * (1 - baseline)
    return int(np.ceil(var * (z_alpha + z_power) ** 2 / mde**2))


def power_at(
    baseline: float, effect: float, n_per_arm: int, alpha: float = 0.05
) -> float:
    """Achieved power for a given true effect and sample size."""
    se = np.sqrt(2 * baseline * (1 - baseline) / n_per_arm)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    return float(
        stats.norm.cdf(abs(effect) / se - z_alpha)
        + stats.norm.cdf(-abs(effect) / se - z_alpha)
    )
