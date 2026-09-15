"""What repeated significance testing does to the false positive rate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PeekingResult", "peeking_false_positive_rate"]


@dataclass
class PeekingResult:
    nominal_alpha: float
    actual_rate: float
    checks: int
    simulations: int

    @property
    def inflation(self) -> float:
        return self.actual_rate / self.nominal_alpha


def peeking_false_positive_rate(
    n_sims: int = 400,
    n: int = 20_000,
    checks: int = 10,
    baseline: float = 0.12,
    alpha: float = 0.05,
    seed: int | None = None,
) -> PeekingResult:
    """
    Run A/A tests - no true effect - testing repeatedly as data accrues and
    stopping at the first significant result.

    Nominal alpha is 5%. Repeated peeking inflates it well past that. This is
    the most common way a careful analyst ships a false win.

    Fix the sample size in advance, or use a sequential procedure built for
    repeated testing (alpha spending, always-valid p-values).
    """
    rng = np.random.default_rng(seed)
    points = np.linspace(n // checks, n, checks).astype(int)
    z_crit = 1.959963984540054  # two-sided 0.05
    false_positives = 0

    for _ in range(n_sims):
        c = rng.binomial(1, baseline, n)
        t = rng.binomial(1, baseline, n)  # identical arms
        for k in points:
            pc, pt = c[:k].mean(), t[:k].mean()
            pooled = (c[:k].sum() + t[:k].sum()) / (2 * k)
            se = np.sqrt(pooled * (1 - pooled) * (2 / k))
            if se > 0 and abs((pt - pc) / se) > z_crit:
                false_positives += 1
                break

    return PeekingResult(alpha, false_positives / n_sims, checks, n_sims)
