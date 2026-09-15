"""Assignment validation. Run this before looking at any effect estimate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["SRMResult", "sample_ratio_mismatch"]


@dataclass
class SRMResult:
    control: int
    treatment: int
    expected_ratio: float
    p_value: float
    passed: bool

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL - do not interpret the result"
        return (
            f"control {self.control:,}  treatment {self.treatment:,}  "
            f"p={self.p_value:.4f}  {verdict}"
        )


def sample_ratio_mismatch(
    df: pd.DataFrame, expected: float = 0.5, threshold: float = 0.001
) -> SRMResult:
    """
    Chi-square test on arm sizes.

    Checked first because a failure here invalidates everything downstream. If
    randomisation broke - a redirect dropping users, logging missing on one arm
    - the effect estimate is biased and no downstream statistics repair it.

    The threshold is deliberately strict (0.001, not 0.05): SRM is checked on
    every experiment, so a 5% threshold would fail one in twenty healthy tests.
    """
    counts = df["arm"].value_counts()
    obs = np.array([counts.get("control", 0), counts.get("treatment", 0)], dtype=float)
    n = obs.sum()
    exp = np.array([n * expected, n * (1 - expected)])

    chi2 = float(((obs - exp) ** 2 / exp).sum())
    p = float(1 - stats.chi2.cdf(chi2, df=1))

    return SRMResult(int(obs[0]), int(obs[1]), expected, p, p > threshold)
