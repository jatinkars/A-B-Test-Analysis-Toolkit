"""Effect estimation: unadjusted and regression-adjusted."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["TestResult", "two_proportion_test", "regression_adjusted"]


@dataclass
class TestResult:
    control_rate: float
    treatment_rate: float
    abs_lift: float
    rel_lift: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool

    def covers(self, value: float) -> bool:
        """Does the interval contain this value? Used by the calibration tests."""
        return self.ci_low <= value <= self.ci_high

    def __str__(self) -> str:
        verdict = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"{self.control_rate:.4f} -> {self.treatment_rate:.4f}  "
            f"lift {self.abs_lift:+.4f} "
            f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}]  "
            f"p={self.p_value:.4f}  {verdict}"
        )


def two_proportion_test(df: pd.DataFrame, alpha: float = 0.05) -> TestResult:
    """
    Two-proportion z-test.

    Note the two standard errors. The p-value uses a POOLED standard error,
    which assumes the null is true - correct, because that is the hypothesis
    being tested. The confidence interval uses an UNPOOLED one, because it must
    hold under the observed rates rather than under the null. Using the pooled
    form for both is a common and quiet error.
    """
    c = df.loc[df.arm == "control", "converted"]
    t = df.loc[df.arm == "treatment", "converted"]
    if len(c) == 0 or len(t) == 0:
        raise ValueError("both arms must be non-empty")

    pc, pt = float(c.mean()), float(t.mean())
    nc, nt = len(c), len(t)

    pooled = (c.sum() + t.sum()) / (nc + nt)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / nc + 1 / nt))
    z = (pt - pc) / se_pooled if se_pooled > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    se_unpooled = np.sqrt(pc * (1 - pc) / nc + pt * (1 - pt) / nt)
    margin = stats.norm.ppf(1 - alpha / 2) * se_unpooled

    return TestResult(
        control_rate=pc,
        treatment_rate=pt,
        abs_lift=pt - pc,
        rel_lift=(pt - pc) / pc if pc else float("nan"),
        ci_low=(pt - pc) - margin,
        ci_high=(pt - pc) + margin,
        p_value=p_value,
        significant=p_value < alpha,
    )


def regression_adjusted(df: pd.DataFrame, covariate: str = "prior_activity") -> dict:
    """
    Logistic regression on treatment plus a pre-experiment covariate.

    Randomisation already delivers an unbiased estimate, so this is not about
    confounding. Conditioning on a pre-treatment variable that predicts the
    outcome reduces residual variance and tightens the interval at the same
    sample size.

    The covariate MUST be measured before assignment.
    """
    import statsmodels.api as sm

    X = pd.DataFrame(
        {
            "treatment": (df["arm"] == "treatment").astype(int),
            covariate: df[covariate] - df[covariate].mean(),
        }
    )
    model = sm.Logit(df["converted"], sm.add_constant(X)).fit(disp=0)
    ci = model.conf_int().loc["treatment"]
    return {
        "coef": float(model.params["treatment"]),
        "odds_ratio": float(np.exp(model.params["treatment"])),
        "p_value": float(model.pvalues["treatment"]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
    }
