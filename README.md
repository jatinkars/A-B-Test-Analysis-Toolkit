# A/B Test Analysis Toolkit

![tests](https://github.com/jatinkars/AB-Test-Analysis-Toolkit/actions/workflows/ci.yml/badge.svg)

Design the test before running it, validate the assignment, estimate the effect
two ways, and quantify what the result cannot support.

```bash
pip install -r requirements.txt
python -m abtest                  # full report on simulated data
python -m abtest --true-lift 0.0  # null case; correctly returns non-significant
python -m abtest --csv mydata.csv # your own experiment
```

## The part that makes this different

Most analytics repos test that the code runs. **This one tests that the
statistics are correct**, and does it in CI on every push.

A 95% confidence interval makes a falsifiable promise: across repeated
experiments it covers the true value 95% of the time. That promise is testable,
and almost nobody tests it. `tests/test_calibration.py` does:

| Test | What it asserts |
|---|---|
| `test_confidence_interval_achieves_nominal_coverage` | The 95% CI covers the known true effect ~95% of the time across 300 replications |
| `test_false_positive_rate_matches_alpha_under_the_null` | With no true effect, a 0.05 test rejects ~5% of the time |
| `test_point_estimate_is_unbiased` | Mean estimate converges on the truth within 4 standard errors |
| `test_achieved_power_matches_design_target` | A test sized for 80% power actually rejects ~80% of the time |
| `test_sample_size_and_mde_are_inverses` | The design functions round-trip to within 2% |

If a refactor silently breaks a standard error, coverage drifts off nominal and
**the build fails**. That is the guarantee worth having.

## What the toolkit does

**1. Design — what can this test detect?**

```
Baseline conversion          12.0%
Sample per arm               20,000
MDE (80% power, alpha 0.05)  0.0091  (7.6% relative)
  to detect 0.5% absolute lift: 66,308 per arm
  to detect 1.0% absolute lift: 16,577 per arm
  to detect 2.0% absolute lift:  4,145 per arm
```

Sample size scales with the inverse square of the effect: halving what you want
to catch quadruples what you need. An underpowered test returns "no significant
difference" whether or not an effect exists, and that null gets misread as
evidence of no effect.

**2. Assignment validation — sample ratio mismatch**

Checked first, because a failure here invalidates everything downstream. If
randomisation broke — a redirect dropping users, logging missing on one arm —
the estimate is biased and no downstream statistics repair it.

The threshold is deliberately strict (p < 0.001, not 0.05). SRM is checked on
every experiment, so a 5% threshold would fail one in twenty healthy tests.

**3. Effect estimate**

```
0.1291 -> 0.1418  lift +0.0126 [+0.0059, +0.0194]  p=0.0002  SIGNIFICANT
True lift +0.0150 inside CI? yes
```

Note the two standard errors. The p-value uses a **pooled** SE, which assumes
the null — correct, since that is the hypothesis under test. The interval uses
an **unpooled** one, because it must hold under the observed rates. Using the
pooled form for both is a common and quiet error, and `test_interval_is_not_the_pooled_one`
guards against someone "simplifying" it back.

**4. Regression adjustment**

Logistic regression on treatment plus a pre-experiment covariate. Randomisation
already gives an unbiased estimate, so this is not about confounding —
conditioning on a variable that predicts the outcome reduces residual variance
and tightens the interval at the same sample size.

The covariate must be measured *before* assignment. Adjusting for anything the
treatment could have influenced reintroduces exactly the bias randomisation
removed.

**5. Why we do not peek**

400 A/A tests — no true effect — each checked 10 times as data accrued, stopping
at the first significant result:

```
Nominal false positive rate  5.0%
Actual false positive rate   21.5%
Inflation factor             4.3x
```

A 5% error rate becomes 21.5%. This is the most common way a careful analyst
ships a false win. Fix the sample size in advance, or use a sequential procedure
built for repeated testing.

## Layout

```
abtest/
  design.py      power, MDE, required sample size
  validate.py    sample ratio mismatch
  inference.py   two-proportion test, regression adjustment
  peeking.py     sequential-testing error inflation
  simulate.py    synthetic data with a known true effect
  __main__.py    CLI report
tests/
  test_units.py        contracts, edge cases, pooled/unpooled guard
  test_calibration.py  statistical guarantees (see above)
```

## Why simulated data

The true effect is unobservable in a real experiment, which is precisely why the
methods cannot be validated on real data. Setting it explicitly lets every
estimator be scored against ground truth. Pass `--csv` and the same analysis
runs on a real experiment.

## Limitations

- **Binary outcomes only.** Continuous and revenue metrics need different
  variance handling; heavy tails need trimming or winsorization.
- **Single metric.** No guardrails, no multiple-comparison correction across a
  metric suite.
- **No CUPED.** Covariate adjustment here is plain regression; CUPED on
  pre-period outcomes would reduce variance further.
- **No sequential testing.** The peeking module measures the problem; it does
  not implement alpha spending or always-valid p-values as a fix.
- **No heterogeneous treatment effects.** Segment analysis needs pre-registered
  segments to avoid fishing.

## Licence

MIT.
