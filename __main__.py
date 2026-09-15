"""Command-line report: design, validate, estimate, then state the limits."""

from __future__ import annotations

import argparse

import pandas as pd

from . import (
    minimum_detectable_effect,
    peeking_false_positive_rate,
    regression_adjusted,
    required_sample_size,
    sample_ratio_mismatch,
    simulate_experiment,
    two_proportion_test,
)

BAR = "-" * 62


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m abtest")
    ap.add_argument("--n", type=int, default=40_000)
    ap.add_argument("--baseline", type=float, default=0.12)
    ap.add_argument("--true-lift", type=float, default=0.015)
    ap.add_argument("--csv", type=str, default=None,
                    help="your own data: columns arm, converted, prior_activity")
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()

    print("\n1. DESIGN")
    print(BAR)
    n_per_arm = args.n // 2
    mde = minimum_detectable_effect(args.baseline, n_per_arm)
    print(f"Baseline conversion          {args.baseline:.1%}")
    print(f"Sample per arm               {n_per_arm:,}")
    print(f"MDE (80% power, alpha 0.05)  {mde:.4f}  ({mde/args.baseline:.1%} relative)")
    for target in (0.005, 0.01, 0.02):
        print(f"  to detect {target:.1%} absolute lift: "
              f"{required_sample_size(args.baseline, target):,} per arm")

    if args.csv:
        df = pd.read_csv(args.csv)
        print(f"\nLoaded {len(df):,} rows from {args.csv}")
    else:
        df = simulate_experiment(args.n, args.baseline, args.true_lift, seed=args.seed)
        print(f"\nSimulated {len(df):,} users, true lift = {args.true_lift:.4f}")
        if 0 < args.true_lift < mde:
            print("NOTE: true lift is below the MDE. This test is underpowered")
            print("      for the effect present; a null result is expected.")

    print("\n2. ASSIGNMENT VALIDATION")
    print(BAR)
    srm = sample_ratio_mismatch(df)
    print(srm)
    if not srm.passed:
        print("\nStopping. A biased assignment cannot be repaired downstream.")
        return 1

    print("\n3. EFFECT ESTIMATE")
    print(BAR)
    r = two_proportion_test(df)
    print(r)
    print(f"Relative lift                {r.rel_lift:+.2%}")
    if not args.csv:
        print(f"True lift {args.true_lift:+.4f} inside CI? "
              f"{'yes' if r.covers(args.true_lift) else 'no'}")

    if "prior_activity" in df.columns:
        print("\n4. REGRESSION-ADJUSTED ESTIMATE")
        print(BAR)
        adj = regression_adjusted(df)
        print(f"Treatment coefficient  {adj['coef']:+.4f}")
        print(f"Odds ratio             {adj['odds_ratio']:.4f}")
        print(f"95% CI (log-odds)      [{adj['ci_low']:+.4f}, {adj['ci_high']:+.4f}]")
        print(f"p-value                {adj['p_value']:.4f}")

    print("\n5. WHY WE DO NOT PEEK")
    print(BAR)
    pk = peeking_false_positive_rate(seed=args.seed)
    print(f"{pk.simulations} A/A tests, checked {pk.checks} times each")
    print(f"Nominal false positive rate  {pk.nominal_alpha:.1%}")
    print(f"Actual false positive rate   {pk.actual_rate:.1%}")
    print(f"Inflation factor             {pk.inflation:.1f}x")
    print("\nStopping at the first significant look turns a 5% error rate into")
    print("the rate above. Fix the sample size in advance, or use a sequential")
    print("procedure designed for repeated testing.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
