"""
Test whether the ChestX-ray14 age distribution is Gaussian.

    XRAY_DATASET_ROOT=/path/to/nih python age_normality_test.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

import config


def report(name: str, ages: np.ndarray) -> None:
    ages = np.asarray(ages, dtype=float)
    n = len(ages)
    print(f"\n=== {name}  (n = {n:,}) ===")
    print(f"  mean {ages.mean():.2f}   median {np.median(ages):.0f}   "
          f"std {ages.std(ddof=1):.2f}   range {ages.min():.0f}-{ages.max():.0f}")

    decade = pd.cut(ages, bins=range(0, 101, 10)).value_counts().idxmax()
    print(f"  most common decade: {decade}")

    # Unlike p-values, these don't shrink with n.
    skew = stats.skew(ages)
    kurt = stats.kurtosis(ages)  # excess
    print(f"  skewness {skew:+.4f}   excess kurtosis {kurt:+.4f}")

    k2, p_k2 = stats.normaltest(ages)
    print(f"  D'Agostino-Pearson  K2 = {k2:.1f}   p = {p_k2:.3g}")

    jb, p_jb = stats.jarque_bera(ages)
    print(f"  Jarque-Bera         JB = {jb:.1f}   p = {p_jb:.3g}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        anderson = stats.anderson(ages, dist="norm")
    crit_5pct = anderson.critical_values[2]
    print(f"  Anderson-Darling    A2 = {anderson.statistic:.2f}   "
          f"critical value (5%) = {crit_5pct:.3f}")

    ks = stats.kstest(ages, "norm", args=(ages.mean(), ages.std(ddof=1)))
    print(f"  Kolmogorov-Smirnov   D = {ks.statistic:.4f}   p = {ks.pvalue:.3g}")

    # Shapiro-Wilk caps at n = 5000, so use subsamples.
    rng = np.random.default_rng(0)
    w_stats, p_values = [], []
    for _ in range(20):
        draw = rng.choice(ages, size=min(5000, n), replace=False)
        w, p = stats.shapiro(draw)
        w_stats.append(w)
        p_values.append(p)
    print(f"  Shapiro-Wilk (20 subsamples of 5,000)   "
          f"W (median) = {np.median(w_stats):.4f}   "
          f"p (median) = {np.median(p_values):.3g}   "
          f"max p = {max(p_values):.3g}")


def main() -> None:
    df = pd.read_csv(config.DATA_ENTRY_CSV)
    ages = df["Patient Age"]

    implausible = ages > 100
    bad_values = sorted(int(v) for v in ages[implausible].unique())
    print(f"Records with implausible age (> 100): {int(implausible.sum())} "
          f"- values {bad_values}")

    report("All images, raw", ages)
    report("All images, plausible ages (<= 100)", ages[~implausible])

    per_patient = df.groupby("Patient ID")["Patient Age"].median()
    report("Per patient (median), plausible ages",
           per_patient[per_patient <= 100])

    scans = df.groupby("Patient ID").size()
    print(f"\nScans per patient: mean {scans.mean():.2f}, "
          f"median {scans.median():.0f}, max {scans.max()}")
    print(f"Patients with more than one scan: "
          f"{(scans > 1).mean():.1%} ({int((scans > 1).sum()):,} of {len(scans):,})")


if __name__ == "__main__":
    main()
