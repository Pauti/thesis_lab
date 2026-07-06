#!/usr/bin/env python3
"""
Paired significance tests for RQ1 (architecture) and RQ2 (augmentation).

Reuses the exact statistical machinery of analyze_mtl_significance.py (RQ3):
Wilcoxon signed-rank (paired, two-sided), Hodges-Lehmann effect estimate, and
Holm-Bonferroni correction across the WT/TC/ET regions within each comparison.

RQ1 — architecture comparison (nnU-Net vs UNETR), paired per case on the 5-fold
      cross-validation Dice (1251 cases, same cases for both models).
      Delta sign: positive = nnU-Net better than UNETR.

RQ2 — augmentation ablation, paired per case on the 125 held-out test cases.
      Each tier (disabled/reduced/extended) is compared against 'default' within
      each architecture. Delta sign: positive = tier better than default.

Dice for both RQ1 and RQ2. HD95 is additionally tested for RQ1 (architecture),
where per-case values are available for both models; pairs with a non-finite
HD95 on either side (empty region in reference or prediction) are dropped per
region, so the HD95 sample size varies by region.

Output:
  experiments/significance_rq1_rq2.json
  stdout tables

Usage:
    python scripts/analyze_significance_rq1_rq2.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_mtl_significance import hodges_lehmann, holm_bonferroni  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"
REGIONS = [("WT", "dice_wt"), ("TC", "dice_tc"), ("ET", "dice_et")]
HD95_REGIONS = [("WT", "hd95_wt"), ("TC", "hd95_tc"), ("ET", "hd95_et")]
ALPHA = 0.05


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).set_index("case_id").sort_index()


def compare(label: str, a_csv: Path, b_csv: Path, sign_note: str) -> dict:
    """Paired Wilcoxon of A vs B per region (delta = A - B)."""
    a, b = _load(a_csv), _load(b_csv)
    common = a.index.intersection(b.index)
    assert len(common) == len(a) == len(b), (
        f"{label}: case mismatch (a={len(a)}, b={len(b)}, common={len(common)})"
    )
    a, b = a.loc[common], b.loc[common]

    print(f"\n{'='*82}")
    print(f"  {label}   (n={len(common)} paired cases)")
    print(f"  delta = {sign_note}")
    print(f"{'='*82}")
    print(f"  {'Region':<8}{'A mean':>10}{'B mean':>10}{'Δ median':>11}"
          f"{'HL est':>10}{'p (Wilcoxon)':>15}{'sig?':>7}")
    print(f"  {'-'*68}")

    regions, pvals = [], []
    for region, col in REGIONS:
        da, db = a[col].to_numpy(), b[col].to_numpy()
        deltas = da - db
        if np.allclose(deltas, 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(da, db, zero_method="wilcox", alternative="two-sided")
        regions.append({
            "region": region,
            "a_mean": float(da.mean()),
            "b_mean": float(db.mean()),
            "median_delta": float(np.median(deltas)),
            "hodges_lehmann": hodges_lehmann(deltas),
            "wilcoxon_stat": float(stat) if not np.isnan(stat) else None,
            "p_value": float(p),
            "n_a_better": int((deltas > 0).sum()),
            "n_b_better": int((deltas < 0).sum()),
            "n_tie": int((deltas == 0).sum()),
        })
        pvals.append(p)

    for r, rej in zip(regions, holm_bonferroni(pvals)):
        r["p_holm_reject"] = bool(rej)
        print(f"  {r['region']:<8}{r['a_mean']:>10.4f}{r['b_mean']:>10.4f}"
              f"{r['median_delta']:>+11.4f}{r['hodges_lehmann']:>+10.4f}"
              f"{r['p_value']:>15.4g}{'yes' if rej else 'ns':>7}")

    return {"comparison": label, "n_cases": int(len(common)),
            "delta_sign": sign_note, "regions": regions}


def compare_hd95(label: str, a_csv: Path, b_csv: Path, sign_note: str) -> dict:
    """Paired Wilcoxon of A vs B per region on HD95 (delta = A - B, in mm).

    HD95 is non-finite when a region is empty in the reference or the
    prediction, so pairs with a NaN/inf value on either side are dropped per
    region and the contributing sample size is reported alongside each result.
    """
    a, b = _load(a_csv), _load(b_csv)
    common = a.index.intersection(b.index)
    assert len(common) == len(a) == len(b), (
        f"{label}: case mismatch (a={len(a)}, b={len(b)}, common={len(common)})"
    )
    a, b = a.loc[common], b.loc[common]

    print(f"\n{'='*82}")
    print(f"  {label}   (up to n={len(common)} paired cases; HD95 in mm)")
    print(f"  delta = {sign_note}")
    print(f"{'='*82}")
    print(f"  {'Region':<8}{'A mean':>10}{'B mean':>10}{'Δ median':>11}"
          f"{'HL est':>10}{'p (Wilcoxon)':>15}{'sig?':>7}{'n':>7}")
    print(f"  {'-'*75}")

    regions, pvals = [], []
    for region, col in HD95_REGIONS:
        da, db = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
        finite = np.isfinite(da) & np.isfinite(db)
        da, db = da[finite], db[finite]
        deltas = da - db
        if np.allclose(deltas, 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(da, db, zero_method="wilcox", alternative="two-sided")
        regions.append({
            "region": region,
            "a_mean": float(da.mean()),
            "b_mean": float(db.mean()),
            "median_delta": float(np.median(deltas)),
            "hodges_lehmann": hodges_lehmann(deltas),
            "wilcoxon_stat": float(stat) if not np.isnan(stat) else None,
            "p_value": float(p),
            "n_pairs": int(finite.sum()),
            "n_a_worse": int((deltas > 0).sum()),
            "n_b_worse": int((deltas < 0).sum()),
            "n_tie": int((deltas == 0).sum()),
        })
        pvals.append(p)

    for r, rej in zip(regions, holm_bonferroni(pvals)):
        r["p_holm_reject"] = bool(rej)
        print(f"  {r['region']:<8}{r['a_mean']:>10.2f}{r['b_mean']:>10.2f}"
              f"{r['median_delta']:>+11.2f}{r['hodges_lehmann']:>+10.2f}"
              f"{r['p_value']:>15.4g}{'yes' if rej else 'ns':>7}{r['n_pairs']:>7}")

    return {"comparison": label, "n_cases": int(len(common)), "metric": "hd95",
            "delta_sign": sign_note, "regions": regions}


def main() -> None:
    results = {"test": "Wilcoxon signed-rank (paired, two-sided)",
               "correction": "Holm-Bonferroni across WT/TC/ET within each comparison",
               "alpha": ALPHA, "rq1": [], "rq1_hd95": [], "rq2": [], "rq2_hd95": []}

    # ── RQ1: architecture comparison (5-fold CV, 1251 cases) ────────────────────
    results["rq1"].append(compare(
        "RQ1  nnU-Net vs UNETR  (5-fold CV)",
        EXP / "exp02_nnunet/analysis/all_folds_brats_dice.csv",
        EXP / "exp03_unetr/analysis/all_folds_brats_dice_nifti.csv",
        "positive = nnU-Net better than UNETR",
    ))

    # ── RQ1: HD95 boundary significance (per-case; empty regions dropped) ────────
    results["rq1_hd95"].append(compare_hd95(
        "RQ1  UNETR vs nnU-Net  HD95  (5-fold CV)",
        EXP / "exp03_unetr/analysis/all_folds_brats_dice_nifti.csv",
        EXP / "exp02_nnunet/analysis/all_folds_hd95.csv",
        "positive = UNETR has larger (worse) HD95 than nnU-Net",
    ))

    # ── RQ2: augmentation tiers vs default (125 test cases) ─────────────────────
    for arch in ("nnunet", "unetr"):
        default = EXP / f"ablation/abl_{arch}_default/test_predictions/ensemble_results.csv"
        for tier in ("disabled", "reduced", "extended"):
            tier_csv = EXP / f"ablation/abl_{arch}_{tier}/test_predictions/ensemble_results.csv"
            results["rq2"].append(compare(
                f"RQ2  {arch}: {tier} vs default",
                tier_csv, default,
                f"positive = {tier} better than default",
            ))

    # ── RQ2: HD95 tiers vs default (125 test cases; empty regions dropped) ───────
    for arch in ("nnunet", "unetr"):
        default = EXP / f"ablation/abl_{arch}_default/test_predictions/ensemble_results_hd95.csv"
        for tier in ("disabled", "reduced", "extended"):
            tier_csv = EXP / f"ablation/abl_{arch}_{tier}/test_predictions/ensemble_results_hd95.csv"
            results["rq2_hd95"].append(compare_hd95(
                f"RQ2  {arch}: {tier} vs default  (HD95)",
                tier_csv, default,
                f"positive = {tier} has larger (worse) HD95 than default",
            ))

    out = EXP / "significance_rq1_rq2.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
