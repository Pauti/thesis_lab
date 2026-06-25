#!/usr/bin/env python3
"""
Paired significance test for RQ3 (Multi-Task Learning) on the held-out test set.

For each architecture (nnU-Net, UNETR) the auxiliary ET-volume ordinal head (MTL)
is compared against its single-task baseline (default augmentation). Both models
were evaluated on the *same* 125 held-out cases, so per-case Dice scores are paired.

We run a Wilcoxon signed-rank test (non-parametric, paired) per region (WT/TC/ET),
report the median per-case delta (MTL - baseline) and the Hodges-Lehmann estimate,
and apply Holm-Bonferroni correction across the 3 regions within each architecture.

The point of this analysis is to back the *negative* finding: the MTL aux task does
not significantly change segmentation quality.

Output:
  experiments/mtl_significance.json
  stdout table

Usage:
    python scripts/analyze_mtl_significance.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent

# (label, "treatment" ensemble csv, "reference" ensemble csv)
# Positive delta = treatment better than reference. For MTL-vs-baseline rows the
# reference is the single-task default-aug baseline; for the v2-vs-v1 row the
# reference is the per-patch MTL (EXP-04), so positive = global target better.
COMPARISONS = [
    (
        "nnU-Net MTL v1 (per-patch) vs baseline",
        REPO / "experiments/exp04_nnunet_mtl/test_predictions/ensemble_results.csv",
        REPO / "experiments/ablation/abl_nnunet_default/test_predictions/ensemble_results.csv",
    ),
    (
        "nnU-Net MTL v2 (global) vs baseline",
        REPO / "experiments/exp06_nnunet_mtl_global/test_predictions/ensemble_results.csv",
        REPO / "experiments/ablation/abl_nnunet_default/test_predictions/ensemble_results.csv",
    ),
    (
        "nnU-Net MTL v2 (global) vs v1 (per-patch)",
        REPO / "experiments/exp06_nnunet_mtl_global/test_predictions/ensemble_results.csv",
        REPO / "experiments/exp04_nnunet_mtl/test_predictions/ensemble_results.csv",
    ),
    (
        "nnU-Net MTL v3 (FiLM + uncertainty) vs baseline",
        REPO / "experiments/exp08_nnunet_mtl_v3/test_predictions/ensemble_results.csv",
        REPO / "experiments/ablation/abl_nnunet_default/test_predictions/ensemble_results.csv",
    ),
    (
        "nnU-Net MTL v3 (FiLM + uncertainty) vs v2 (global)",
        REPO / "experiments/exp08_nnunet_mtl_v3/test_predictions/ensemble_results.csv",
        REPO / "experiments/exp06_nnunet_mtl_global/test_predictions/ensemble_results.csv",
    ),
    (
        "UNETR MTL v1 (per-patch) vs baseline",
        REPO / "experiments/exp05_unetr_mtl/test_predictions/ensemble_results.csv",
        REPO / "experiments/ablation/abl_unetr_default/test_predictions/ensemble_results.csv",
    ),
    (
        "UNETR MTL v2 (global) vs baseline",
        REPO / "experiments/exp07_unetr_mtl_global/test_predictions/ensemble_results.csv",
        REPO / "experiments/ablation/abl_unetr_default/test_predictions/ensemble_results.csv",
    ),
]

REGIONS = [("WT", "dice_wt"), ("TC", "dice_tc"), ("ET", "dice_et")]
ALPHA = 0.05


def hodges_lehmann(deltas: np.ndarray) -> float:
    """Hodges-Lehmann estimator: median of pairwise Walsh averages."""
    n = len(deltas)
    walsh = (deltas[:, None] + deltas[None, :])[np.triu_indices(n)] / 2.0
    return float(np.median(walsh))


def holm_bonferroni(pvals: list[float], alpha: float = ALPHA) -> list[bool]:
    """Return per-test reject flags under Holm-Bonferroni step-down control."""
    m = len(pvals)
    order = np.argsort(pvals)
    reject = [False] * m
    for rank, idx in enumerate(order):
        if pvals[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break  # step-down: once we fail, all larger p-values also fail
    return reject


def compare(label: str, mtl_csv: Path, base_csv: Path) -> dict:
    mtl = pd.read_csv(mtl_csv).set_index("case_id").sort_index()
    base = pd.read_csv(base_csv).set_index("case_id").sort_index()

    common = mtl.index.intersection(base.index)
    assert len(common) == len(mtl) == len(base), (
        f"{label}: case mismatch (mtl={len(mtl)}, base={len(base)}, common={len(common)})"
    )
    mtl, base = mtl.loc[common], base.loc[common]

    print(f"\n{'='*78}")
    print(f"  {label}: MTL vs single-task baseline   (n={len(common)} paired cases)")
    print(f"{'='*78}")
    print(f"  {'Region':<8}{'MTL mean':>10}{'Base mean':>11}{'Δ median':>11}"
          f"{'HL est':>10}{'p (Wilcoxon)':>15}{'sig?':>7}")
    print(f"  {'-'*70}")

    region_results = []
    pvals = []
    for region, col in REGIONS:
        d_mtl = mtl[col].to_numpy()
        d_base = base[col].to_numpy()
        deltas = d_mtl - d_base

        # zero_method="wilcox" drops exact ties; guard the all-zero edge case.
        if np.allclose(deltas, 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(d_mtl, d_base, zero_method="wilcox", alternative="two-sided")

        region_results.append({
            "region": region,
            "mtl_mean": float(d_mtl.mean()),
            "baseline_mean": float(d_base.mean()),
            "median_delta": float(np.median(deltas)),
            "hodges_lehmann": hodges_lehmann(deltas),
            "wilcoxon_stat": float(stat) if not np.isnan(stat) else None,
            "p_value": float(p),
            "n_mtl_better": int((deltas > 0).sum()),
            "n_baseline_better": int((deltas < 0).sum()),
            "n_tie": int((deltas == 0).sum()),
        })
        pvals.append(p)

    rejects = holm_bonferroni(pvals)
    for r, rej, p in zip(region_results, rejects, pvals):
        r["p_holm_reject"] = bool(rej)
        sig = "yes" if rej else "ns"
        print(f"  {r['region']:<8}{r['mtl_mean']:>10.4f}{r['baseline_mean']:>11.4f}"
              f"{r['median_delta']:>+11.4f}{r['hodges_lehmann']:>+10.4f}"
              f"{p:>15.4g}{sig:>7}")

    return {"architecture": label, "n_cases": int(len(common)), "regions": region_results}


def main():
    results = [compare(label, mtl, base) for label, mtl, base in COMPARISONS]

    out_path = REPO / "experiments" / "mtl_significance.json"
    payload = {
        "test": "Wilcoxon signed-rank (paired, two-sided)",
        "correction": "Holm-Bonferroni across WT/TC/ET within each architecture",
        "alpha": ALPHA,
        "delta_sign": "positive = MTL better than single-task baseline",
        "comparisons": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'='*78}")
    print("  Interpretation")
    print(f"{'='*78}")
    any_sig = False
    for res in results:
        for r in res["regions"]:
            if r["p_holm_reject"]:
                any_sig = True
                direction = "improves" if r["hodges_lehmann"] > 0 else "degrades"
                print(f"  {res['architecture']} {r['region']}: MTL {direction} "
                      f"segmentation (p={r['p_value']:.4g}, Holm-significant).")
    if not any_sig:
        print("  No region shows a statistically significant difference between MTL and")
        print("  the single-task baseline (all Holm-corrected p > 0.05). The auxiliary")
        print("  ET-volume head does not measurably change segmentation quality.")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
