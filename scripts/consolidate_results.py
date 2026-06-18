#!/usr/bin/env python3
"""
Consolidate ALL experiment metrics into one canonical master table.

Pure AGGREGATION — reads the per-case CSVs that each experiment already produced
and computes summary statistics. It never re-computes Dice or HD95 (those come
from utils/metrics.py / utils/hd95.py via the inference + HD95 scripts); this
script only summarises what is on disk.

Per region (WT/TC/ET) it reports:
  Dice : mean, std, median, min, max          (always available)
  HD95 : mean, std, n_contributing            (NaN-aware; only where computed)

HD95 is missing for the UNETR experiments until their GPU re-inference is run and
for the 5-fold-CV RQ1 baselines (no predictions on disk). Those rows show blank
HD95 and fill in automatically once the *_hd95.csv files appear — just re-run.

Output:
  experiments/results_master.csv    one row per experiment, all stats (wide)
  experiments/results_master.json   same data, structured

Usage:
    python scripts/consolidate_results.py
    python scripts/consolidate_results.py --status   # only print availability
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"
REGIONS = ["wt", "tc", "et"]


# ── Experiment registry ─────────────────────────────────────────────────────────
# (experiment_id, rq, arch, condition, eval_protocol, dice_csv, hd95_csv_or_None)
# hd95_csv may point at a file that does not exist yet; it is checked at runtime.

def _abl(arch: str, tier: str) -> tuple:
    base = EXP / "ablation" / f"abl_{arch}_{tier}" / "test_predictions"
    return (f"abl_{arch}_{tier}", "RQ2", arch, tier, "3fold_ensemble",
            base / "ensemble_results.csv", base / "ensemble_results_hd95.csv")


def _mtl(exp_id: str, arch: str, condition: str) -> tuple:
    base = EXP / exp_id / "test_predictions"
    return (exp_id, "RQ3", arch, condition, "3fold_ensemble",
            base / "ensemble_results.csv", base / "ensemble_results_hd95.csv")


REGISTRY: list[tuple] = [
    # RQ1 — architecture comparison, 5-fold CV on 1251 cases.
    # HD95 sources: exp02 from disk (compute_hd95_exp02_cv_from_disk.py);
    # exp03 inline in its re-inference CSV (infer_exp03.py, after GPU re-run).
    ("exp02_nnunet", "RQ1", "nnunet", "baseline", "5fold_cv",
     EXP / "exp02_nnunet/analysis/all_folds_brats_dice.csv",
     EXP / "exp02_nnunet/analysis/all_folds_hd95.csv"),
    ("exp03_unetr", "RQ1", "unetr", "baseline", "5fold_cv",
     EXP / "exp03_unetr/analysis/all_folds_brats_dice_nifti.csv",
     EXP / "exp03_unetr/analysis/all_folds_brats_dice_nifti.csv"),
    # RQ2 — augmentation ablation, 3-fold ensemble on 125 held-out cases
    _abl("nnunet", "disabled"), _abl("nnunet", "reduced"),
    _abl("nnunet", "default"), _abl("nnunet", "extended"),
    _abl("unetr", "disabled"), _abl("unetr", "reduced"),
    _abl("unetr", "default"), _abl("unetr", "extended"),
    # RQ3 — multi-task learning, 3-fold ensemble on 125 held-out cases
    _mtl("exp04_nnunet_mtl", "nnunet", "mtl_v1_perpatch"),
    _mtl("exp06_nnunet_mtl_global", "nnunet", "mtl_v2_global"),
    _mtl("exp08_nnunet_mtl_v3", "nnunet", "mtl_v3_fusion_uncert"),
    _mtl("exp05_unetr_mtl", "unetr", "mtl_v1_perpatch"),
    _mtl("exp07_unetr_mtl_global", "unetr", "mtl_v2_global"),
]


def _dice_stats(s: pd.Series, prefix: str) -> dict:
    v = s.dropna()
    return {
        f"{prefix}_mean":   float(v.mean()),
        f"{prefix}_std":    float(v.std(ddof=1)),
        f"{prefix}_median": float(v.median()),
        f"{prefix}_min":    float(v.min()),
        f"{prefix}_max":    float(v.max()),
    }


def _hd95_stats(s: pd.Series, prefix: str) -> dict:
    v = s.dropna()  # NaN = region absent in pred or GT → excluded
    n = int(v.count())
    return {
        f"{prefix}_mean": float(v.mean()) if n else None,
        f"{prefix}_std":  float(v.std(ddof=1)) if n > 1 else None,
        f"{prefix}_n":    n,
    }


def process(entry: tuple) -> dict | None:
    exp_id, rq, arch, condition, protocol, dice_csv, hd95_csv = entry

    if not dice_csv.exists():
        print(f"  SKIP {exp_id}: Dice CSV not found ({dice_csv.name}) — experiment not ready")
        return None

    dice_df = pd.read_csv(dice_csv)
    # HD95 source may exist but not yet carry hd95_* columns (e.g. exp03 before
    # its GPU re-inference) → treat as unavailable until the columns are present.
    hd95_df = None
    if hd95_csv and hd95_csv.exists():
        cand = pd.read_csv(hd95_csv)
        if all(f"hd95_{r}" in cand.columns for r in REGIONS):
            hd95_df = cand

    row: dict = {
        "experiment_id":  exp_id,
        "rq":             rq,
        "arch":           arch,
        "condition":      condition,
        "eval_protocol":  protocol,
        "n_cases":        len(dice_df),
        "hd95_available": hd95_df is not None,
    }
    for r in REGIONS:
        row.update(_dice_stats(dice_df[f"dice_{r}"], f"dice_{r}"))
    for r in REGIONS:
        if hd95_df is not None:
            row.update(_hd95_stats(hd95_df[f"hd95_{r}"], f"hd95_{r}"))
        else:
            row[f"hd95_{r}_mean"] = row[f"hd95_{r}_std"] = None
            row[f"hd95_{r}_n"] = 0
    return row


def print_table(rows: list[dict]) -> None:
    h = (f"{'experiment':<26}{'RQ':>4}{'arch':>8}{'cond':>16}{'n':>5}"
         f"{'Dice WT':>9}{'Dice TC':>9}{'Dice ET':>9}"
         f"{'HD WT':>7}{'HD TC':>7}{'HD ET':>7}")
    print("\n" + "-" * len(h) + f"\n{h}\n" + "-" * len(h))
    for r in rows:
        def hd(reg):
            v = r[f"hd95_{reg}_mean"]
            return f"{v:.2f}" if v is not None else "  -"
        print(f"{r['experiment_id']:<26}{r['rq']:>4}{r['arch']:>8}{r['condition']:>16}"
              f"{r['n_cases']:>5}"
              f"{r['dice_wt_mean']:>9.4f}{r['dice_tc_mean']:>9.4f}{r['dice_et_mean']:>9.4f}"
              f"{hd('wt'):>7}{hd('tc'):>7}{hd('et'):>7}")
    print("-" * len(h))

    missing = [r["experiment_id"] for r in rows if not r["hd95_available"]]
    if missing:
        print(f"\nHD95 still missing for {len(missing)} experiment(s) — re-run after GPU jobs:")
        print("  " + ", ".join(missing))


def main() -> None:
    ap = argparse.ArgumentParser(description="Consolidate all experiment metrics (aggregation only)")
    ap.add_argument("--status", action="store_true", help="only print availability, do not write files")
    args = ap.parse_args()

    rows = [r for r in (process(e) for e in REGISTRY) if r is not None]
    print_table(rows)

    if args.status:
        return

    df = pd.DataFrame(rows)
    csv_path = EXP / "results_master.csv"
    json_path = EXP / "results_master.json"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote: {csv_path}\nWrote: {json_path}")


if __name__ == "__main__":
    main()
