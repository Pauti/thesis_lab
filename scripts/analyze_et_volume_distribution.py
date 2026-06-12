"""
RQ3 feasibility check: ET volume distribution for ordinal classification.

Counts ET voxels (label==3) per case across BraTS 2021, then evaluates
whether 2-3 ordinal classes can be cleanly separated (balanced, non-trivial).

Breakpoints are derived ONLY from the ablation training pool (1126 cases),
never from the 125 held-out test cases, to avoid information leakage.
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np

RAW = Path("/data/paul_daten/nnunet/nnUNet_raw/Dataset001_BraTS/labelsTr")
PREP = Path("/data/paul_daten/nnunet/nnUNet_preprocessed/Dataset001_BraTS")
OUT = Path("/home/paul/repositories/thesis_lab/experiments/et_volume_distribution.json")
ET_LABEL = 3  # remapped: bg=0, NCR=1, ED=2, ET=3

test_ids = set(json.loads((PREP / "test_set.json").read_text())["case_ids"])
abl = json.loads((PREP / "splits_ablation.json").read_text())
pool_ids = set()
for f in abl:
    pool_ids |= set(f["train"]) | set(f["val"])

records = {}
files = sorted(RAW.glob("*.nii.gz"))
for i, fp in enumerate(files):
    cid = fp.name.replace(".nii.gz", "")
    arr = np.asanyarray(nib.load(str(fp)).dataobj)
    et = int(np.count_nonzero(arr == ET_LABEL))
    tumor = int(np.count_nonzero(arr > 0))
    records[cid] = {"et": et, "tumor": tumor}
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(files)}", flush=True)

OUT.write_text(json.dumps(records, indent=0))
print(f"saved {len(records)} -> {OUT}", flush=True)


def summarize(name, ids):
    et = np.array([records[c]["et"] for c in ids if c in records])
    absent = int((et == 0).sum())
    present = et[et > 0]
    print(f"\n===== {name} (n={len(et)}) =====")
    print(f"ET absent (==0): {absent} ({absent/len(et)*100:.1f}%)")
    print(f"ET present:      {len(present)} ({len(present)/len(et)*100:.1f}%)")
    print(f"ET voxels  min={et.min()} max={et.max()} median={int(np.median(et))} "
          f"mean={et.mean():.0f}")
    qs = [10, 25, 33, 50, 66, 67, 75, 90]
    print("percentiles (all cases):")
    for q in qs:
        print(f"  p{q:>2} = {np.percentile(et, q):>8.0f}")
    return et


pool_et = summarize("ABLATION TRAIN POOL", pool_ids)
summarize("TEST SET", test_ids)
summarize("FULL DATASET", records.keys())

# --- candidate ordinal binnings, breakpoints from POOL only ---
print("\n\n##### CANDIDATE BINNINGS (breakpoints from pool quantiles) #####")


def report_bins(et_all, edges, labels):
    # edges: interior cut points; bins = [-inf, e0, e1, ..., +inf]
    full = np.array([records[c]["et"] for c in records])
    cuts = [-1] + list(edges) + [np.inf]
    print(f"\n-- {len(labels)} classes, edges={[int(e) for e in edges]} --")
    for name, dat, tag in [("pool", et_all, pool_ids), ("test", None, test_ids),
                           ("full", full, records.keys())]:
        arr = np.array([records[c]["et"] for c in tag if c in records])
        counts = []
        for k in range(len(labels)):
            lo, hi = cuts[k], cuts[k + 1]
            counts.append(int(((arr > lo) & (arr <= hi)).sum()))
        pct = [f"{c}({c/len(arr)*100:.0f}%)" for c in counts]
        print(f"   {name:>4} n={len(arr):>4}: " + "  ".join(
            f"{labels[k]}={pct[k]}" for k in range(len(labels))))


# 3-class tertiles among ALL pool cases (absent folds into lowest)
t33, t67 = np.percentile(pool_et, [33.3, 66.6])
report_bins(pool_et, [t33, t67], ["low", "mid", "high"])

# 2-class median split
med = np.percentile(pool_et, 50)
report_bins(pool_et, [med], ["low", "high"])

# 3-class: absent as its own class? (shows why it fails)
report_bins(pool_et, [0.5, np.percentile(pool_et[pool_et > 0], 50)],
            ["absent", "small", "large"])
