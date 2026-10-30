"""X5 baseline comparison for the top-30 % uplift.

Reference practice: compare uplift in the top-k segment with the overall uplift (Surry & Radcliffe 2011, §1, Fig. 1).
Uses the stored holdout predictions only (model and scores unchanged). Bootstrap replicates are regenerated exactly
as in holdout_eval.bootstrap (default_rng(20261016), treated draw then control draw, sorted indices) and
checked against the stored uplift_at_30 interval, so every interval below comes from the same 2,000 replicates.
  overall  = mean(Y | W=1) - mean(Y | W=0) on the whole holdout
  top30    = sklift uplift_at_k(k=0.3, by_group) (stored headline)
  diff     = top30 - overall ; ratio = top30 / overall
Output: results/report_baseline.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import holdout_eval as EV  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results/final/holdout_eval"
OUT = ROOT / "results/report_baseline.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ate(y, w):
    return float(y[w == 1].mean() - y[w == 0].mean())


def main() -> None:
    res = json.loads((EVAL / "holdout_eval.json").read_text())
    pp = EVAL / "holdout_predictions.parquet"
    if sha(pp) != res["predictions"]["sha256"]:
        raise RuntimeError("predictions sha256 differs")
    p = pd.read_parquet(pp).sort_values("client_id", kind="mergesort").reset_index(drop=True)
    y, u, w = p.target.to_numpy(int), p.uplift.to_numpy(float), p.treatment_flg.to_numpy(int)
    top = EV.uplift_at_30(y, u, w)
    if abs(top - res["metrics"]["uplift_at_30"]) > 1e-12:
        raise RuntimeError("top-30 % does not reproduce the stored value")
    ov = ate(y, w)

    rng = np.random.default_rng(EV.SEED_BOOT)
    it, ic = np.flatnonzero(w == 1), np.flatnonzero(w == 0)
    bt, bo = np.empty(EV.N_BOOT), np.empty(EV.N_BOOT)
    for b in range(EV.N_BOOT):
        idx = np.sort(np.concatenate([rng.choice(it, size=len(it), replace=True),
                                      rng.choice(ic, size=len(ic), replace=True)]))
        _ = EV.QN.qini_auc(y[idx], u[idx], w[idx])          # keep the identical call sequence (no RNG use)
        bt[b] = EV.uplift_at_30(y[idx], u[idx], w[idx])
        bo[b] = ate(y[idx], w[idx])
    ci = lambda a: [float(x) for x in np.percentile(a, [2.5, 97.5])]  # noqa: E731
    if max(abs(a - b) for a, b in zip(ci(bt), res["metrics"]["uplift_at_30_ci95"])) > 1e-12:
        raise RuntimeError("bootstrap replicates differ from the stored run")
    out = {"script_sha256": sha(Path(__file__).resolve()),
           "inputs": {"holdout_eval.json": sha(EVAL / "holdout_eval.json"), "holdout_predictions.parquet": sha(pp)},
           "reference": "Surry & Radcliffe (2011) Quality measures for uplift models, §1 and Fig. 1",
           "n": int(len(y)), "n_treated": int(w.sum()),
           "overall": {"treated_rate": float(y[w == 1].mean()), "control_rate": float(y[w == 0].mean()),
                       "uplift": ov, "uplift_pp": 100 * ov, "ci95_pp": [100 * x for x in ci(bo)]},
           "top30": {"uplift_pp": 100 * top, "ci95_pp": [100 * x for x in ci(bt)]},
           "top30_minus_overall": {"pp": 100 * (top - ov), "ci95_pp": [100 * x for x in ci(bt - bo)],
                                   "share_of_reps_le_0": float(np.mean(bt - bo <= 0))},
           "top30_over_overall": {"ratio": top / ov, "ci95": ci(bt / bo)},
           "bootstrap": {"n_boot": EV.N_BOOT, "seed": EV.SEED_BOOT,
                         "note": "same replicates as holdout_eval.json (uplift_at_30 interval reproduced within 1e-12)"}}
    if OUT.exists():
        raise RuntimeError(f"{OUT} exists; not overwriting")
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
