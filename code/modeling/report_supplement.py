"""X5 report supplement (descriptive numbers quoted in the report).

Derived from existing result files only; nothing is refit and the sealed holdout_eval/ folder is not written.
(1) holdout top-30 % response rate per group (by_group, same cut rule as sklift uplift_at_k: per group
    n_g = int(len * 0.3), uplift argsort mergesort reversed, client_id lexicographic order) -> must reproduce
    holdout_eval.json uplift_at_30 within 1e-12 and n_cut.
(2) qini_table.csv descriptive: sd min/max, mean range of ranks 1-20, #(engine, learner) pairs with full > compact.
Output: results/report_supplement.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results/final/holdout_eval"
SEL = ROOT / "results/selection"
OUT = ROOT / "results/report_supplement.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    res = json.loads((EVAL / "holdout_eval.json").read_text())
    pred_path = EVAL / "holdout_predictions.parquet"
    if sha(pred_path) != res["predictions"]["sha256"]:
        raise RuntimeError("holdout predictions sha256 differs from holdout_eval.json")
    p = pd.read_parquet(pred_path).sort_values("client_id", kind="mergesort").reset_index(drop=True)
    a1 = {}
    for g, name in ((1, "treated"), (0, "control")):
        s = p[p.treatment_flg == g]
        n_cut = int(len(s) * 0.3)
        order = np.argsort(s.uplift.to_numpy(), kind="mergesort")[::-1][:n_cut]
        y = s.target.to_numpy()[order]
        if n_cut != res["metrics"]["boundary_ties"][name]["n_cut"]:
            raise RuntimeError("n_cut differs")
        a1[name] = {"n_group": int(len(s)), "n_cut": n_cut, "responders": int(y.sum()), "rate": float(y.mean())}
    diff = a1["treated"]["rate"] - a1["control"]["rate"]
    if abs(diff - res["metrics"]["uplift_at_30"]) > 1e-12:
        raise RuntimeError("top-30 % difference does not reproduce uplift_at_30")
    a1["difference"] = diff

    qt = pd.read_csv(SEL / "qini_table.csv", float_precision="round_trip")
    top20 = qt[qt["rank"] <= 20]["mean"]
    piv = qt.pivot_table(index=["engine", "learner"], columns="track", values="mean")
    a3 = {"sd_min": float(qt.sd.min()), "sd_max": float(qt.sd.max()),
          "rank1_20_mean_min": float(top20.min()), "rank1_20_mean_max": float(top20.max()),
          "rank1_20_mean_width": float(top20.max() - top20.min()),
          "pairs_full_gt_compact": int((piv["full"] > piv["compact"]).sum()), "pairs_total": int(len(piv)),
          "optimism_mean_min": float(pd.read_csv(SEL / "optimism_table.csv", float_precision="round_trip")
                                     .optimism_mean.min()),
          "optimism_mean_max": float(pd.read_csv(SEL / "optimism_table.csv", float_precision="round_trip")
                                     .optimism_mean.max())}
    out = {"script_sha256": sha(Path(__file__).resolve()),
           "inputs": {"holdout_eval.json": sha(EVAL / "holdout_eval.json"),
                      "holdout_predictions.parquet": sha(pred_path),
                      "qini_table.csv": sha(SEL / "qini_table.csv"),
                      "optimism_table.csv": sha(SEL / "optimism_table.csv")},
           "holdout_top30_by_group": a1, "selection_descriptive": a3}
    if OUT.exists():
        raise RuntimeError(f"{OUT} exists; not overwriting")
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
