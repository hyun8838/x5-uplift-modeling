"""X5 selection-stage metric check: rank the 28 candidates by the outer-fold mean of Qini, Uplift AUC and
uplift@30 % (sklift uplift_at_k, k=0.3, by_group). Train outer-fold predictions only; holdout is not read.
The selection itself is unchanged (outer mean Qini); this table shows whether the other metrics agree.
Output: {out_dir}/metric_check.csv, metric_check.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklift.metrics import uplift_at_k, uplift_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qini as QN  # noqa: E402
import tune as TU  # noqa: E402

ROOT = TU.ROOT
SEL = ROOT / "results/selection"
STUDIES = ROOT / "results/nested/studies"
METRICS = ("qini", "uplift_auc", "uplift_at_30")


def main(out_dir: Path) -> None:
    qt = pd.read_csv(SEL / "qini_table.csv", float_precision="round_trip").set_index("candidate")
    tr = TU.load_train_only()
    truth = pd.DataFrame({"client_id": np.asarray(tr["ids"]).astype(str), "w": np.asarray(tr["w"]),
                          "y": np.asarray(tr["y"])}).set_index("client_id")
    rows = []
    for cand in qt.index:
        for k in range(5):
            p = pd.read_parquet(STUDIES / f"{cand}__outer{k}.outer_pred.parquet")
            p["client_id"] = p.client_id.astype(str)
            p = p.sort_values("client_id", kind="mergesort").reset_index(drop=True)
            t = truth.loc[p.client_id]
            y, w, u = t.y.to_numpy(), t.w.to_numpy(), p.uplift.to_numpy()
            q = QN.qini_auc(y, u, w)
            assert abs(q - qt.loc[cand, f"q_outer{k}"]) <= 1e-9, (cand, k)   # same Qini as the selection table
            rows.append({"candidate": cand, "k": k, "qini": q, "uplift_auc": uplift_auc_score(y, u, w),
                         "uplift_at_30": uplift_at_k(y, u, w, strategy="by_group", k=0.3)})
    df = pd.DataFrame(rows)
    agg = df.groupby("candidate")[list(METRICS)].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    for m in METRICS:
        agg[f"rank_{m}"] = agg[f"{m}_mean"].rank(ascending=False, method="min").astype(int)
    agg = agg.sort_values("rank_qini")
    out_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_dir / "metric_check.csv")
    selected = json.loads((SEL / "selection_result.json").read_text())["selected"]
    fold_best = {m: df.pivot(index="k", columns="candidate", values=m).idxmax(axis=1).tolist() for m in METRICS}
    summary = {"selected": selected,
               "selected_ranks": {m: int(agg.loc[selected, f"rank_{m}"]) for m in METRICS},
               "uplift_at_30_pp": {"max": 100 * agg.uplift_at_30_mean.max(),
                                   "selected": 100 * agg.loc[selected, "uplift_at_30_mean"],
                                   "median_fold_sd": 100 * agg.uplift_at_30_std.median()},
               "spearman_between_candidate_means": agg[[f"{m}_mean" for m in METRICS]].corr("spearman").round(4).to_dict(),
               "best_candidate_per_fold": fold_best}
    (out_dir / "metric_check.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/post_metric_check")
    main(ap.parse_args().out_dir)
