"""X5 subgroup uplift: supplementary analyses for the three stated limits.
Same groups and the same bootstrap (seed 20261019, same draw order) as post_subgroup.py; the stored 95 %
intervals are reproduced first, else stop.
  (1) headroom scale  : (rate_T - rate_C) / (1 - rate_C) per subgroup and per last-minus-first contrast
  (2) interaction LPM : Y ~ 1 + W + group dummies + W x group dummies (6 variables jointly, first level = reference)
                        reported: W x last-level coefficients (age: 65-100); plus spend_90d x purchases_90d 3x3 cells
  (3) Bonferroni 99.17 % : percentile 0.4167 / 99.5833 for each family of 6 contrasts
Output: results/post_subgroup_limits/{scale.csv, contrasts_all.csv, cross_3x3.csv, limits.json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final as FIN  # noqa: E402
import learners as LR  # noqa: E402
import post_subgroup as SG  # noqa: E402
from post_heterogeneity import EVAL, ROOT, sha  # noqa: E402

OUT = ROOT / "results/post_subgroup_limits"
VARS = ["spend_90d", "purchases_90d", "category_count", "days_since_issue", "express_points_used", "age"]
LAST = {"spend_90d": 2, "purchases_90d": 2, "category_count": 2, "days_since_issue": 2, "express_points_used": 1, "age": 3}


def build_levels(x: pd.DataFrame, cuts: dict) -> dict:
    lv = {}
    for c in SG.TERT:
        lo, hi = cuts[c]
        v = x[c].to_numpy(float)
        lv[c] = np.where(v <= lo, 0, np.where(v <= hi, 1, 2))
    lv["express_points_used"] = (x["express_points_used"].to_numpy(float) != 0).astype(int)
    a = x["age"].to_numpy(float)
    ai = np.full(len(a), 4)
    for k, (l, h) in enumerate([(18, 34), (35, 49), (50, 64), (65, 100)]):
        ai[(a >= l) & (a <= h)] = k
    lv["age"] = ai
    return lv


def design(w: np.ndarray, lv: dict) -> tuple[np.ndarray, list]:
    cols, names = [np.ones(len(w)), w.astype(float)], ["const", "W"]
    for c in VARS:
        for j in range(1, int(lv[c].max()) + 1):
            d = (lv[c] == j).astype(float)
            cols += [d, d * w]
            names += [f"{c}={j}", f"W:{c}={j}"]
    return np.column_stack(cols), names


def main() -> None:
    res = json.loads((EVAL / "holdout_eval.json").read_text())
    pp = EVAL / "holdout_predictions.parquet"
    if sha(pp) != res["predictions"]["sha256"]:
        raise RuntimeError("predictions sha256 differs")
    p = pd.read_parquet(pp).sort_values("client_id", kind="mergesort").reset_index(drop=True)
    hold = FIN.load_holdout_features()
    if list(hold.client_id) != list(p.client_id):
        raise RuntimeError("rows not aligned")
    y, w = p.target.to_numpy(int), p.treatment_flg.to_numpy(int)
    x = LR.prepare_matrix(hold, list(LR.FEATURES_38))
    nan_holdout = {c: int(x[c].isna().sum()) for c in VARS}
    cuts = SG.train_tertiles()
    lv = build_levels(x, cuts)
    keys = [(c, j) for c in VARS for j in range(int(lv[c].max()) + 1)]
    X, names = design(w, lv)
    wi = {c: names.index(f"W:{c}={LAST[c]}") for c in VARS}
    cross = (lv["spend_90d"] * 3 + lv["purchases_90d"])

    def stats(yy, ww, ll, cc, XX):
        up, hr = [], []
        for c, j in keys:
            m = ll[c] == j
            t, k = yy[m & (ww == 1)].mean(), yy[m & (ww == 0)].mean()
            up.append(t - k)
            hr.append((t - k) / (1 - k))
        beta = np.linalg.lstsq(XX, yy.astype(float), rcond=None)[0]
        cr = [yy[(cc == q) & (ww == 1)].mean() - yy[(cc == q) & (ww == 0)].mean() for q in range(9)]
        return np.array(up), np.array(hr), beta, np.array(cr)

    up0, hr0, b0, cr0 = stats(y, w, lv, cross, X)
    rng = np.random.default_rng(SG.SEED)
    it, ic = np.flatnonzero(w == 1), np.flatnonzero(w == 0)
    B = SG.N_BOOT
    bu, bh, bb, bc = np.empty((B, len(keys))), np.empty((B, len(keys))), np.empty((B, len(names))), np.empty((B, 9))
    for b in range(B):
        idx = np.concatenate([rng.choice(it, len(it)), rng.choice(ic, len(ic))])
        bu[b], bh[b], bb[b], bc[b] = stats(y[idx], w[idx], {c: v[idx] for c, v in lv.items()}, cross[idx], X[idx])

    stored = pd.read_csv(ROOT / "results/post_subgroup/subgroups.csv")
    lo, hi = np.percentile(bu, [2.5, 97.5], axis=0)
    if not (np.allclose(100 * lo, stored.ci_lo_pp, atol=1e-9) and np.allclose(100 * hi, stored.ci_hi_pp, atol=1e-9)
            and np.allclose(100 * up0, stored.uplift_pp, atol=1e-9)):
        raise RuntimeError("stored subgroup intervals not reproduced")

    hlo, hhi = np.percentile(bh, [2.5, 97.5], axis=0)
    scale = stored[["name", "level_name", "n", "rate_control", "uplift_pp"]].copy()
    scale["headroom_pct"], scale["h_ci_lo"], scale["h_ci_hi"] = 100 * hr0, 100 * hlo, 100 * hhi

    rows = []
    for c in VARS:
        k0, k1 = keys.index((c, 0)), keys.index((c, LAST[c]))
        for kind, est, rep in [("pp", up0[k1] - up0[k0], bu[:, k1] - bu[:, k0]),
                               ("headroom_pct", hr0[k1] - hr0[k0], bh[:, k1] - bh[:, k0]),
                               ("lpm_interaction_pp", b0[wi[c]], bb[:, wi[c]])]:
            q = 100 * np.percentile(rep, [2.5, 97.5, 2.5 / 6, 100 - 2.5 / 6])
            rows.append({"feature": c, "measure": kind, "est": 100 * est, "ci95_lo": q[0], "ci95_hi": q[1],
                         "ciB_lo": q[2], "ciB_hi": q[3]})
    con = pd.DataFrame(rows)
    stc = pd.read_csv(ROOT / "results/post_subgroup/contrasts.csv")
    pp_rows = con[con.measure == "pp"].reset_index(drop=True)
    if not np.allclose(pp_rows.ci95_lo, stc.ci_lo_pp, atol=1e-9):
        raise RuntimeError("stored contrasts not reproduced")

    clo, chi = np.percentile(bc, [2.5, 97.5], axis=0)
    names3 = ["낮음", "중간", "높음"]
    cr = pd.DataFrame({"spend_90d": [names3[q // 3] for q in range(9)], "purchases_90d": [names3[q % 3] for q in range(9)],
                       "n": [int((cross == q).sum()) for q in range(9)], "uplift_pp": 100 * cr0,
                       "ci_lo_pp": 100 * clo, "ci_hi_pp": 100 * chi})

    OUT.mkdir(parents=True, exist_ok=False)
    scale.to_csv(OUT / "scale.csv", index=False)
    con.to_csv(OUT / "contrasts_all.csv", index=False)
    cr.to_csv(OUT / "cross_3x3.csv", index=False)
    out = {"script_sha256": sha(Path(__file__).resolve()),
           "inputs": {"holdout_predictions.parquet": sha(pp)}, "reproduced_stored_intervals": True,
           "nan_holdout": nan_holdout, "lpm_columns": names, "bootstrap": {"n_boot": B, "seed": SG.SEED},
           "files": {f.name: sha(f) for f in sorted(OUT.iterdir())}}
    (OUT / "limits.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    pd.set_option("display.width", 220)
    print(scale.round(3).to_string(index=False))
    print(con.round(3).to_string(index=False))
    print(cr.round(3).to_string(index=False))
    print("nan", nan_holdout)
    print("sha", sha(OUT / "limits.json"), sha(Path(__file__).resolve()))


if __name__ == "__main__":
    main()
