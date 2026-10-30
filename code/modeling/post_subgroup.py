"""X5 subgroup uplift by pre-chosen features.
  - spend_90d, purchases_90d, category_count, days_since_issue: tertiles of the TRAIN 70 % feature values (holdout values not used)
  - express_points_used: 0 / non-zero
  - age: 18-34 / 35-49 / 50-64 / 65-100, other values reported as "범위 밖"
Per subgroup: treated rate - control rate on holdout, bootstrap within W 2,000 reps (seed 20261019), percentile 95 %.
Per variable: last level - first level (age: 65-100 minus 18-34). Descriptive, no multiplicity adjustment, no model.
Output: results/post_subgroup/{subgroups.csv, contrasts.csv, subgroup.json, fig5_subgroups.png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final as FIN  # noqa: E402
import learners as LR  # noqa: E402
import tune as TU  # noqa: E402
from post_heterogeneity import EVAL, ROOT, sha  # noqa: E402

OUT = ROOT / "results/post_subgroup"
SEED, N_BOOT = 20261019, 2000
TERT = {"spend_90d": "90일 구매액", "purchases_90d": "90일 구매 횟수", "category_count": "구매 카테고리 수",
        "days_since_issue": "카드 발급 후 경과일"}
plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def train_tertiles() -> dict:
    split = pd.read_parquet(TU.SPLIT_PATH, columns=["client_id", "split"])
    tr = set(split.loc[split.split == "train", "client_id"])
    f = pd.read_parquet(TU.FEATURES_PATH, columns=["client_id", *TERT])
    f = f.loc[f.client_id.isin(tr)]
    if len(f) != 140027:
        raise RuntimeError("train rows")
    return {c: [float(q) for q in f[c].quantile([1 / 3, 2 / 3])] for c in TERT}


def fmt(v):
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.1f}"


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
    cuts = train_tertiles()

    labels = {}  # variable -> (korean name, ordered level names, level index per row)
    for c, name in TERT.items():
        lo, hi = cuts[c]
        v = x[c].to_numpy(float)
        idx = np.where(v <= lo, 0, np.where(v <= hi, 1, 2))
        labels[c] = (name, [f"낮음 (≤ {fmt(lo)})", f"중간", f"높음 (> {fmt(hi)})"], idx)
    e = x["express_points_used"].to_numpy(float)
    labels["express_points_used"] = ("익스프레스 포인트 사용", ["0", "0 아님"], (e != 0).astype(int))
    a = x["age"].to_numpy(float)
    ai = np.full(len(a), 4)
    for k, (l, h) in enumerate([(18, 34), (35, 49), (50, 64), (65, 100)]):
        ai[(a >= l) & (a <= h)] = k
    labels["age"] = ("나이", ["18–34", "35–49", "50–64", "65–100", "범위 밖 (해석하지 않음)"], ai)
    first_last = {c: (0, len(v[1]) - 1) for c, v in labels.items()}
    first_last["age"] = (0, 3)

    def uplift(yy, ww, mask):
        return yy[mask & (ww == 1)].mean() - yy[mask & (ww == 0)].mean()

    rng = np.random.default_rng(SEED)
    it, ic = np.flatnonzero(w == 1), np.flatnonzero(w == 0)
    keys = [(c, j) for c, v in labels.items() for j in range(len(v[1]))]
    boot = np.empty((N_BOOT, len(keys)))
    for b in range(N_BOOT):
        idx = np.concatenate([rng.choice(it, len(it)), rng.choice(ic, len(ic))])
        yb, wb = y[idx], w[idx]
        for k, (c, j) in enumerate(keys):
            boot[b, k] = uplift(yb, wb, labels[c][2][idx] == j)
    lo95, hi95 = np.percentile(boot, [2.5, 97.5], axis=0)

    rows = []
    for k, (c, j) in enumerate(keys):
        m = labels[c][2] == j
        rows.append({"feature": c, "name": labels[c][0], "level": j, "level_name": labels[c][1][j], "n": int(m.sum()),
                     "n_treated": int((m & (w == 1)).sum()), "rate_treated": y[m & (w == 1)].mean(),
                     "rate_control": y[m & (w == 0)].mean(), "uplift_pp": 100 * uplift(y, w, m),
                     "ci_lo_pp": 100 * lo95[k], "ci_hi_pp": 100 * hi95[k]})
    sub = pd.DataFrame(rows)
    con = []
    for c, (f0, f1) in first_last.items():
        k0, k1 = keys.index((c, f0)), keys.index((c, f1))
        d = boot[:, k1] - boot[:, k0]
        con.append({"feature": c, "name": labels[c][0], "contrast": f"{labels[c][1][f1]} − {labels[c][1][f0]}",
                    "diff_pp": sub.uplift_pp[k1] - sub.uplift_pp[k0],
                    "ci_lo_pp": 100 * np.percentile(d, 2.5), "ci_hi_pp": 100 * np.percentile(d, 97.5)})
    con = pd.DataFrame(con)

    OUT.mkdir(parents=True, exist_ok=False)
    sub.to_csv(OUT / "subgroups.csv", index=False)
    con.to_csv(OUT / "contrasts.csv", index=False)

    ov = 100 * (y[w == 1].mean() - y[w == 0].mean())
    fig, ax = plt.subplots(figsize=(8, 8.4))
    ypos, yl, yv = [], [], 0
    for c in labels:
        s = sub[sub.feature == c]
        for r in s.itertuples():
            ax.errorbar(r.uplift_pp, yv, xerr=[[r.uplift_pp - r.ci_lo_pp], [r.ci_hi_pp - r.uplift_pp]], fmt="o",
                        color="#4C72B0", ecolor="#333333", capsize=3, ms=5)
            ypos.append(yv)
            yl.append(f"{r.name}: {r.level_name}  (n={r.n:,})")
            yv -= 1
        yv -= 0.6
    ax.axvline(ov, color="#333333", ls="--", lw=1, label=f"holdout 전체 {ov:+.2f}%p")
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(yl, fontsize=8.5)
    ax.set_xlabel("처치군 − 대조군 반응률 (%p), 95% 부트스트랩 구간")
    ax.set_title("그림 5. 변수로 나눈 집단별 처치효과 (holdout)", fontsize=11)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_subgroups.png", dpi=150)
    plt.close(fig)

    out = {"script_sha256": sha(Path(__file__).resolve()),
           "inputs": {"holdout_predictions.parquet": sha(pp), "features": sha(TU.FEATURES_PATH),
                      "split": sha(TU.SPLIT_PATH)},
           "train_tertile_cuts": cuts, "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "scheme": "within W, percentile"},
           "overall_uplift_pp": ov, "files": {f.name: sha(f) for f in sorted(OUT.iterdir())}}
    (OUT / "subgroup.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    pd.set_option("display.width", 220)
    print(sub.drop(columns=["feature"]).round(3).to_string(index=False))
    print(con.round(3).to_string(index=False))
    print(json.dumps(cuts))
    print("sha", sha(OUT / "subgroup.json"), sha(Path(__file__).resolve()))


if __name__ == "__main__":
    main()
