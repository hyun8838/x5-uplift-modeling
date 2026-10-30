"""X5 post-evaluation heterogeneity description.

Stored holdout predictions + holdout features only; model and scores unchanged. Groups are cut on the sealed
uplift score over all holdout customers (client_id order, ties broken by mergesort rank as in pandas qcut on rank).
  (1) Decile table  : 10 score deciles (decile 1 = highest score). Per decile n, treated/control response rate,
                      uplift = rate_T - rate_C, bootstrap 95 % interval. Reference: Surry & Radcliffe (2011) Fig. 1;
                      sklift uplift_by_percentile(strategy="overall").
  (2a) GATES        : 4 score quartiles (G1 = highest). Per group uplift, bootstrap 95 % interval, G1 - G4.
                      Reference: Chernozhukov, Demirer, Duflo, Fernandez-Val (2018/2020) "Generic ML inference",
                      simplified to the single train/holdout split (no repeated splitting).
  (2b) CLAN         : feature means in G1 vs G4 and their difference with Welch 95 % interval (normal approx.).
Bootstrap: 2,000 reps, resample within W (same scheme as the holdout evaluation), numpy default_rng(20261018).
Descriptive only: no causal reading of CLAN differences, no new model.
Output: results/post_heterogeneity/{heterogeneity.json, deciles.csv, gates.csv, clan.csv, fig3_deciles.png,
        fig4_gates.png}
"""
from __future__ import annotations

import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results/final/holdout_eval"
OUT = ROOT / "results/post_heterogeneity"
SEED, N_BOOT = 20261018, 2000
plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def groups(u, k):
    """Group 1 = highest score. Equal-size groups on the score rank (stable)."""
    r = np.empty(len(u), int)
    r[np.argsort(-u, kind="mergesort")] = np.arange(len(u))
    return (r * k // len(u)) + 1


def group_uplift(y, w, g, k):
    t = np.array([y[(g == j) & (w == 1)].mean() for j in range(1, k + 1)])
    c = np.array([y[(g == j) & (w == 0)].mean() for j in range(1, k + 1)])
    return t, c, t - c


def main() -> None:
    res = json.loads((EVAL / "holdout_eval.json").read_text())
    pp = EVAL / "holdout_predictions.parquet"
    if sha(pp) != res["predictions"]["sha256"]:
        raise RuntimeError("predictions sha256 differs")
    p = pd.read_parquet(pp).sort_values("client_id", kind="mergesort").reset_index(drop=True)
    y, u, w = p.target.to_numpy(int), p.uplift.to_numpy(float), p.treatment_flg.to_numpy(int)
    hold = FIN.load_holdout_features()
    if list(hold.client_id) != list(p.client_id):
        raise RuntimeError("feature rows not aligned with predictions")
    OUT.mkdir(parents=True, exist_ok=False)

    g10, g4 = groups(u, 10), groups(u, 4)
    rng = np.random.default_rng(SEED)
    it, ic = np.flatnonzero(w == 1), np.flatnonzero(w == 0)
    b10, b4, bdiff = np.empty((N_BOOT, 10)), np.empty((N_BOOT, 4)), np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([rng.choice(it, len(it)), rng.choice(ic, len(ic))])
        b10[b] = group_uplift(y[idx], w[idx], g10[idx], 10)[2]
        b4[b] = group_uplift(y[idx], w[idx], g4[idx], 4)[2]
        bdiff[b] = b4[b, 0] - b4[b, 3]
    pct = lambda a: np.percentile(a, [2.5, 97.5], axis=0)  # noqa: E731

    t10, c10, u10 = group_uplift(y, w, g10, 10)
    lo10, hi10 = pct(b10)
    dec = pd.DataFrame({"decile": range(1, 11), "n": np.bincount(g10)[1:],
                        "n_treated": [int(((g10 == j) & (w == 1)).sum()) for j in range(1, 11)],
                        "score_min": [u[g10 == j].min() for j in range(1, 11)],
                        "score_max": [u[g10 == j].max() for j in range(1, 11)],
                        "rate_treated": t10, "rate_control": c10, "uplift_pp": 100 * u10,
                        "ci_lo_pp": 100 * lo10, "ci_hi_pp": 100 * hi10})
    t4, c4, u4 = group_uplift(y, w, g4, 4)
    lo4, hi4 = pct(b4)
    gates = pd.DataFrame({"group": [f"G{j}" for j in range(1, 5)], "n": np.bincount(g4)[1:],
                          "rate_treated": t4, "rate_control": c4, "uplift_pp": 100 * u4,
                          "ci_lo_pp": 100 * lo4, "ci_hi_pp": 100 * hi4})
    dlo, dhi = pct(bdiff)

    x = LR.prepare_matrix(hold, list(LR.FEATURES_38))
    rows = []
    for col in x.columns:
        a, z = x.loc[g4 == 1, col].astype(float), x.loc[g4 == 4, col].astype(float)
        a, z = a[np.isfinite(a)], z[np.isfinite(z)]
        d = a.mean() - z.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + z.var(ddof=1) / len(z))
        pooled = np.sqrt((a.var(ddof=1) + z.var(ddof=1)) / 2)
        rows.append({"feature": col, "mean_G1": a.mean(), "mean_G4": z.mean(), "median_G1": a.median(),
                     "median_G4": z.median(), "diff": d, "ci_lo": d - 1.96 * se, "ci_hi": d + 1.96 * se,
                     "std_diff": d / pooled if pooled > 0 else np.nan, "n_G1": len(a), "n_G4": len(z)})
    clan = pd.DataFrame(rows).sort_values("std_diff", key=np.abs, ascending=False)

    dec.to_csv(OUT / "deciles.csv", index=False)
    gates.to_csv(OUT / "gates.csv", index=False)
    clan.to_csv(OUT / "clan.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ov = 100 * (y[w == 1].mean() - y[w == 0].mean())
    ax.bar(dec.decile, dec.uplift_pp, color="#C44E52", alpha=0.85)
    ax.errorbar(dec.decile, dec.uplift_pp, yerr=[dec.uplift_pp - dec.ci_lo_pp, dec.ci_hi_pp - dec.uplift_pp],
                fmt="none", ecolor="#333333", capsize=3, lw=1)
    ax.axhline(ov, color="#333333", ls="--", lw=1, label=f"holdout 전체 {ov:+.2f}%p")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("점수 10분위 (1 = 점수 최상위 10%)")
    ax.set_ylabel("처치군 − 대조군 반응률 (%p)")
    ax.set_title("그림 3. 점수 10분위별 관찰 uplift (holdout, 95% 부트스트랩 구간)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_deciles.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.bar(gates.group, gates.uplift_pp, color="#4C72B0", alpha=0.85)
    ax.errorbar(range(4), gates.uplift_pp, yerr=[gates.uplift_pp - gates.ci_lo_pp, gates.ci_hi_pp - gates.uplift_pp],
                fmt="none", ecolor="#333333", capsize=4, lw=1)
    ax.axhline(ov, color="#333333", ls="--", lw=1, label=f"holdout 전체 {ov:+.2f}%p")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel("점수 4분위 그룹 (G1 = 점수 최상위 25%)")
    ax.set_ylabel("처치군 − 대조군 반응률 (%p)")
    ax.set_title("그림 4. GATES: 점수 4분위 그룹별 관찰 uplift", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_gates.png", dpi=150)
    plt.close(fig)

    out = {"script_sha256": sha(Path(__file__).resolve()),
           "inputs": {"holdout_predictions.parquet": sha(pp), "holdout_eval.json": sha(EVAL / "holdout_eval.json")},
           "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "scheme": "within W, percentile 95 %"},
           "overall_uplift_pp": ov,
           "gates_G1_minus_G4_pp": {"diff": 100 * (u4[0] - u4[3]), "ci95": [100 * dlo, 100 * dhi]},
           "files": {f.name: sha(f) for f in sorted(OUT.iterdir())}}
    (OUT / "heterogeneity.json").write_text(json.dumps(out, indent=2) + "\n")
    pd.set_option("display.width", 200)
    print(dec.round(4).to_string(index=False))
    print(gates.round(4).to_string(index=False))
    print(json.dumps(out["gates_G1_minus_G4_pp"]))
    print(clan.head(15).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
