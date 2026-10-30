"""KL filter checks on synthetic data (linear-probability generator so prognostic features are
prognostic on the probability scale, i.e. they do not modify P_T - P_C).

Pass criteria, over REPS replicate data sets, k = 10 of 28 features:
  1. effect modifiers mod_0, mod_1, gender are in the top 10 in every replicate.
  2. every prognostic feature's mean rank is worse (larger) than every modifier's mean rank.
Unit checks: ties broken by name; missingness bin carries signal; gender = 3 category bins, one variable;
deterministic repeat; uses only rows passed in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import kl_filter as KL

REPS = 5
N = 20_000
N_NOISE = 21
MODIFIERS = ["mod_0", "mod_1", "gender"]
PROGNOSTIC = ["prog_0", "prog_1", "prog_2", "prog_3"]


def make_lp(n: int, seed: int):
    rng = np.random.default_rng(seed)
    d = {f"prog_{i}": rng.normal(size=n) for i in range(4)}
    d["mod_0"] = rng.normal(size=n)
    d["mod_1"] = rng.uniform(size=n)
    d["gender"] = rng.choice(np.array(["F", "M", "U"], dtype=object), size=n, p=[0.37, 0.17, 0.46])
    for i in range(N_NOISE):
        d[f"noise_{i:02d}"] = rng.normal(size=n)
    x = pd.DataFrame(d)
    x.loc[rng.uniform(size=n) < 0.10, "prog_3"] = np.nan
    base = (0.45 + 0.12 * np.tanh(x.prog_0) + 0.08 * np.tanh(x.prog_1) - 0.08 * np.tanh(x.prog_2)
            + 0.05 * np.tanh(x.prog_3.fillna(0))).to_numpy()
    tau = (0.10 * np.tanh(1.5 * x.mod_0) + 0.12 * (x.mod_1 - 0.5) + 0.06 * (x.gender == "U") - 0.03).to_numpy()
    w = (rng.uniform(size=n) < 0.5).astype(int)
    y = (rng.uniform(size=n) < base + w * tau).astype(int)
    return x, w, y


def run(log) -> dict:
    feats = None
    ranks, selected = [], []
    for r in range(REPS):
        x, w, y = make_lp(N, seed=1000 + r)
        feats = list(x.columns)
        t = KL.kl_scores(x, w, y, feats)
        ranks.append(t.set_index("feature")["rank"])
        selected.append(set(t.feature.head(KL.TOP_K)))
        if r == 0:
            log("  replicate 0 ranking (top 12): " + ", ".join(
                f"{f}={s:.5f}" for f, s in zip(t.feature.head(12), t.score.head(12))))
    R = pd.concat(ranks, axis=1)
    mean_rank = R.mean(axis=1)
    sel_rate = pd.Series({f: np.mean([f in s for s in selected]) for f in feats})
    c1 = all(sel_rate[m] == 1.0 for m in MODIFIERS)
    c2 = bool(mean_rank[PROGNOSTIC].min() > mean_rank[MODIFIERS].max())
    noise = [f for f in feats if f.startswith("noise_")]
    log(f"  selection rate (top {KL.TOP_K}, {REPS} reps): modifiers "
        + ", ".join(f"{m}={sel_rate[m]:.1f}" for m in MODIFIERS)
        + f" | prognostic mean {sel_rate[PROGNOSTIC].mean():.2f} | noise mean {sel_rate[noise].mean():.2f}")
    log("  mean rank: modifiers " + ", ".join(f"{m}={mean_rank[m]:.1f}" for m in MODIFIERS)
        + " | prognostic " + ", ".join(f"{p}={mean_rank[p]:.1f}" for p in PROGNOSTIC)
        + f" | noise mean {mean_rank[noise].mean():.1f}")

    # unit checks
    x, w, y = make_lp(8000, seed=77)
    x2 = x[["mod_0", "noise_00"]].copy()
    x2["zz_copy"] = x2["mod_0"]
    x2["aa_copy"] = x2["mod_0"]
    t = KL.kl_scores(x2, w, y, ["zz_copy", "noise_00", "mod_0", "aa_copy"])
    tie_ok = t.feature.head(3).tolist() == ["aa_copy", "mod_0", "zz_copy"]
    # missingness-only signal: feature is NaN exactly where an injected effect lives
    rng = np.random.default_rng(5)
    n = 8000
    flag = rng.uniform(size=n) < 0.3
    ww = (rng.uniform(size=n) < 0.5).astype(int)
    yy = (rng.uniform(size=n) < 0.4 + 0.25 * ww * flag).astype(int)
    xm = pd.DataFrame({"miss_signal": np.where(flag, np.nan, 1.0), "pure_noise": rng.normal(size=n)})
    tm = KL.kl_scores(xm, ww, yy, ["miss_signal", "pure_noise"])
    miss_ok = tm.feature.iloc[0] == "miss_signal"
    nb = int(KL.kl_scores(x, w, y, ["gender"]).n_bins.iloc[0])
    gender_ok = nb == 3
    sel_a = KL.select_top_k(x, w, y, list(x.columns))
    sel_b = KL.select_top_k(x, w, y, list(x.columns))
    det_ok = sel_a == sel_b and len(sel_a) == KL.TOP_K
    half = KL.select_top_k(x.iloc[:4000], w[:4000], y[:4000], list(x.columns))
    half2 = KL.select_top_k(x.iloc[:4000].reset_index(drop=True), w[:4000], y[:4000], list(x.columns))
    local_ok = half == half2   # result is a function of the passed rows only (index-free)
    bins_dec = int(KL.kl_scores(x, w, y, ["mod_0"]).n_bins.iloc[0])
    bins_miss = int(KL.kl_scores(x, w, y, ["prog_3"]).n_bins.iloc[0])
    log(f"  unit: tie->name order {tie_ok}; missing bin carries signal {miss_ok}; gender bins {nb}; "
        f"deterministic {det_ok}; row-local {local_ok}; bins mod_0={bins_dec} prog_3(with NaN)={bins_miss}")
    out = {"reps": REPS, "n": N, "k": KL.TOP_K, "n_features": len(feats), "eps": KL.EPS, "n_bins": KL.N_BINS,
           "selection_rate": sel_rate.round(3).to_dict(), "mean_rank": mean_rank.round(2).to_dict(),
           "criteria": {"modifiers_always_selected": c1, "prognostic_rank_worse_than_modifiers": c2},
           "unit": {"tie_by_name": tie_ok, "missing_bin_signal": miss_ok, "gender_bins": nb,
                    "deterministic": det_ok, "row_local": local_ok, "bins_continuous": bins_dec,
                    "bins_with_missing": bins_miss}}
    out["pass"] = bool(c1 and c2 and tie_ok and miss_ok and gender_ok and det_ok and local_ok
                       and bins_dec == 10 and bins_miss == 11)
    return out
