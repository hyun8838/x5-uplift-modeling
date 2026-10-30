"""X5 uplift KL filter (Zhao, Zhang, Harinen, Yung 2020, arXiv:2005.03447).

score(feature) = sum_b (n_b / N) * KL( P_T(Y | b) || P_C(Y | b) )
  - b: bins of the feature computed from the INPUT data only (call it on a fold's training rows).
    numeric : interior edges = unique 10/20/.../90% quantiles of the non-missing values (numpy 'linear'),
              bin = number of edges strictly below the value (right-closed intervals, like pd.qcut);
              duplicate edges are dropped, so heavily tied features get fewer bins.
              missing values form one extra bin.
    categorical (gender): one bin per category (missing would be its own bin).
  - KL for a binary outcome: p log(p/q) + (1-p) log((1-p)/(1-q)), p = P_T(Y=1|b), q = P_C(Y=1|b).
  - smoothing: p and q are clipped to [EPS, 1-EPS], EPS = 1e-6 (same constant causalml's filter uses for q).
    A bin with no treated or no control rows contributes 0 (no contrast is observable there).
- Top k = 10 by score; ties broken by feature name (ascending) so the result is deterministic.
- gender is ranked as ONE variable (38-feature basis, before dummy expansion).
- No forced inclusion (no RFM override).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BINS = 10
EPS = 1e-6
TOP_K = 10
CATEGORICAL = ("gender",)


def bin_codes(values: pd.Series, n_bins: int = N_BINS, categorical: bool = False) -> np.ndarray:
    """Integer bin code per row. Missing -> its own code (-1)."""
    if categorical:
        s = values.astype(object)
        miss = s.isna().to_numpy()
        cats = sorted(set(s[~miss]))
        lookup = {c: i for i, c in enumerate(cats)}
        codes = np.array([lookup.get(v, -1) if not m else -1 for v, m in zip(s, miss)], dtype=np.int64)
        return codes
    x = values.to_numpy(dtype=float)
    miss = np.isnan(x)
    codes = np.full(len(x), -1, dtype=np.int64)
    if (~miss).any():
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges = np.unique(np.quantile(x[~miss], qs))
        codes[~miss] = np.searchsorted(edges, x[~miss], side="left")
    return codes


def _bernoulli_kl(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def kl_score(values: pd.Series, w: np.ndarray, y: np.ndarray, n_bins: int = N_BINS,
             categorical: bool = False, eps: float = EPS) -> tuple[float, int]:
    codes = bin_codes(values, n_bins, categorical)
    df = pd.DataFrame({"b": codes, "w": w, "y": y})
    g = df.groupby(["b", "w"]).y.agg(["sum", "size"]).unstack("w", fill_value=0)
    n_t, n_c = g[("size", 1)].to_numpy(float), g[("size", 0)].to_numpy(float)
    s_t, s_c = g[("sum", 1)].to_numpy(float), g[("sum", 0)].to_numpy(float)
    n_b = n_t + n_c
    ok = (n_t > 0) & (n_c > 0)
    kl = np.zeros(len(n_b))
    kl[ok] = _bernoulli_kl(s_t[ok] / n_t[ok], s_c[ok] / n_c[ok], eps)
    return float(np.sum(n_b / len(df) * kl)), int(len(n_b))


def kl_scores(frame: pd.DataFrame, w, y, features: list[str], n_bins: int = N_BINS,
              eps: float = EPS) -> pd.DataFrame:
    w = np.asarray(w).astype(int)
    y = np.asarray(y).astype(int)
    assert len(frame) == len(w) == len(y) and set(np.unique(w)) == {0, 1}
    rows = []
    for f in features:
        s, nb = kl_score(frame[f], w, y, n_bins, categorical=f in CATEGORICAL, eps=eps)
        rows.append({"feature": f, "score": s, "n_bins": nb})
    out = pd.DataFrame(rows)
    # deterministic order: score desc, then feature name asc
    out = out.sort_values(["score", "feature"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def select_top_k(frame: pd.DataFrame, w, y, features: list[str], k: int = TOP_K) -> list[str]:
    """Top-k features in rank order. Uses only the rows passed in."""
    table = kl_scores(frame, w, y, features)
    return table.feature.head(k).tolist()
