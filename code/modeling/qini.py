"""X5 standard normalized Qini and curve coordinates for figures.

qini_auc = scikit-uplift 0.5.1 qini_auc_score(y, uplift, w, negative_effect=True):
  (AUC(model curve) - AUC(random line)) / (AUC(perfect curve, negative effects included) - AUC(random line)).
Tied scores are aggregated as whole blocks inside sklift.qini_curve, so row order does not matter.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import auc
from sklift.metrics import perfect_qini_curve, qini_auc_score, qini_curve


def _arrays(y, uplift, w):
    y, uplift, w = np.asarray(y).astype(int), np.asarray(uplift, dtype=float), np.asarray(w).astype(int)
    assert y.shape == uplift.shape == w.shape and y.ndim == 1
    assert set(np.unique(w)) == {0, 1} and set(np.unique(y)) <= {0, 1}
    assert np.isfinite(uplift).all()
    return y, uplift, w


def qini_auc(y, uplift, w) -> float:
    y, uplift, w = _arrays(y, uplift, w)
    with warnings.catch_warnings():
        # sklift 0.5.1 calls sklearn's deprecated stable_cumsum (sklearn 1.9): FutureWarning only, value unaffected
        warnings.filterwarnings("ignore", message=".*stable_cumsum.*", category=FutureWarning)
        v = float(qini_auc_score(y, uplift, w, negative_effect=True))
    assert np.isfinite(v), "degenerate Qini normalization"
    return v


def qini_curve_coords(y, uplift, w, normalize: bool = True) -> dict:
    """Coordinates of model curve, perfect curve (negative_effect=True) and random line.

    normalize=True divides x by n and y by n (fraction targeted, incremental responders per customer),
    which leaves the normalized Qini unchanged and lets fold curves of different size share one axis.
    """
    y, uplift, w = _arrays(y, uplift, w)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*stable_cumsum.*", category=FutureWarning)
        xa, ya = qini_curve(y, uplift, w)
        xp, yp = perfect_qini_curve(y, w, negative_effect=True)
    xr, yr = np.array([0.0, xp[-1]]), np.array([0.0, yp[-1]])
    n = float(len(y)) if normalize else 1.0
    return {"x": np.asarray(xa, float) / n, "y": np.asarray(ya, float) / n,
            "x_perfect": np.asarray(xp, float) / n, "y_perfect": np.asarray(yp, float) / n,
            "x_random": xr / n, "y_random": yr / n, "n": int(len(y))}


def qini_from_coords(c: dict) -> float:
    base = auc(c["x_random"], c["y_random"])
    return float((auc(c["x"], c["y"]) - base) / (auc(c["x_perfect"], c["y_perfect"]) - base))
