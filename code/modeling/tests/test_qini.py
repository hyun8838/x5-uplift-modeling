"""Qini wrapper checks.

Pass criteria:
  1. wrapper value == sklift qini_auc_score(negative_effect=True) called directly (exact) on 3 score vectors.
  2. Qini recomputed from the curve coordinates == wrapper (|diff| < 1e-12).
  3. perfect ranking score (y*w - y*(1-w)) gives 1 (|diff| < 1e-9).
  4. constant score gives 0 (|q| < 1e-12); true-tau ranking > 0.
  5. 200 random score permutations: |mean| < 3 * SE and |mean| < 0.01.
  6. row order invariance (shuffled rows, same value within 1e-12).
"""
from __future__ import annotations

import warnings

import numpy as np
from sklift.metrics import qini_auc_score

import synth
import qini as QN


def run(log) -> dict:
    _, w, y, tau = synth.make(20_000, seed=303)
    rng = np.random.default_rng(11)
    scores = {"true_tau": tau, "noisy_tau": tau + rng.normal(0, 0.2, len(tau)), "random": rng.normal(size=len(tau))}
    eq = {}
    for k, s in scores.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            direct = float(qini_auc_score(y, s, w, negative_effect=True))
        eq[k] = {"wrapper": QN.qini_auc(y, s, w), "direct": direct}
    c1 = all(v["wrapper"] == v["direct"] for v in eq.values())
    coords_diff = max(abs(QN.qini_from_coords(QN.qini_curve_coords(y, s, w)) - QN.qini_auc(y, s, w))
                      for s in scores.values())
    c2 = coords_diff < 1e-12
    perfect = QN.qini_auc(y, y * w - y * (1 - w), w)
    c3 = abs(perfect - 1) < 1e-9
    const = QN.qini_auc(y, np.zeros(len(y)), w)
    c4 = abs(const) < 1e-12 and eq["true_tau"]["wrapper"] > 0
    perms = np.array([QN.qini_auc(y, rng.permutation(len(y)).astype(float), w) for _ in range(200)])
    mean, se = perms.mean(), perms.std(ddof=1) / np.sqrt(len(perms))
    c5 = abs(mean) < 3 * se and abs(mean) < 0.01
    idx = rng.permutation(len(y))
    c6 = abs(QN.qini_auc(y[idx], scores["noisy_tau"][idx], w[idx]) - QN.qini_auc(y, scores["noisy_tau"], w)) < 1e-12
    log("  wrapper vs direct: " + ", ".join(f"{k} {v['wrapper']:.6f}/{v['direct']:.6f}" for k, v in eq.items()))
    log(f"  coords recompute max diff {coords_diff:.2e}; perfect {perfect:.6f}; constant {const:.2e}; "
        f"random perms mean {mean:+.5f} (SE {se:.5f}, sd {perms.std(ddof=1):.4f}); order invariant {c6}")
    return {"equal_to_sklift": eq, "coords_max_diff": coords_diff, "perfect": perfect, "constant": const,
            "random_perm_mean": float(mean), "random_perm_se": float(se), "random_perm_sd": float(perms.std(ddof=1)),
            "criteria": {"c1_equal": c1, "c2_coords": c2, "c3_perfect": c3, "c4_const_and_oracle": c4,
                         "c5_random_zero": bool(c5), "c6_order": bool(c6)},
            "pass": bool(c1 and c2 and c3 and c4 and c5 and c6)}
