"""Learner recovery on synthetic data with known tau (7 learners x 2 engines).

Pass criteria (applied to scenarios 'mixed' and 'positive', separate test sample):
  Spearman(pred, true tau) >= SPEARMAN_MIN, sign agreement >= SIGN_MIN, Qini(pred) > QINI_MIN.
Supplementary (not pass/fail): scenario 'mixed' with treatment share 0.3 (checks the constant-e formulas);
reports the same metrics plus mean(pred) vs true ATE.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import spearmanr

import synth
import learners as LR
import qini as QN

SPEARMAN_MIN = 0.3
SIGN_MIN = 0.6
QINI_MIN = 0.0
N_TRAIN, N_TEST = 20_000, 20_000
FIXED_PARAMS = dict(depth=3, learning_rate=0.05, n_estimators=300, reg_lambda=1.0,
                    min_leaf_hessian=1.0, subsample=0.8, colsample=0.8)
SEED = 20261009
SCENARIOS = {"mixed": dict(scenario="mixed", p_treat=0.5),
             "positive": dict(scenario="positive", p_treat=0.5)}
SUPPLEMENTARY = {"mixed_e0.3": dict(scenario="mixed", p_treat=0.3)}


def _evaluate(name, spec, log, judged: bool):
    x, w, y, _ = synth.make(N_TRAIN, seed=101, scenario=spec["scenario"], p_treat=spec["p_treat"])
    xt, wt, yt, tt = synth.make(N_TEST, seed=202, scenario=spec["scenario"], p_treat=spec["p_treat"])
    cols = list(x.columns)
    X, XT = LR.prepare_matrix(x, cols), LR.prepare_matrix(xt, cols)
    oracle = QN.qini_auc(yt, tt, wt)
    info = {"scenario": name, "train_treated_share": float(w.mean()), "test_true_ate": float(tt.mean()),
            "test_share_tau_negative": float((tt < 0).mean()), "oracle_qini_true_tau": oracle, "rows": []}
    log(f"  [{name}] treated share {w.mean():.3f}, true ATE {tt.mean():+.4f}, "
        f"tau<0 share {(tt < 0).mean():.3f}, oracle Qini {oracle:.4f}")
    log(f"  {'engine':9s} {'learner':10s} {'spearman':>8s} {'sign':>6s} {'qini':>7s} "
        f"{'mean_pred':>9s} {'sec':>5s} pass")
    for eng in LR.ENGINES:
        for ln in LR.LEARNERS:
            t0 = time.time()
            m = LR.fit(ln, X, w, y, FIXED_PARAMS, eng, SEED, n_jobs=4)
            u = m.predict_uplift(XT)
            sec = time.time() - t0
            rho = float(spearmanr(u, tt)[0])
            sign = float(np.mean(np.sign(u) == np.sign(tt)))
            q = QN.qini_auc(yt, u, wt)
            ok = (rho >= SPEARMAN_MIN) and (sign >= SIGN_MIN) and (q > QINI_MIN) if judged else None
            row = {"engine": eng, "learner": ln, "spearman": rho, "sign_agreement": sign, "qini": q,
                   "mean_pred": float(u.mean()), "seconds": sec, "n_submodels": m.n_submodels(), "pass": ok}
            info["rows"].append(row)
            log(f"  {eng:9s} {ln:10s} {rho:8.3f} {sign:6.3f} {q:7.4f} {u.mean():+9.4f} {sec:5.1f} "
                f"{'' if ok is None else ('PASS' if ok else 'FAIL')}")
    return info


def lgb_min_data_probe(log) -> dict:
    """Informational (no pass/fail): does min_leaf_hessian change LightGBM predictions when
    min_data_in_leaf stays at the LightGBM default 20, versus min_data_in_leaf = 1?
    T-learner (logistic) and the DR final regression (squared error) on synthetic 'mixed' data."""
    x, w, y, _ = synth.make(N_TRAIN, seed=101)
    xt, _, _, tt = synth.make(5000, seed=202)
    X, XT = LR.prepare_matrix(x, list(x.columns)), LR.prepare_matrix(xt, list(x.columns))
    hess = (0.01, 0.1, 1.0, 4.0, 20.0, 50.0)
    res = {}
    for ln in ("t", "dr"):
        for mdl in (None, 1):
            preds = {}
            for h in hess:
                p = dict(FIXED_PARAMS, depth=6, n_estimators=200, min_leaf_hessian=h)
                preds[h] = LR.fit(ln, X, w, y, p, "lightgbm", SEED, n_jobs=4, lgb_min_data_in_leaf=mdl).predict_uplift(XT)
            same = [h for h in hess if np.array_equal(preds[h], preds[hess[0]])]
            key = f"{ln}/min_data_in_leaf={'default20' if mdl is None else mdl}"
            res[key] = {"hessian_values_identical_to_0.01": same,
                        "spearman_vs_tau": {str(h): float(spearmanr(preds[h], tt)[0]) for h in hess}}
            log(f"  probe {key}: min_leaf_hessian values giving predictions identical to 0.01: {same}; "
                f"spearman " + ", ".join(f"{h}:{res[key]['spearman_vs_tau'][str(h)]:.3f}" for h in hess))
    return res


def run(log) -> dict:
    out = {"criteria": {"spearman_min": SPEARMAN_MIN, "sign_agreement_min": SIGN_MIN, "qini_gt": QINI_MIN},
           "fixed_params": FIXED_PARAMS, "seed": SEED, "n_train": N_TRAIN, "n_test": N_TEST,
           "n_jobs": 4, "scenarios": {}, "supplementary": {}}
    for name, spec in SCENARIOS.items():
        out["scenarios"][name] = _evaluate(name, spec, log, judged=True)
    for name, spec in SUPPLEMENTARY.items():
        out["supplementary"][name] = _evaluate(name, spec, log, judged=False)
    fails = [(s, r["engine"], r["learner"]) for s, v in out["scenarios"].items() for r in v["rows"] if not r["pass"]]
    # interface / reproducibility checks
    x, w, y, _ = synth.make(3000, seed=7)
    X = LR.prepare_matrix(x, list(x.columns))
    rep = {}
    for eng in LR.ENGINES:
        for ln in LR.LEARNERS:
            a = LR.fit(ln, X, w, y, FIXED_PARAMS, eng, 5).predict_uplift(X)
            b = LR.fit(ln, X, w, y, FIXED_PARAMS, eng, 5).predict_uplift(X)
            rep[f"{eng}/{ln}"] = bool(np.array_equal(a, b))
    # preprocessing is row-wise: transforming a subset equals subsetting the transform
    full = LR.prepare_matrix(x, list(x.columns))
    sub = LR.prepare_matrix(x.iloc[100:200], list(x.columns))
    rowwise = bool(full.iloc[100:200].equals(sub))
    dummies_ok = bool(set(full.columns) >= {"gender_F", "gender_M", "gender_U"}
                      and (full[["gender_F", "gender_M", "gender_U"]].sum(axis=1) == 1).all())
    nan_kept = bool(full["age"].isna().sum() == x["age"].isna().sum() and (full["age"].min() < 0))
    out["checks"] = {"same_seed_same_prediction": rep, "preprocessing_rowwise": rowwise,
                     "gender_three_dummies": dummies_ok, "age_raw_nan_kept": nan_kept}
    log(f"  same seed -> identical predictions: {sum(rep.values())}/{len(rep)}; preprocessing row-wise {rowwise}; "
        f"gender 3 dummies {dummies_ok}; age raw + NaN kept {nan_kept}")
    out["lgb_min_data_in_leaf_probe"] = lgb_min_data_probe(log)
    out["pass"] = (not fails) and all(rep.values()) and rowwise and dummies_ok and nan_kept
    out["failed"] = fails
    return out
