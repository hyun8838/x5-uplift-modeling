"""Tuning-frame checks on synthetic data (no real data).

Pass criteria, N_TRIALS = 4:
  A. fresh run completes; result has inner_best, outer_qini, optimism, per-fold inner Qini.
  B. second fresh run (new SQLite, same seeds) -> identical trial parameter sequence, best params, outer Qini.
  C. interrupted run (2 trials, then a killed RUNNING trial, then a crashing trial) resumed to 4 trials ->
     same finished-trial parameter sequence and best params as A; stale trial marked FAIL.
  D. reuse refused (FingerprintMismatch) for changed input (one y flipped), changed config (sampler seed),
     changed code hash.
  E. compact track: every inner selection equals KL top-10 recomputed on that inner training part only,
     outer selection equals KL on the outer training part; 10 features each.
  F. prune flag runs (MedianPruner, 1st inner fold) and records COMPLETE/PRUNED counts.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

import synth
import kl_filter as KL
import tune as TU

N = 6000
N_TRIALS = 4


def make_fold_data(seed: int = 31, k: int = 0) -> TU.FoldData:
    x, w, y, _ = synth.make(N, seed=seed, n_noise=8)
    ids = np.array([f"c{i:06d}" for i in range(N)])
    key = 2 * w + y
    outer = list(StratifiedKFold(5, shuffle=True, random_state=20261009).split(x, key))
    tr, te = outer[k]
    inner = np.full(len(tr), -1)
    for j, (_, va) in enumerate(StratifiedKFold(3, shuffle=True, random_state=20261010 + k).split(tr, key[tr])):
        inner[va] = j
    return TU.FoldData(x, w, y, ids, list(x.columns), tr, te, inner)


def _finished_params(db: str, name: str):
    st = optuna.load_study(study_name=name, storage=f"sqlite:///{db}")
    fin = [t for t in st.get_trials(deepcopy=False)
           if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    return [t.params for t in fin], [t.value for t in fin], st


def run(log, scratch: Path) -> dict:
    out = {}
    d = make_fold_data()
    base = scratch / "tune"
    cases = [TU.TuneConfig("xgboost", "t", "full", 0, N_TRIALS, sampler_seed=123, model_seed=7),
             TU.TuneConfig("lightgbm", "lai", "compact", 0, N_TRIALS, sampler_seed=123, model_seed=7)]
    all_ok = True
    for cfg in cases:
        tag = cfg.study_name
        t0 = time.time()
        ra = TU.run_study(d, cfg, base / "A", out_dir=base / "A_out")
        ta = time.time() - t0
        pa, va, _ = _finished_params(ra["storage"], tag)
        a_ok = all(k in ra for k in ("inner_best", "outer_qini", "optimism_inner_minus_outer")) \
            and len(ra["inner_best_fold_qini"]) == 3 and ra["n_trials"]["complete"] == N_TRIALS
        rb = TU.run_study(d, cfg, base / "B")
        pb, vb, _ = _finished_params(rb["storage"], tag)
        b_ok = pa == pb and va == vb and ra["best_params"] == rb["best_params"] and ra["outer_qini"] == rb["outer_qini"]

        # C: interrupted + resumed
        cdir = base / "C"
        r1 = TU.run_study(d, cfg, cdir, stop_after=2)
        st = optuna.load_study(study_name=tag, storage=f"sqlite:///{r1['db']}")
        st.ask()                                     # a trial left RUNNING (killed process)
        crashed = False
        try:
            TU.run_study(d, cfg, cdir, fail_at_trial=3)   # next trial number is 3 -> crashes
        except RuntimeError as e:
            crashed = "simulated interruption" in str(e)
        rc = TU.run_study(d, cfg, cdir)
        pc, vc, stc = _finished_params(rc["storage"], tag)
        states = [t.state.name for t in stc.get_trials(deepcopy=False)]
        c_ok = (crashed and pc == pa and vc == va and rc["best_params"] == ra["best_params"]
                and rc["outer_qini"] == ra["outer_qini"] and states.count("FAIL") == 2 and rc["resumed"])

        # D: fingerprint mismatch refusals (reuse the A database)
        refusals = {}
        d_bad = make_fold_data()
        d_bad.y = d_bad.y.copy()
        d_bad.y[0] = 1 - d_bad.y[0]
        for label, kwargs in (("input", dict(d=d_bad, cfg=cfg)),
                              ("config", dict(d=d, cfg=TU.TuneConfig(cfg.engine, cfg.learner, cfg.track, 0, N_TRIALS,
                                                                       sampler_seed=999, model_seed=7))),
                              ("code", dict(d=d, cfg=cfg, code_hash="0" * 64))):
            try:
                TU.run_study(kwargs["d"], kwargs["cfg"], base / "A", code_hash=kwargs.get("code_hash"))
                refusals[label] = False
            except TU.FingerprintMismatch as e:
                refusals[label] = label in str(e)
        d_ok = all(refusals.values())

        # E: compact-track selection locality
        e_ok = None
        if cfg.track == "compact":
            e_ok = True
            for j in range(3):
                tr = d.train_idx[d.inner_fold != j]
                exp = KL.select_top_k(d.frame.iloc[tr], d.w[tr], d.y[tr], d.features)
                e_ok &= ra["selected_inner"][str(j)] == exp and len(exp) == 10
            exp_o = KL.select_top_k(d.frame.iloc[d.train_idx], d.w[d.train_idx], d.y[d.train_idx], d.features)
            e_ok &= ra["selected_outer"] == exp_o and len(exp_o) == 10
        ok = a_ok and b_ok and c_ok and d_ok and (e_ok in (None, True))
        all_ok &= ok
        out[tag] = {"A_complete": a_ok, "B_same_seed_identical": b_ok, "C_resume_identical": c_ok,
                    "C_trial_states": states, "D_refusals": refusals, "E_compact_selection_local": e_ok,
                    "inner_best": ra["inner_best"], "outer_qini": ra["outer_qini"],
                    "optimism": ra["optimism_inner_minus_outer"], "best_params": ra["best_params"],
                    "seconds_run_A": ta, "trial_seconds_mean": ra["trial_seconds_mean"],
                    "selected_outer": ra["selected_outer"], "pass": ok}
        log(f"  {tag}: A {a_ok} | B same-seed identical {b_ok} | C resume identical {c_ok} (states {states}) | "
            f"D refusals {refusals} | E {e_ok} | inner best {ra['inner_best']:.4f} outer {ra['outer_qini']:.4f} | "
            f"{ta:.1f}s for {N_TRIALS} trials")

    # F: pruning option
    cfg_p = TU.TuneConfig("xgboost", "t", "full", 0, 8, sampler_seed=123, model_seed=7, prune=True,
                          pruner_startup_trials=2)
    rp = TU.run_study(d, cfg_p, base / "P")
    f_ok = rp["n_trials"]["complete"] + rp["n_trials"]["pruned"] == 8 and rp["n_trials"]["complete"] >= 1
    out["prune_option"] = {"n_trials": rp["n_trials"], "pass": f_ok}
    log(f"  prune option (8 trials, MedianPruner startup 2): {rp['n_trials']} ok {f_ok}")
    out["pass"] = bool(all_ok and f_ok)
    out["n"] = N
    out["n_trials"] = N_TRIALS
    return out
