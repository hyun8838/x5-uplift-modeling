"""X5 Optuna TPE tuning frame for one (engine x learner x track, outer fold k) study.

- Objective: mean standard Qini over the 3 inner folds of outer fold k (inner folds from data/splits).
- Track 'compact': the KL filter (top 10) is recomputed on each inner training part; the final outer evaluation
  recomputes it on the whole outer-k training part. Track 'full': all 38 features.
  (Selections do not depend on hyperparameters, so they are computed once per study and reused by every trial.)
- Sequential study (n_jobs=1), TPE sampler with fixed seed. The sampler RNG is re-seeded before every trial
  from (sampler_seed, number of finished trials), so an interrupted-and-resumed study proposes exactly the same
  parameters as an uninterrupted one (given the same completed history).
- SQLite storage per study (one file per study -> no write contention between parallel study processes).
  Resume: RUNNING trials left by a killed process are marked FAIL, then trials are added until
  n_trials COMPLETE/PRUNED trials exist.
- Fingerprint = {input, config, code}. Stored in study user_attrs. Any difference -> FingerprintMismatch (no reuse).
- Pruning (stop after the 1st inner fold, MedianPruner) exists only as an option flag (`prune`, default False).
  Studies run without pruning, 40 trials per study (7 learners x 2 engines x 2 tracks = 28 candidates x 5 outer folds).
- Diagnostics saved: inner best value, outer Qini, their difference (tuning optimism), per-fold inner Qini.
- Real-data loader (`load_train_only`) returns train rows only and asserts that no holdout client_id is present;
  target bytes of holdout rows are never parsed. Unit tests (tests) use synthetic data only.

CLI (real data, nested tuning stage; the unit tests do not call it):
  python modeling/tune.py --engine xgboost --learner t --track full --outer 0 --n-trials 40 \
      --storage-dir <dir> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.trial import TrialState

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kl_filter as KL  # noqa: E402
import learners as LR  # noqa: E402
import qini as QN  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = [ROOT / "modeling" / f for f in ("learners.py", "kl_filter.py", "qini.py", "tune.py")]
N_INNER = 3
TRACKS = ("full", "compact")

# search space (both engines, common meaning)
SEARCH_SPACE = {
    "depth": ("int", 1, 6, False),
    "learning_rate": ("float", 0.01, 0.3, True),
    "n_estimators": ("int", 50, 1000, False),
    "reg_lambda": ("float", 0.001, 10.0, True),
    "min_leaf_hessian": ("float", 0.01, 50.0, True),
    "subsample": ("float", 0.6, 1.0, False),
    "colsample": ("float", 0.6, 1.0, False),
}
assert tuple(SEARCH_SPACE) == LR.PARAM_KEYS
# Passed explicitly to the TPESampler constructor. In Optuna 5.0.0 multivariate=None means "multivariate TPE for
# single-objective studies" (TPESampler._is_multivariate), so these single-objective (maximize) studies use
# multivariate TPE.
SAMPLER_SETTINGS = {"class": "TPESampler", "n_startup_trials": 10, "n_ei_candidates": 24,
                    "multivariate": None, "multivariate_effective_single_objective": True,
                    "constant_liar": True, "per_trial_reseed": "seed + 7919*n_finished"}

# real-data inputs (split outputs)
FEATURES_PATH = ROOT / "data/processed/features_full38.parquet"
SPLIT_PATH = ROOT / "data/splits/split.parquet"
INNER_PATH = ROOT / "data/splits/inner_folds.parquet"
LABELS_PATH = ROOT / "data/og/retailhero-uplift/data/uplift_train.csv"
EXPECTED_SHA = {
    FEATURES_PATH.name: "77049b8c7031d1ad628e334933d8044e31ddf50b7d8f7b2196dba11929763721",
    SPLIT_PATH.name: "a210d7b4da32400941d801e72b25eecaeef34ddd84ee37fa0a6c1cdffa9d47da",
    INNER_PATH.name: "6371e2b33d9a4d1ccc0ce8b11d525bb506e8b25ce60220d2075ef5607f14d07a",
    LABELS_PATH.name: "ae3dc4c17e98426fa4233fa03421328d259b8720e56ad0380b260ca2edd19bd4",
}
EXPECTED_TRAIN_ROWS = 140_027


class FingerprintMismatch(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- config / data
@dataclass(frozen=True)
class TuneConfig:
    engine: str
    learner: str
    track: str
    outer_fold: int
    n_trials: int
    sampler_seed: int
    model_seed: int
    top_k: int = KL.TOP_K
    prune: bool = False
    pruner_startup_trials: int = 5
    lgb_min_data_in_leaf: int | None = LR.LGB_MIN_DATA_IN_LEAF
    n_jobs: int = 1

    def __post_init__(self):
        assert self.engine in LR.ENGINES and self.learner in LR.LEARNERS and self.track in TRACKS
        assert self.n_trials >= 1 and self.n_jobs >= 1

    @property
    def study_name(self) -> str:
        return f"{self.track}__{self.engine}__{self.learner}__outer{self.outer_fold}"


@dataclass
class FoldData:
    """One outer fold. frame holds raw features (gender as string); positions index into frame."""
    frame: pd.DataFrame
    w: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    features: list
    train_idx: np.ndarray            # outer-k training rows
    test_idx: np.ndarray             # outer-k evaluation rows
    inner_fold: np.ndarray           # aligned with train_idx, values 0..N_INNER-1
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.frame)
        assert len(self.w) == len(self.y) == len(self.ids) == n
        assert len(np.intersect1d(self.train_idx, self.test_idx)) == 0
        assert len(self.inner_fold) == len(self.train_idx)
        assert set(np.unique(self.inner_fold)) == set(range(N_INNER))
        assert set(self.features) <= set(self.frame.columns)


def input_fingerprint(d: FoldData) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(list(d.frame.columns)).encode())
    h.update(pd.util.hash_pandas_object(d.frame, index=False).to_numpy().tobytes())
    for a in (np.asarray(d.w, np.int64), np.asarray(d.y, np.int64), np.asarray(d.train_idx, np.int64),
              np.asarray(d.test_idx, np.int64), np.asarray(d.inner_fold, np.int64)):
        h.update(a.tobytes())
    h.update("\x00".join(map(str, d.ids)).encode())
    h.update(json.dumps(list(d.features)).encode())
    return h.hexdigest()


def config_fingerprint(cfg: TuneConfig) -> str:
    return _sha_json({"config": asdict(cfg), "search_space": SEARCH_SPACE, "sampler": SAMPLER_SETTINGS,
                      "kl": {"n_bins": KL.N_BINS, "eps": KL.EPS}, "crossfit_folds": LR.CROSSFIT_FOLDS,
                      "n_inner": N_INNER})


def code_fingerprint() -> str:
    import lightgbm, sklearn, sklift, xgboost  # noqa: E401
    files = {p.name: sha256_file(p) for p in CODE_FILES}
    versions = {m.__name__: m.__version__ for m in (np, pd, optuna, xgboost, lightgbm, sklearn, sklift)}
    return _sha_json({"files": files, "versions": versions, "python": platform.python_version(),
                      "machine": platform.machine(), "platform": platform.platform()})


# ----------------------------------------------------------------------------- sampler
class DeterministicTPESampler(TPESampler):
    """TPE whose RNG is re-seeded before each trial from (seed, #finished trials)."""

    def __init__(self, seed: int):
        super().__init__(seed=seed, n_startup_trials=SAMPLER_SETTINGS["n_startup_trials"],
                         n_ei_candidates=SAMPLER_SETTINGS["n_ei_candidates"],
                         multivariate=SAMPLER_SETTINGS["multivariate"],
                         constant_liar=SAMPLER_SETTINGS["constant_liar"])
        self._base_seed = int(seed)

    def before_trial(self, study, trial):
        n_done = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)))
        s = (self._base_seed + 7919 * n_done) % (2 ** 32 - 1)
        self._rng = LazyRandomState(s)
        self._random_sampler._rng = LazyRandomState((s + 1) % (2 ** 32 - 1))
        super().before_trial(study, trial)


def suggest(trial) -> dict:
    p = {}
    for name, (kind, lo, hi, log) in SEARCH_SPACE.items():
        if kind == "int":
            p[name] = trial.suggest_int(name, lo, hi, log=log)
        else:
            p[name] = trial.suggest_float(name, lo, hi, log=log)
    return p


# ----------------------------------------------------------------------------- core
def _columns_for(selected: list[str], features: list[str]) -> list[str]:
    """Selected raw features in canonical order."""
    s = set(selected)
    return [f for f in features if f in s]


def compute_selections(d: FoldData, cfg: TuneConfig) -> dict:
    if cfg.track == "full":
        inner = {j: list(d.features) for j in range(N_INNER)}
        return {"inner": inner, "outer": list(d.features)}
    inner = {}
    for j in range(N_INNER):
        tr = d.train_idx[d.inner_fold != j]
        inner[j] = KL.select_top_k(d.frame.iloc[tr], d.w[tr], d.y[tr], d.features, cfg.top_k)
    outer = KL.select_top_k(d.frame.iloc[d.train_idx], d.w[d.train_idx], d.y[d.train_idx], d.features, cfg.top_k)
    return {"inner": inner, "outer": outer}


def _fit_predict(d: FoldData, cfg: TuneConfig, params: dict, tr: np.ndarray, va: np.ndarray,
                 selected: list[str]) -> np.ndarray:
    cols = _columns_for(selected, d.features)
    xtr = LR.prepare_matrix(d.frame.iloc[tr], cols)
    xva = LR.prepare_matrix(d.frame.iloc[va], cols)
    m = LR.fit(cfg.learner, xtr, d.w[tr], d.y[tr], params, cfg.engine, cfg.model_seed,
               n_jobs=cfg.n_jobs, lgb_min_data_in_leaf=cfg.lgb_min_data_in_leaf)
    return m.predict_uplift(xva)


def make_objective(d: FoldData, cfg: TuneConfig, selections: dict, fail_at_trial: int | None = None):
    def objective(trial):
        if fail_at_trial is not None and trial.number == fail_at_trial:   # test hook: simulated crash
            raise RuntimeError("simulated interruption")
        params = suggest(trial)
        scores = []
        t0 = time.time()
        for j in range(N_INNER):
            tr = d.train_idx[d.inner_fold != j]
            va = d.train_idx[d.inner_fold == j]
            u = _fit_predict(d, cfg, params, tr, va, selections["inner"][j])
            scores.append(QN.qini_auc(d.y[va], u, d.w[va]))
            if cfg.prune and j == 0:
                trial.report(scores[0], step=0)
                if trial.should_prune():
                    trial.set_user_attr("inner_qini", scores)
                    trial.set_user_attr("seconds", time.time() - t0)
                    raise optuna.TrialPruned()
        trial.set_user_attr("inner_qini", scores)
        trial.set_user_attr("seconds", time.time() - t0)
        return float(np.mean(scores))
    return objective


def _open_study(cfg: TuneConfig, storage_dir: Path, fingerprint: dict):
    storage_dir.mkdir(parents=True, exist_ok=True)
    db = storage_dir / f"{cfg.study_name}.sqlite3"
    storage = optuna.storages.RDBStorage(f"sqlite:///{db}")
    sampler = DeterministicTPESampler(cfg.sampler_seed)
    pruner = (optuna.pruners.MedianPruner(n_startup_trials=cfg.pruner_startup_trials, n_warmup_steps=0)
              if cfg.prune else optuna.pruners.NopPruner())
    names = optuna.study.get_all_study_names(storage)
    if cfg.study_name in names:
        study = optuna.load_study(study_name=cfg.study_name, storage=storage, sampler=sampler, pruner=pruner)
        stored = study.user_attrs.get("fingerprint")
        if stored != fingerprint:
            diff = sorted(k for k in fingerprint if (stored or {}).get(k) != fingerprint[k])
            raise FingerprintMismatch(f"{cfg.study_name}: stored study differs in {diff}; refusing to reuse {db}")
        resumed = True
    else:
        assert not names, f"unexpected studies in {db}: {names}"
        study = optuna.create_study(study_name=cfg.study_name, storage=storage, sampler=sampler,
                                    pruner=pruner, direction="maximize")
        study.set_user_attr("fingerprint", fingerprint)
        study.set_user_attr("config", asdict(cfg))
        resumed = False
    stale = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.RUNNING]
    for t in stale:   # left by a killed process (studies are single-process by design)
        storage.set_trial_state_values(t._trial_id, TrialState.FAIL)
    return study, db, resumed, len(stale)


def run_study(d: FoldData, cfg: TuneConfig, storage_dir: Path, out_dir: Path | None = None,
              code_hash: str | None = None, fail_at_trial: int | None = None,
              stop_after: int | None = None) -> dict:
    """Run (or resume) one study, then refit best params on the outer training part and score outer test.

    stop_after: test hook, stop once this many trials are finished (simulates an interruption, no final step).
    """
    started = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fp = {"input": input_fingerprint(d), "config": config_fingerprint(cfg),
          "code": code_hash if code_hash is not None else code_fingerprint()}
    study, db, resumed, n_stale = _open_study(cfg, Path(storage_dir), fp)
    finished = lambda: len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)))  # noqa: E731
    selections = compute_selections(d, cfg)
    target = cfg.n_trials if stop_after is None else min(stop_after, cfg.n_trials)
    remaining = target - finished()
    if remaining > 0:
        study.optimize(make_objective(d, cfg, selections, fail_at_trial), n_trials=remaining,
                       n_jobs=1, catch=(), gc_after_trial=False, show_progress_bar=False)
    if stop_after is not None and stop_after < cfg.n_trials:
        return {"stopped_after": finished(), "db": str(db)}
    assert finished() == cfg.n_trials

    best = study.best_trial
    best_params = {k: best.params[k] for k in SEARCH_SPACE}
    t_outer = time.time()
    u = _fit_predict(d, cfg, best_params, d.train_idx, d.test_idx, selections["outer"])
    outer_q = QN.qini_auc(d.y[d.test_idx], u, d.w[d.test_idx])
    trials = study.get_trials(deepcopy=False)
    complete = [t for t in trials if t.state == TrialState.COMPLETE]
    result = {
        "study": cfg.study_name, "fingerprint": fp, "config": asdict(cfg),
        "search_space": SEARCH_SPACE, "sampler": SAMPLER_SETTINGS,
        "best_trial_number": best.number, "best_params": best_params,
        "inner_best": float(best.value), "inner_best_fold_qini": best.user_attrs.get("inner_qini"),
        "outer_qini": float(outer_q), "optimism_inner_minus_outer": float(best.value - outer_q),
        "selected_inner": {str(k): v for k, v in selections["inner"].items()},
        "selected_outer": selections["outer"],
        "n_trials": {"complete": len(complete),
                     "pruned": sum(t.state == TrialState.PRUNED for t in trials),
                     "fail": sum(t.state == TrialState.FAIL for t in trials)},
        "trial_seconds_mean": float(np.mean([t.user_attrs.get("seconds", np.nan) for t in complete])) if complete else None,
        "outer_refit_seconds": time.time() - t_outer,
        "resumed": resumed, "stale_running_marked_fail": n_stale,
        "elapsed_seconds_this_call": time.time() - started,
        "storage": str(db),
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        res_path = out_dir / f"{cfg.study_name}.json"
        pred_path = out_dir / f"{cfg.study_name}.outer_pred.parquet"
        if res_path.exists():
            old = json.loads(res_path.read_text())
            if old["fingerprint"] != fp:
                raise FingerprintMismatch(f"existing result {res_path} has a different fingerprint")
            if old["best_params"] != best_params or abs(old["outer_qini"] - outer_q) > 1e-12:
                raise RuntimeError(f"existing result {res_path} is not reproduced")
            return old
        out_dir.mkdir(parents=True, exist_ok=True)
        pred = pd.DataFrame({"client_id": d.ids[d.test_idx], "outer_fold": cfg.outer_fold, "uplift": u})
        tmp = pred_path.with_name(pred_path.name + ".tmp")
        pred.to_parquet(tmp, index=False)
        os.replace(tmp, pred_path)
        result["outer_pred_sha256"] = sha256_file(pred_path)
        atomic_write_json(res_path, result)
    return result


# ----------------------------------------------------------------------------- real data (train only)
def _check_sha(path: Path) -> None:
    got = sha256_file(path)
    assert got == EXPECTED_SHA[path.name], f"{path.name} sha256 {got} != expected"


def _read_train_targets(train_w: dict[str, int], holdout_ids: set[str]) -> dict[str, int]:
    """Parse target bytes for train rows only. For train lines the treatment flag is checked against the split
    and the target is parsed. For holdout lines only the client_id is checked (membership in the holdout id set);
    neither the treatment flag nor the target bytes of holdout lines are sliced, parsed or stored."""
    y: dict[str, int] = {}
    seen = 0
    with LABELS_PATH.open("rb") as f:
        assert f.readline().rstrip(b"\r\n") == b"client_id,treatment_flg,target"
        for line in f:
            i = line.find(b",")
            j = line.find(b",", i + 1)
            assert i > 0 and j > i
            cid = line[:i].decode("ascii")
            seen += 1
            if cid in train_w:
                assert int(line[i + 1:j]) == train_w[cid], "W mismatch"
                v = line[j + 1:].strip()
                assert v in (b"0", b"1"), "non-binary train target"
                y[cid] = int(v)
            else:
                assert cid in holdout_ids, "client in labels but in neither split"
                # holdout: target bytes are not sliced, parsed or stored
    assert seen == len(train_w) + len(holdout_ids) and len(y) == len(train_w)
    return y


def load_train_only(verify_sha: bool = True) -> dict:
    """Train rows only (holdout stays sealed). Returns frame (38 raw features), w, y, ids, outer_fold."""
    if verify_sha:
        for p in (FEATURES_PATH, SPLIT_PATH, INNER_PATH, LABELS_PATH):
            _check_sha(p)
    split = pd.read_parquet(SPLIT_PATH)
    assert split.client_id.is_unique and set(split.split.unique()) == {"train", "holdout"}
    train_ids = set(split.loc[split.split == "train", "client_id"])
    holdout_ids = set(split.loc[split.split == "holdout", "client_id"])
    assert not train_ids & holdout_ids and len(train_ids) == EXPECTED_TRAIN_ROWS
    feats = pd.read_parquet(FEATURES_PATH, columns=["client_id", "treatment_flg", *LR.FEATURES_38])
    assert "target" not in feats.columns
    tr = feats.loc[feats.client_id.isin(train_ids)].sort_values("client_id", kind="mergesort").reset_index(drop=True)
    del feats
    assert not tr.client_id.isin(holdout_ids).any(), "holdout client in train loader output"
    assert len(tr) == EXPECTED_TRAIN_ROWS and tr.client_id.is_unique
    w_map = dict(zip(tr.client_id, tr.treatment_flg.astype(int)))
    y_map = _read_train_targets(w_map, holdout_ids)
    of = split.set_index("client_id").outer_fold
    out = {
        "frame": tr[LR.FEATURES_38].copy(),
        "w": tr.treatment_flg.to_numpy(int),
        "y": tr.client_id.map(y_map).to_numpy(int),
        "ids": tr.client_id.to_numpy(str),
        "outer_fold": tr.client_id.map(of).to_numpy(int),
    }
    assert set(np.unique(out["outer_fold"])) == set(range(5))
    assert not set(out["ids"]) & holdout_ids
    return out


def real_fold_data(loaded: dict, k: int) -> FoldData:
    inner = pd.read_parquet(INNER_PATH)
    part = inner.loc[inner.outer_fold == k]
    pos = pd.Series(np.arange(len(loaded["ids"])), index=loaded["ids"])
    assert part.client_id.isin(pos.index).all(), "inner fold file has a non-train client"
    train_idx = pos.loc[part.client_id].to_numpy()
    assert len(train_idx) == len(np.unique(train_idx)), "duplicate rows in train_idx"
    test_idx = np.flatnonzero(loaded["outer_fold"] == k)
    assert set(train_idx) == set(np.flatnonzero(loaded["outer_fold"] != k))
    return FoldData(loaded["frame"], loaded["w"], loaded["y"], loaded["ids"], list(LR.FEATURES_38),
                    train_idx, test_idx, part.inner_fold.to_numpy(int), meta={"outer_fold": k})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=LR.ENGINES)
    ap.add_argument("--learner", required=True, choices=LR.LEARNERS)
    ap.add_argument("--track", required=True, choices=TRACKS)
    ap.add_argument("--outer", type=int, required=True)
    ap.add_argument("--n-trials", type=int, required=True)
    ap.add_argument("--sampler-seed", type=int, default=20261009)
    ap.add_argument("--model-seed", type=int, default=20261009)
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--lgb-min-data-in-leaf", type=int, default=LR.LGB_MIN_DATA_IN_LEAF)
    ap.add_argument("--storage-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    cfg = TuneConfig(a.engine, a.learner, a.track, a.outer, a.n_trials, a.sampler_seed, a.model_seed,
                     prune=a.prune, lgb_min_data_in_leaf=a.lgb_min_data_in_leaf)
    d = real_fold_data(load_train_only(), a.outer)
    r = run_study(d, cfg, a.storage_dir, a.out_dir)
    print(json.dumps({k: r[k] for k in ("study", "inner_best", "outer_qini", "best_params")}, indent=2))


if __name__ == "__main__":
    main()
