"""X5 final stage: final re-tuning, final fit, Y-free holdout scores, seal manifest.

Steps
 1. Selection input: {selection_dir}/selection_result.json + outputs, SHAs verified (selection.verify_selection_dir).
    status 'tie_choice_required' -> a choice file is required ({"chosen", "reason", "revision",
    "decided_at"}; chosen must be in the tie list). status 'selected' -> no choice file allowed.
 2. final_folds.parquet SHA == final_folds.meta.json output_sha256 == expected value.
 3. Train 70 % (140,027, tune.load_train_only) as one FoldData: train_idx = all rows, test_idx = empty,
    inner_fold = final_fold. TuneConfig(outer_fold=-1) -> study name '{track}__{engine}__{learner}__outer-1'.
 4. Re-tune: tune._open_study + compute_selections + make_objective + study.optimize (40 trials, sampler
    and model seed 20261009, NopPruner, same search space / objective). Storage {out_dir}/optuna (never under
    results/nested). Fingerprint = tune {input, config, code} + this file's SHA + selection_result SHA.
    compact track: inner KL selection on final_fold != j rows; final model selection on the whole train.
 5. Final model: best params, fit on all train rows (e = mean(W_train), asserted). Checked bit-identical to
    tune._fit_predict on a row subset. Saved as pickle.
 6. Holdout features only (load_holdout_features): features_full38.parquet (SHA) -> holdout rows of the
    split file, columns client_id, treatment_flg, FEATURES_38. uplift_train.csv is never opened.
    Scores sorted by client_id -> holdout_scores.parquet (client_id, treatment_flg, uplift) + SHA.
 7. Seal manifest (selection files, choice file, final params, model, holdout scores, final folds, inputs,
    code fingerprint + 4 module SHAs + stage script SHAs, env lock SHA + installed versions, NB interval)
    -> seal_manifest.json and seal_manifest.sha256.
Selection guard: selection_result.json must come from the current selection.py (script SHA) with
outer-Qini recomputation and split id check both on.
Re-run: existing outputs must be reproduced (scores exactly, params equal, existing model predicts the same
holdout scores); existing files are never overwritten. Screen: SHAs only (no metric values).

Usage (after selection):
  python modeling/final.py --selection-dir results/selection --out-dir results/final [--choice <file>]
"""
from __future__ import annotations

import argparse
import importlib.metadata as im
import json
import pickle
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState

sys.path.insert(0, str(Path(__file__).resolve().parent))
import learners as LR  # noqa: E402
import selection as SEL  # noqa: E402
import tune as TU  # noqa: E402

ROOT = TU.ROOT
NESTED_DIR = SEL.NESTED_DIR
DEFAULT_OUT_DIR = ROOT / "results" / "final"
FINAL_FOLDS_PATH = ROOT / "data/splits/final_folds.parquet"
FINAL_FOLDS_META = ROOT / "data/splits/final_folds.meta.json"
EXPECTED_FINAL_FOLDS_SHA = "ffb024d793272a8d765fd74ab0f586c0477486646d1a1b5f3abf0f9d7a8f4708"
LOCK_PATH = ROOT / "env/requirements.lock.txt"
STAGE_SCRIPTS = ("selection.py", "final.py", "holdout_eval.py")
EXPECTED_HOLDOUT_ROWS = 60_012
EXPECTED_HOLDOUT_TREATED = 29_995
N_FINAL_FOLDS = 3
FINAL_N_TRIALS = 40
SEED = 20261009
EQUIV_CHECK_ROWS = 2000

SCORES_NAME = "holdout_scores.parquet"
MODEL_NAME = "final_model.pkl"
PARAMS_NAME = "final_params.json"
STUDY_NAME_FILE = "final_study.json"
MANIFEST_NAME = "seal_manifest.json"
MANIFEST_SHA_NAME = "seal_manifest.sha256"
CHOICE_KEYS = ("chosen", "reason", "revision", "decided_at")

sha256_file = SEL.sha256_file
json_bytes = SEL.json_bytes
atomic_write_bytes = SEL.atomic_write_bytes


class SealError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- guards / checks
def assert_outside_nested(path: Path, nested_dir: Path = NESTED_DIR) -> None:
    try:
        SEL.assert_outside_nested(path, nested_dir)
    except SEL.SelectionInputError as ex:
        raise SealError(str(ex)) from None


def check_final_folds(path: Path = FINAL_FOLDS_PATH, meta_path: Path = FINAL_FOLDS_META,
                      expected_sha: str = EXPECTED_FINAL_FOLDS_SHA) -> str:
    got = sha256_file(path)
    meta = json.loads(Path(meta_path).read_text())
    meta_sha = meta["output_sha256"][Path(path).name]
    if not (got == meta_sha == expected_sha):
        raise SealError(f"final folds sha256 mismatch: file {got[:12]} meta {meta_sha[:12]} expected {expected_sha[:12]}")
    return got


def module_shas() -> dict:
    return {p.name: sha256_file(p) for p in TU.CODE_FILES}


def stage_script_shas() -> dict:
    d = Path(__file__).resolve().parent
    return {n: (sha256_file(d / n) if (d / n).exists() else None) for n in STAGE_SCRIPTS}


def env_snapshot(lock_path: Path = LOCK_PATH) -> dict:
    """Lock-file SHA + installed versions of every pinned package; any difference -> SealError."""
    pins, mism = {}, {}
    for line in Path(lock_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, ver = line.split("==", 1)
        pins[name] = ver
        try:
            inst = im.version(name)
        except im.PackageNotFoundError:
            inst = None
        if inst != ver:
            mism[name] = {"lock": ver, "installed": inst}
    if mism:
        raise SealError(f"installed packages differ from lock file: {sorted(mism)}")
    return {"lock_path": str(Path(lock_path).relative_to(ROOT)) if ROOT in Path(lock_path).resolve().parents
            else str(lock_path), "lock_sha256": sha256_file(lock_path), "n_pins": len(pins), "pins": pins,
            "python": platform.python_version(), "machine": platform.machine(), "platform": platform.platform()}


# ----------------------------------------------------------------------------- selection input
def resolve_candidate(sel_dir: Path, choice_path: Path | None = None) -> dict:
    v = SEL.verify_selection_dir(sel_dir)
    r = v["result"]
    # the selection must have been produced by the current selection.py with its full checks on
    sel_script = Path(SEL.__file__).resolve()
    if r.get("script", {}).get("sha256") != sha256_file(sel_script):
        raise SealError("selection_result.json was produced by a different selection.py (script sha256 differs)")
    if r.get("outer_qini_recomputed_from_outer_pred") is not True \
            or r.get("outer_pred_ids_checked_against_split") is not True:
        raise SealError("selection ran without outer-Qini recomputation or split id check (--no-recompute / "
                        "--no-split-check); refusing to seal")
    choice = None
    if r["status"] == SEL.STATUS_SELECTED:
        if choice_path is not None:
            raise SealError("selection has a unique top candidate; a choice file is not applicable")
        cand = r["selected"]
    elif r["status"] == SEL.STATUS_TIE:
        if choice_path is None:
            raise SealError("selection ended in a tie; a choice file is required")
        choice = json.loads(Path(choice_path).read_text())
        missing = [k for k in CHOICE_KEYS if not isinstance(choice.get(k), str) or not choice[k].strip()]
        if missing:
            raise SealError(f"choice file lacks {missing}")
        if choice["chosen"] not in r["tie"]:
            raise SealError("choice file 'chosen' is not in the tie list")
        cand = choice["chosen"]
    else:
        raise SealError(f"unknown selection status {r['status']!r}")
    parts = tuple(cand.split("__"))
    if len(parts) != 3 or parts not in SEL.CANDIDATES:
        raise SealError(f"unknown candidate {cand!r}")
    track, engine, learner = parts
    aux = json.loads((Path(sel_dir) / "nb_aux.json").read_text())
    single = aux["by_reference"][cand]["single"]
    return {"candidate": cand, "track": track, "engine": engine, "learner": learner,
            "selection_status": r["status"], "selection_files_sha256": v["files_sha256"],
            "choice": choice,
            "choice_sha256": sha256_file(choice_path) if choice_path is not None else None,
            "nb_interval": {"flag": bool(single["nb_flag"]), "mean": single["mean"], "lo": single["lo"],
                            "hi": single["hi"], "rule": aux["nb_flag_rule"]}}


# ----------------------------------------------------------------------------- data
def build_final_folddata(loaded: dict, final_folds: pd.DataFrame) -> TU.FoldData:
    ids = np.asarray(loaded["ids"])
    ff = final_folds.set_index("client_id").final_fold
    if not ff.index.is_unique or set(ff.index) != set(ids) or len(ff) != len(ids):
        raise SealError("final fold file ids != train ids")
    inner = ff.loc[ids].to_numpy(int)
    if set(np.unique(inner)) != set(range(N_FINAL_FOLDS)):
        raise SealError("final_fold values are not 0..2")
    n = len(ids)
    return TU.FoldData(loaded["frame"], np.asarray(loaded["w"]), np.asarray(loaded["y"]), ids,
                       list(LR.FEATURES_38), np.arange(n), np.array([], dtype=np.int64), inner,
                       meta={"outer_fold": -1, "final": True})


def load_holdout_features(features_path: Path = TU.FEATURES_PATH, split_path: Path = TU.SPLIT_PATH,
                          expected_sha: dict | None = None, expected_rows: int | None = EXPECTED_HOLDOUT_ROWS,
                          expected_treated: int | None = EXPECTED_HOLDOUT_TREATED,
                          labels_path: Path = TU.LABELS_PATH) -> pd.DataFrame:
    """Holdout rows WITHOUT Y: client_id, treatment_flg, FEATURES_38.

    Reads only the features parquet (sha-checked) and the split file. The label file is never opened."""
    expected_sha = TU.EXPECTED_SHA if expected_sha is None else expected_sha
    lab = Path(labels_path).resolve()
    for p in (features_path, split_path):
        if Path(p).resolve() == lab:
            raise SealError("holdout feature loader must not read the label file")
        got = sha256_file(p)
        if got != expected_sha[Path(p).name]:
            raise SealError(f"{Path(p).name} sha256 mismatch")
    split = pd.read_parquet(split_path, columns=["client_id", "split"])
    hold = set(split.loc[split.split == "holdout", "client_id"])
    train = set(split.loc[split.split == "train", "client_id"])
    cols = ["client_id", "treatment_flg", *LR.FEATURES_38]
    f = pd.read_parquet(features_path, columns=cols)
    if list(f.columns) != cols or "target" in f.columns:
        raise SealError("unexpected holdout feature columns")
    h = f.loc[f.client_id.isin(hold)].sort_values("client_id", kind="mergesort").reset_index(drop=True)
    del f
    if h.client_id.isin(train).any() or not h.client_id.is_unique or set(h.client_id) != hold:
        raise SealError("holdout id check failed")
    if expected_rows is not None and len(h) != expected_rows:
        raise SealError(f"holdout rows {len(h)} != {expected_rows}")
    w = h.treatment_flg.to_numpy(int)
    if set(np.unique(w)) != {0, 1}:
        raise SealError("holdout treatment flag not binary")
    if expected_treated is not None and int(w.sum()) != expected_treated:
        raise SealError("holdout treated count differs from expected")
    return h


def score_holdout(model, cols: list, hold: pd.DataFrame) -> pd.DataFrame:
    x = LR.prepare_matrix(hold, cols)
    u = model.predict_uplift(x)
    out = pd.DataFrame({"client_id": hold.client_id.to_numpy(str), "treatment_flg": hold.treatment_flg.to_numpy(int),
                        "uplift": np.asarray(u, dtype=float)})
    out = out.sort_values("client_id", kind="mergesort").reset_index(drop=True)
    if not (out.client_id.is_unique and np.isfinite(out.uplift).all()):
        raise SealError("holdout scores: duplicate ids or non-finite uplift")
    return out


# ----------------------------------------------------------------------------- re-tune / fit
def run_final_tuning(d: TU.FoldData, cfg: TU.TuneConfig, storage_dir: Path, extra_fp: dict) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    code_fp = TU.code_fingerprint()
    fp = {"input": TU.input_fingerprint(d), "config": TU.config_fingerprint(cfg), "code": code_fp, **extra_fp}
    study, db, resumed, n_stale = TU._open_study(cfg, Path(storage_dir), fp)
    finished = lambda: len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)))  # noqa: E731
    selections = TU.compute_selections(d, cfg)
    remaining = cfg.n_trials - finished()
    if remaining > 0:
        study.optimize(TU.make_objective(d, cfg, selections), n_trials=remaining, n_jobs=1, catch=(),
                       gc_after_trial=False, show_progress_bar=False)
    trials = study.get_trials(deepcopy=False)
    complete = [t for t in trials if t.state == TrialState.COMPLETE]
    if len(complete) != cfg.n_trials or any(t.state == TrialState.PRUNED for t in trials):
        raise SealError("final study does not have exactly n_trials COMPLETE trials")
    best = study.best_trial
    return {"fingerprint": fp, "db": db, "resumed": resumed, "stale_marked_fail": n_stale,
            "selections": selections, "best_trial_number": best.number,
            "best_params": {k: best.params[k] for k in TU.SEARCH_SPACE},
            "inner_best": float(best.value), "inner_best_fold_qini": best.user_attrs.get("inner_qini"),
            "n_trials": {"complete": len(complete), "fail": sum(t.state == TrialState.FAIL for t in trials)}}


def fit_final_model(d: TU.FoldData, cfg: TU.TuneConfig, params: dict, selected: list):
    cols_raw = TU._columns_for(selected, d.features)
    x = LR.prepare_matrix(d.frame.iloc[d.train_idx], cols_raw)
    model = LR.fit(cfg.learner, x, d.w[d.train_idx], d.y[d.train_idx], params, cfg.engine, cfg.model_seed,
                   n_jobs=cfg.n_jobs, lgb_min_data_in_leaf=cfg.lgb_min_data_in_leaf)
    if model.e != float(np.mean(d.w[d.train_idx])):
        raise SealError("e != mean(W_train)")
    # same result as the tune refit path (_fit_predict) -> bit-identical predictions on a row subset
    sub = d.train_idx[: min(EQUIV_CHECK_ROWS, len(d.train_idx))]
    ref = TU._fit_predict(d, cfg, params, d.train_idx, sub, selected)
    mine = model.predict_uplift(LR.prepare_matrix(d.frame.iloc[sub], cols_raw))
    if not np.array_equal(ref, mine):
        raise SealError("final model differs from tune._fit_predict refit")
    return model, cols_raw


# ----------------------------------------------------------------------------- write-or-verify helpers
def _write_json_or_verify(path: Path, obj: dict, ignore=("created_at",)) -> None:
    data = json_bytes(obj)
    if path.exists():
        old = json.loads(path.read_text())
        new = json.loads(data)
        if {k: v for k, v in old.items() if k not in ignore} != {k: v for k, v in new.items() if k not in ignore}:
            raise SealError(f"existing {path.name} is not reproduced")
        return
    atomic_write_bytes(path, data)


def _write_scores_or_verify(path: Path, scores: pd.DataFrame) -> None:
    if path.exists():
        old = pd.read_parquet(path)
        if list(old.columns) != list(scores.columns) or len(old) != len(scores) \
                or not (old.client_id.to_numpy() == scores.client_id.to_numpy()).all() \
                or not np.array_equal(old.treatment_flg.to_numpy(int), scores.treatment_flg.to_numpy(int)) \
                or not np.array_equal(old.uplift.to_numpy(float), scores.uplift.to_numpy(float)):
            raise SealError("existing holdout scores are not reproduced")
        return
    tmp = path.with_name(path.name + ".tmp")
    scores.to_parquet(tmp, index=False)
    tmp.replace(path)


def _write_model_or_verify(path: Path, model, cols: list, hold: pd.DataFrame, scores: pd.DataFrame) -> None:
    if path.exists():
        old = pickle.loads(path.read_bytes())
        if not np.array_equal(score_holdout(old, cols, hold).uplift.to_numpy(float), scores.uplift.to_numpy(float)):
            raise SealError("existing model does not reproduce the holdout scores")
        return
    atomic_write_bytes(path, pickle.dumps(model, protocol=5))


# ----------------------------------------------------------------------------- main routine
def run_final(selection_dir: Path, out_dir: Path, loaded: dict, final_folds: pd.DataFrame, hold: pd.DataFrame,
              *, choice_path: Path | None = None, nested_dir: Path = NESTED_DIR, input_sha: dict | None = None,
              final_folds_sha: str | None = None, env: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    assert_outside_nested(out_dir, nested_dir)
    if Path(out_dir).resolve() == Path(selection_dir).resolve():
        raise SealError("final out_dir must differ from the selection dir")
    if final_folds_sha is None:
        raise SealError("final folds sha256 must be verified before run_final")
    sel = resolve_candidate(selection_dir, choice_path)
    if set(hold.client_id) & set(np.asarray(loaded["ids"])):
        raise SealError("holdout ids overlap train ids")
    d = build_final_folddata(loaded, final_folds)
    cfg = TU.TuneConfig(sel["engine"], sel["learner"], sel["track"], outer_fold=-1, n_trials=FINAL_N_TRIALS,
                        sampler_seed=SEED, model_seed=SEED)
    me = Path(__file__).resolve()
    extra_fp = {"final_stage_script": sha256_file(me),
                "selection_result": sel["selection_files_sha256"]["selection_result.json"],
                "choice": sel["choice_sha256"]}
    started = time.time()
    tune = run_final_tuning(d, cfg, out_dir / "optuna", extra_fp)
    selected = tune["selections"]["outer"]
    model, cols_raw = fit_final_model(d, cfg, tune["best_params"], selected)
    scores = score_holdout(model, cols_raw, hold)
    if len(scores) != len(hold):
        raise SealError("score rows != holdout rows")

    study_res = {"candidate": sel["candidate"], "study": cfg.study_name,
                 "fingerprint": tune["fingerprint"], "config": asdict(cfg), "search_space": TU.SEARCH_SPACE,
                 "sampler": TU.SAMPLER_SETTINGS, "best_trial_number": tune["best_trial_number"],
                 "best_params": tune["best_params"], "inner_best": tune["inner_best"],
                 "inner_best_fold_qini": tune["inner_best_fold_qini"], "n_trials": tune["n_trials"],
                 "selected_by_final_fold": {str(k): v for k, v in tune["selections"]["inner"].items()},
                 "selected_final_model": selected,
                 "storage": str(Path(tune["db"]).resolve().relative_to(out_dir.resolve())),
                 "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    params = {"candidate": sel["candidate"], "track": sel["track"], "engine": sel["engine"],
              "learner": sel["learner"], "best_params": tune["best_params"], "model_seed": SEED,
              "lgb_min_data_in_leaf": cfg.lgb_min_data_in_leaf, "n_jobs": cfg.n_jobs,
              "selected_features": selected, "model_columns_raw": cols_raw,
              "model_columns": list(model.columns), "e_train": model.e, "n_train": int(len(d.train_idx))}
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json_or_verify(out_dir / STUDY_NAME_FILE, study_res, ignore=("created_at", "storage"))
    _write_json_or_verify(out_dir / PARAMS_NAME, params)
    _write_scores_or_verify(out_dir / SCORES_NAME, scores)
    _write_model_or_verify(out_dir / MODEL_NAME, model, cols_raw, hold, scores)

    files = {n: sha256_file(out_dir / n) for n in (STUDY_NAME_FILE, PARAMS_NAME, MODEL_NAME, SCORES_NAME)}
    manifest = {
        "stage": "seal manifest",
        "candidate": sel["candidate"], "selection_status": sel["selection_status"],
        "selection_dir": str(selection_dir), "selection_files_sha256": sel["selection_files_sha256"],
        "choice_sha256": sel["choice_sha256"], "nb_interval": sel["nb_interval"],
        "final_folds_sha256": final_folds_sha, "inputs_sha256": input_sha or {},
        "final_study": {"name": cfg.study_name, "fingerprint": tune["fingerprint"], "n_trials": tune["n_trials"],
                        "best_trial_number": tune["best_trial_number"]},
        "files_sha256": files,
        "holdout_scores": {"file": SCORES_NAME, "sha256": files[SCORES_NAME], "rows": int(len(scores)),
                           "treated": int(scores.treatment_flg.sum()), "sorted_by": "client_id (lexicographic)",
                           "computed_without_labels": True},
        "model": {"file": MODEL_NAME, "sha256": files[MODEL_NAME], "e_train": model.e, "n_train": int(len(d.train_idx)),
                  "format": "pickle protocol 5 of learners.UpliftModel"},
        "code": {"code_fingerprint": tune["fingerprint"]["code"], "module_sha256": module_shas(),
                 "stage_script_sha256": stage_script_shas()},
        "env": env if env is not None else {},
        "elapsed_seconds_this_call": round(time.time() - started, 1),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    mpath = out_dir / MANIFEST_NAME
    _write_json_or_verify(mpath, manifest, ignore=("created_at", "elapsed_seconds_this_call"))
    msha = sha256_file(mpath)
    line = f"{msha}  {MANIFEST_NAME}\n".encode()
    sp = out_dir / MANIFEST_SHA_NAME
    if sp.exists():
        if sp.read_bytes() != line:
            raise SealError("existing seal_manifest.sha256 does not match the manifest")
    else:
        atomic_write_bytes(sp, line)
    return {"manifest_path": str(mpath), "manifest_sha256": msha, "scores_sha256": files[SCORES_NAME],
            "model_sha256": files[MODEL_NAME], "candidate": sel["candidate"], "study": cfg.study_name}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--choice", type=Path, default=None)
    a = ap.parse_args()
    assert_outside_nested(a.out_dir)
    env = env_snapshot()
    ffsha = check_final_folds()
    loaded = TU.load_train_only(verify_sha=True)          # train rows only; checks 4 input SHAs
    final_folds = pd.read_parquet(FINAL_FOLDS_PATH)
    hold = load_holdout_features()                        # holdout X and W only; label file not opened
    input_sha = {TU.FEATURES_PATH.name: TU.EXPECTED_SHA[TU.FEATURES_PATH.name],
                 TU.SPLIT_PATH.name: TU.EXPECTED_SHA[TU.SPLIT_PATH.name],
                 TU.INNER_PATH.name: TU.EXPECTED_SHA[TU.INNER_PATH.name],
                 "uplift_train.csv (train rows only, via load_train_only)": TU.EXPECTED_SHA[TU.LABELS_PATH.name]}
    r = run_final(a.selection_dir, a.out_dir, loaded, final_folds, hold, choice_path=a.choice,
                  input_sha=input_sha, final_folds_sha=ffsha, env=env)
    print(json.dumps({k: r[k] for k in ("manifest_sha256", "scores_sha256", "model_sha256", "study")}, indent=2))


if __name__ == "__main__":
    main()
