"""X5 holdout one-time evaluation.

Gate (gate()): {final_dir}/seal_manifest.json and seal_manifest.sha256 must exist; manifest SHA ==
  seal_manifest.sha256; holdout_scores.parquet, model and final_params SHA == manifest;
  final_folds_sha256 == expected value.
  Current code / environment == manifest record: SHA of the three stage scripts, the four
  fingerprinted modules, tune.code_fingerprint(), lock-file SHA, installed pinned versions, python/platform.
  Any failure -> SealError, the label file is not touched.
Holdout Y loader (load_holdout_labels): only with a GateToken from gate(); label file SHA checked; holdout rows
  only; label W must equal feature W for every holdout customer (explicit raise, not assert).
Before reading Y the stored scores are recomputed from the sealed model + holdout features and must be identical.
Metrics (client_id lexicographic order):
  qini              qini.qini_auc (standard normalized Qini)
  uplift_at_30      sklift uplift_at_k(k=0.3, strategy="by_group"), stored as proportion and %p
  bootstrap         2,000 reps, resample within W=1 and W=0 separately (Y not stratified), numpy
                    default_rng(20261016), per rep: treated draw then control draw, indices sorted;
                    percentile 95 % (2.5, 97.5; numpy 'linear') for qini and uplift_at_30
  permutation       5,000 shuffles of the score vector (W, Y fixed), default_rng(20261017),
                    one-sided b = #(perm Qini >= observed), p = (1 + b) / (1 + 5000)
  boundary ties     tie block at the top-30 % cut, per group
Output ({out_dir}, default {final_dir}/holdout_eval): holdout_predictions.parquet (client_id, treatment_flg,
  target, uplift) and holdout_eval.json. Both are written into a temp directory "{out_dir}.tmp" which is then
  renamed to {out_dir} in one step; a "{out_dir}.tmp" left by an interrupted run is moved
  aside to "{out_dir}.tmp_orphan_<timestamp>" (never deleted) and the run proceeds; if that orphan already holds a
  result JSON, its seal/code SHAs and metrics must match this run within 1e-12. No screen output.
Re-run: allowed only if code, scores, model and manifest SHAs are unchanged; every stored metric must be
  reproduced within 1e-12, otherwise RuntimeError; nothing is overwritten.

Usage:
  python modeling/holdout_eval.py --final-dir results/final
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklift.metrics import uplift_at_k

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final as FIN  # noqa: E402
import qini as QN  # noqa: E402
import selection as SEL  # noqa: E402
import tune as TU  # noqa: E402

ROOT = TU.ROOT
EVAL_DIRNAME = "holdout_eval"
RESULT_NAME = "holdout_eval.json"
PRED_NAME = "holdout_predictions.parquet"
LABELS_SHA = TU.EXPECTED_SHA[TU.LABELS_PATH.name]
K_TOP = 0.3
N_BOOT = 2000
N_PERM = 5000
SEED_BOOT = 20261016
SEED_PERM = 20261017
TOL = 1e-12

sha256_file = SEL.sha256_file
SealError = FIN.SealError


@dataclass(frozen=True)
class GateToken:
    final_dir: Path
    manifest: dict
    manifest_sha256: str
    scores_sha256: str
    model_sha256: str
    params_sha256: str


ENV_KEYS = ("lock_sha256", "pins", "python", "machine", "platform")


def check_code_env(man: dict) -> None:
    """The code and environment running now must be the sealed ones."""
    code = man.get("code") or {}
    if FIN.stage_script_shas() != code.get("stage_script_sha256"):
        raise SealError("code changed after sealing: stage script sha256 differs from manifest")
    if FIN.module_shas() != code.get("module_sha256"):
        raise SealError("code changed after sealing: fingerprinted module sha256 differs from manifest")
    if TU.code_fingerprint() != code.get("code_fingerprint"):
        raise SealError("code changed after sealing: code fingerprint differs from manifest")
    env_now = FIN.env_snapshot()          # also raises if installed versions differ from the lock file
    env_man = man.get("env") or {}
    diff = [k for k in ENV_KEYS if env_now.get(k) != env_man.get(k)]
    if diff:
        raise SealError(f"environment changed after sealing: {diff}")


def gate(final_dir: Path) -> GateToken:
    final_dir = Path(final_dir)
    mpath, spath = final_dir / FIN.MANIFEST_NAME, final_dir / FIN.MANIFEST_SHA_NAME
    for p in (mpath, spath):
        if not p.exists():
            raise SealError(f"{p.name} missing: holdout Y loader refuses to run")
    msha = sha256_file(mpath)
    if spath.read_text().split()[0] != msha:
        raise SealError("seal_manifest.sha256 does not match seal_manifest.json")
    man = json.loads(mpath.read_text())
    if man.get("final_folds_sha256") != FIN.EXPECTED_FINAL_FOLDS_SHA:
        raise SealError("manifest final folds sha256 differs from the expected value")
    check_code_env(man)
    scores_sha = sha256_file(final_dir / FIN.SCORES_NAME)
    if not (scores_sha == man["files_sha256"][FIN.SCORES_NAME] == man["holdout_scores"]["sha256"]):
        raise SealError("holdout scores sha256 differs between manifest and file")
    model_sha = sha256_file(final_dir / FIN.MODEL_NAME)
    if model_sha != man["files_sha256"][FIN.MODEL_NAME] or model_sha != man["model"]["sha256"]:
        raise SealError("model sha256 differs from manifest")
    params_sha = sha256_file(final_dir / FIN.PARAMS_NAME)
    if params_sha != man["files_sha256"][FIN.PARAMS_NAME]:
        raise SealError("final params sha256 differs from manifest")
    return GateToken(final_dir, man, msha, scores_sha, model_sha, params_sha)


def _recheck(token: GateToken) -> None:
    d = token.final_dir
    if (sha256_file(d / FIN.MANIFEST_NAME) != token.manifest_sha256
            or sha256_file(d / FIN.SCORES_NAME) != token.scores_sha256
            or sha256_file(d / FIN.MODEL_NAME) != token.model_sha256):
        raise SealError("sealed files changed after the gate")


# ----------------------------------------------------------------------------- holdout Y loader
def load_holdout_labels(token: GateToken, client_ids: np.ndarray, w_features: np.ndarray,
                        labels_path: Path = TU.LABELS_PATH, expected_sha: str = LABELS_SHA) -> np.ndarray:
    """Y of the holdout customers in the given client order. Only after gate()."""
    if not isinstance(token, GateToken):
        raise SealError("holdout Y loader requires a GateToken from gate()")
    _recheck(token)
    if sha256_file(labels_path) != expected_sha:
        raise SealError("label file sha256 mismatch")
    lab = pd.read_csv(labels_path, dtype={"client_id": str})
    if list(lab.columns) != ["client_id", "treatment_flg", "target"]:
        raise SealError("unexpected label file header")
    ids = pd.Index(np.asarray(client_ids, dtype=str))
    if not ids.is_unique:
        raise SealError("duplicate holdout ids")
    sub = lab.loc[lab.client_id.isin(ids)].set_index("client_id")
    del lab
    if not sub.index.is_unique or len(sub) != len(ids):
        raise SealError("holdout ids not found exactly once in label file")
    sub = sub.loc[ids]
    w_lab = sub.treatment_flg.to_numpy(int)
    if not np.array_equal(w_lab, np.asarray(w_features, dtype=int)):
        raise SealError("label W != feature W")
    y = sub.target.to_numpy(int)
    if not set(np.unique(y)) <= {0, 1}:
        raise SealError("holdout target not binary")
    return y


# ----------------------------------------------------------------------------- metrics
def uplift_at_30(y, u, w) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        v = float(uplift_at_k(np.asarray(y, int), np.asarray(u, float), np.asarray(w, int), strategy="by_group", k=K_TOP))
    if not math.isfinite(v):
        raise RuntimeError("uplift_at_k not finite")
    return v


def bootstrap(y, u, w, n_boot: int = N_BOOT, seed: int = SEED_BOOT) -> dict:
    y, u, w = np.asarray(y, int), np.asarray(u, float), np.asarray(w, int)
    rng = np.random.default_rng(seed)
    it, ic = np.flatnonzero(w == 1), np.flatnonzero(w == 0)
    q, a = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = np.sort(np.concatenate([rng.choice(it, size=len(it), replace=True),
                                      rng.choice(ic, size=len(ic), replace=True)]))
        wb = w[idx]
        if int(wb.sum()) != len(it) or int((wb == 0).sum()) != len(ic):
            raise RuntimeError("bootstrap group sizes not preserved")
        q[b] = QN.qini_auc(y[idx], u[idx], wb)
        a[b] = uplift_at_30(y[idx], u[idx], wb)
    return {"qini": q, "uplift_at_30": a}


def permutation(y, u, w, n_perm: int = N_PERM, seed: int = SEED_PERM) -> dict:
    y, u, w = np.asarray(y, int), np.asarray(u, float), np.asarray(w, int)
    obs = QN.qini_auc(y, u, w)
    rng = np.random.default_rng(seed)
    b = 0
    for _ in range(n_perm):
        if QN.qini_auc(y, u[rng.permutation(len(u))], w) >= obs:
            b += 1
    return {"b": int(b), "p": (1 + b) / (1 + n_perm), "n_perm": n_perm}


def boundary_ties(u, w, k: float = K_TOP) -> dict:
    out = {}
    for g, name in ((1, "treated"), (0, "control")):
        ug = np.sort(np.asarray(u, float)[np.asarray(w, int) == g])[::-1]
        n_cut = int(len(ug) * k)
        bval = ug[n_cut - 1]
        block = int(np.sum(ug == bval))
        in_top = int(np.sum(ug[:n_cut] == bval))
        out[name] = {"n_group": int(len(ug)), "n_cut": n_cut, "tie_block_size": block, "tie_block_in_top": in_top,
                     "straddles_cut": bool(block > in_top), "tie_block_share_of_group": block / len(ug)}
    return out


def compute_metrics(y, u, w, n_boot: int = N_BOOT, n_perm: int = N_PERM,
                    seed_boot: int = SEED_BOOT, seed_perm: int = SEED_PERM) -> dict:
    q = QN.qini_auc(y, u, w)
    a = uplift_at_30(y, u, w)
    bs = bootstrap(y, u, w, n_boot, seed_boot)
    pm = permutation(y, u, w, n_perm, seed_perm)
    qlo, qhi = (float(x) for x in np.percentile(bs["qini"], [2.5, 97.5]))
    alo, ahi = (float(x) for x in np.percentile(bs["uplift_at_30"], [2.5, 97.5]))
    return {"n": int(len(y)), "n_treated": int(np.sum(w)),
            "qini": q, "qini_ci95": [qlo, qhi],
            "uplift_at_30": a, "uplift_at_30_pp": 100.0 * a, "uplift_at_30_ci95": [alo, ahi],
            "uplift_at_30_pp_ci95": [100.0 * alo, 100.0 * ahi],
            "permutation": pm, "boundary_ties": boundary_ties(u, w),
            "settings": {"k": K_TOP, "strategy": "by_group", "n_boot": n_boot, "seed_boot": seed_boot,
                         "bootstrap": "within W groups, Y not stratified, sorted indices, percentile linear",
                         "n_perm": n_perm, "seed_perm": seed_perm,
                         "permutation": "shuffle scores, W and Y fixed, one-sided >=, p=(1+b)/(1+n_perm)",
                         "rng": "numpy default_rng (PCG64)", "order": "client_id lexicographic"}}


def _close(a, b, tol=TOL) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, z, tol) for x, z in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b


# ----------------------------------------------------------------------------- run
def _tmp_dir(out_dir: Path) -> Path:
    return out_dir.with_name(out_dir.name + ".tmp")


def _move_orphan_tmp(out_dir: Path) -> list:
    """A temp dir left by an interrupted run is renamed aside with a timestamp (never deleted).
    Returns (moved names, the orphan's result JSON if it had one -> compared with this run)."""
    tmp = _tmp_dir(out_dir)
    if not tmp.exists():
        return [], None
    orphan_result = json.loads((tmp / RESULT_NAME).read_text()) if (tmp / RESULT_NAME).exists() else None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest, i = out_dir.with_name(f"{out_dir.name}.tmp_orphan_{stamp}"), 1
    while dest.exists():
        dest, i = out_dir.with_name(f"{out_dir.name}.tmp_orphan_{stamp}_{i}"), i + 1
    os.rename(tmp, dest)
    return [dest.name], orphan_result


def _write_predictions(path: Path, pred: pd.DataFrame) -> None:
    pred.to_parquet(path, index=False)
    with path.open("rb") as f:
        os.fsync(f.fileno())


def _write_result_json(path: Path, result: dict) -> None:
    SEL.atomic_write_bytes(path, SEL.json_bytes(result))


def _fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run_eval(final_dir: Path, out_dir: Path | None = None, *, labels_path: Path = TU.LABELS_PATH,
             labels_sha: str = LABELS_SHA, features_path: Path = TU.FEATURES_PATH, split_path: Path = TU.SPLIT_PATH,
             feature_sha: dict | None = None, expected_rows: int | None = FIN.EXPECTED_HOLDOUT_ROWS,
             expected_treated: int | None = FIN.EXPECTED_HOLDOUT_TREATED,
             nested_dir: Path = SEL.NESTED_DIR) -> dict:
    final_dir = Path(final_dir)
    out_dir = Path(out_dir) if out_dir is not None else final_dir / EVAL_DIRNAME
    FIN.assert_outside_nested(out_dir, nested_dir)
    token = gate(final_dir)                                            # before anything touches labels

    # sealed scores == model applied to holdout features (label file not opened here)
    scores = pd.read_parquet(final_dir / FIN.SCORES_NAME)
    params = json.loads((final_dir / FIN.PARAMS_NAME).read_text())
    hold = FIN.load_holdout_features(features_path, split_path, feature_sha, expected_rows, expected_treated,
                                     labels_path=labels_path)
    model = pickle.loads((final_dir / FIN.MODEL_NAME).read_bytes())
    re_scores = FIN.score_holdout(model, params["model_columns_raw"], hold)
    if not (list(scores.client_id) == list(re_scores.client_id)
            and np.array_equal(scores.uplift.to_numpy(float), re_scores.uplift.to_numpy(float))):
        raise SealError("sealed holdout scores are not reproduced by the sealed model")
    if not scores.client_id.is_monotonic_increasing:
        raise SealError("scores are not in client_id order")
    w = scores.treatment_flg.to_numpy(int)
    if not np.array_equal(w, re_scores.treatment_flg.to_numpy(int)):
        raise SealError("score-file W != feature W")
    u = scores.uplift.to_numpy(float)

    code = {"eval_script_sha256": sha256_file(Path(__file__).resolve()),
            "stage_script_sha256": FIN.stage_script_shas(),
            "code_fingerprint": TU.code_fingerprint(), "module_sha256": FIN.module_shas()}
    seal = {"manifest_sha256": token.manifest_sha256, "scores_sha256": token.scores_sha256,
            "model_sha256": token.model_sha256, "params_sha256": token.params_sha256}
    res_path, pred_path = out_dir / RESULT_NAME, out_dir / PRED_NAME
    if out_dir.exists() and not (res_path.exists() and pred_path.exists()):
        raise RuntimeError("output dir exists but is incomplete (it is only ever created by one rename)")
    old = json.loads(res_path.read_text()) if res_path.exists() else None
    if old is not None and (old["seal"] != seal or old["code"] != code):
        raise RuntimeError("re-run refused: code / scores / model / manifest SHA differ from stored run")
    moved, orphan_result = ([], None) if old is not None else _move_orphan_tmp(out_dir)   # keep, never delete
    if orphan_result is not None and (orphan_result.get("seal") != seal or orphan_result.get("code") != code):
        raise RuntimeError("orphan result from an interrupted run has different seal / code SHAs")

    y = load_holdout_labels(token, scores.client_id.to_numpy(str), w, labels_path, labels_sha)
    metrics = compute_metrics(y, u, w)
    pred = pd.DataFrame({"client_id": scores.client_id.to_numpy(str), "treatment_flg": w, "target": y, "uplift": u})

    if orphan_result is not None and not _close(orphan_result.get("metrics"), json.loads(json.dumps(metrics))):
        raise RuntimeError("orphan result from an interrupted run differs from this run (> 1e-12)")
    if old is not None:
        if not _close(old["metrics"], json.loads(json.dumps(metrics))):
            raise RuntimeError("re-run metrics differ from stored results (> 1e-12)")
        oldp = pd.read_parquet(pred_path)
        if not (list(oldp.client_id) == list(pred.client_id) and np.array_equal(oldp.target, pred.target)
                and np.array_equal(oldp.treatment_flg, pred.treatment_flg) and np.array_equal(oldp.uplift, pred.uplift)):
            raise RuntimeError("re-run predictions differ from stored file")
        return {"result_path": str(res_path), "result_sha256": sha256_file(res_path), "reused": True}

    # write both files into "{out_dir}.tmp", then one directory rename
    tmp_dir = _tmp_dir(out_dir)
    tmp_dir.mkdir(parents=True)
    _write_predictions(tmp_dir / PRED_NAME, pred)
    result = {"stage": "holdout evaluation", "candidate": token.manifest["candidate"],
              "nb_interval": token.manifest["nb_interval"], "seal": seal, "code": code, "labels_sha256": labels_sha,
              "predictions": {"file": PRED_NAME, "sha256": sha256_file(tmp_dir / PRED_NAME)},
              "orphan_tmp_moved_aside": moved,
              "metrics": metrics, "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    _write_result_json(tmp_dir / RESULT_NAME, result)
    _fsync_dir(tmp_dir)
    os.rename(tmp_dir, out_dir)          # same filesystem, target absent -> atomic
    _fsync_dir(out_dir.parent)
    return {"result_path": str(res_path), "result_sha256": sha256_file(res_path), "reused": False,
            "orphan_tmp_moved_aside": moved}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    run_eval(a.final_dir, a.out_dir)    # no screen output by design


if __name__ == "__main__":
    main()
