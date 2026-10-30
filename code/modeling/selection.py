"""X5 selection stage.

Input : {studies_dir}/{track}__{engine}__{learner}__outer{k}.json and .outer_pred.parquet (140 each,
        output of modeling/tune.py).
Checks per study: file set exact (140, no extra/tmp), study name, config (engine/learner/track/outer k,
        n_trials 40, seeds 20261009, top_k 10, prune False, LGB min_data_in_leaf 1, n_jobs 1), one code fingerprint
        shared by all 140 studies, n_trials.complete == 40 and pruned == 0, outer_pred SHA == JSON value,
        outer_pred rows (columns, outer_fold == k, client ids unique; optionally == split outer fold k),
        inner_best/outer_qini finite, optimism == inner_best - outer_qini, same input fingerprint for all
        28 candidates at each k. Optional: outer Qini recomputed from outer_pred + train labels.
Output (out_dir, never results/nested): atomic writes, SHA of every file in selection_result.json
  qini_table.csv          28 rows: candidate, track, engine, learner, q_outer0..4, mean, sd (ddof=1), rank
  optimism_table.csv      28 rows: inner_best - outer per fold, mean
  kl_selection_counts.csv compact track: per feature, #outer folds selected (0..5) [main], #inner selections
                          (0..15) [auxiliary column, not a mandatory report; see column_notes]
  nb_aux.json             auxiliary Nadeau-Bengio intervals (rho 0.25, t_{4,0.975} 2.7764, two-sided):
                          top vs each of the other 27 (paired by outer fold) and top single-candidate
                          interval + flag (interval contains 0 or mean <= 0). No p-values, no Holm.
  selection_result.json   status 'selected' | 'tie_choice_required', selected candidate or tie list,
                          input SHAs, constants, output-file SHAs, code SHAs
Screen: status, counts and SHAs only (no metric values) by default.
Re-run: if outputs already exist they must be byte-identical (selection_result.json: all keys but
        created_at), otherwise RuntimeError. Existing files are never overwritten.

Usage (after all 140 studies are complete):
  python modeling/selection.py --studies-dir results/nested/studies --out-dir results/selection
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import learners as LR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NESTED_DIR = ROOT / "results" / "nested"
DEFAULT_STUDIES_DIR = NESTED_DIR / "studies"
DEFAULT_OUT_DIR = ROOT / "results" / "selection"
SPLIT_PATH = ROOT / "data/splits/split.parquet"

N_TRIALS = 40
N_OUTER = 5
SEED = 20261009
LGB_MIN_DATA_IN_LEAF = 1
TRACKS = ("full", "compact")
TOP_K = 10                 # compact track k; checked in every study config
TIE_TOL = 1e-12
NB_RHO = 0.25              # fixed constant
NB_T = 2.7764              # t_{4, 0.975}, two-sided 95 %
STATUS_SELECTED = "selected"
STATUS_TIE = "tie_choice_required"

# order: track (full, compact) x engine (xgboost, lightgbm) x learner (LR.LEARNERS order)
CANDIDATES = [(t, e, l) for t in TRACKS for e in LR.ENGINES for l in LR.LEARNERS]
assert len(CANDIDATES) == 28


class SelectionInputError(RuntimeError):
    pass


def cand_id(track: str, engine: str, learner: str) -> str:
    return f"{track}__{engine}__{learner}"


def study_name(track: str, engine: str, learner: str, k: int) -> str:
    return f"{track}__{engine}__{learner}__outer{k}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()


def assert_outside_nested(path: Path, nested_dir: Path = NESTED_DIR) -> None:
    """Outputs of later stages must never be written under results/nested."""
    p, n = Path(path).resolve(), Path(nested_dir).resolve()
    if p == n or n in p.parents:
        raise SelectionInputError(f"refusing to write under the nested directory: {p}")


# ----------------------------------------------------------------------------- statistics (pure)
def nb_halfwidth(values, rho: float = NB_RHO, t: float = NB_T) -> float:
    """t * sqrt((1/K + rho) * s^2), s^2 sample variance (ddof=1) of the K fold values."""
    v = np.asarray(values, dtype=float)
    if not (v.ndim == 1 and len(v) == N_OUTER and np.isfinite(v).all()):
        raise SelectionInputError("NB input must be 5 finite fold values")
    s2 = float(np.var(v, ddof=1))
    return float(t * math.sqrt((1.0 / len(v) + rho) * s2))


def nb_interval(values, rho: float = NB_RHO, t: float = NB_T) -> dict:
    v = np.asarray(values, dtype=float)
    m = float(np.mean(v))
    h = nb_halfwidth(v, rho, t)
    return {"mean": m, "sd": float(np.std(v, ddof=1)), "halfwidth": h, "lo": m - h, "hi": m + h}


def nb_flag(interval: dict) -> bool:
    """Flag rule: interval contains 0 (closed) or mean <= 0."""
    return bool(interval["lo"] <= 0.0 <= interval["hi"] or interval["mean"] <= 0.0)


def rank_desc(means: np.ndarray, tol: float = TIE_TOL) -> np.ndarray:
    """Competition rank by mean (1 = best); means within tol of each other share a rank."""
    means = np.asarray(means, dtype=float)
    return np.array([1 + int(np.sum(means > m + tol)) for m in means], dtype=int)


# ----------------------------------------------------------------------------- loading / checks
def _expected_config(track, engine, learner, k) -> dict:
    return {"engine": engine, "learner": learner, "track": track, "outer_fold": k, "n_trials": N_TRIALS,
            "sampler_seed": SEED, "model_seed": SEED, "top_k": TOP_K, "prune": False,
            "lgb_min_data_in_leaf": LGB_MIN_DATA_IN_LEAF, "n_jobs": 1}


def load_studies(studies_dir: Path, expected_code_fp: str | None = None,
                 split_path: Path | None = None, truth: dict | None = None) -> dict:
    """Read and verify the 140 study results. Returns per-study dicts (values not printed).

    expected_code_fp: if None, every study must carry the same code fingerprint as the first one.
    truth: optional {"ids": array, "w": array, "y": array} of train rows -> outer Qini is recomputed from the
           outer_pred file with qini.qini_auc and must equal the JSON value within 1e-12.
    """
    studies_dir = Path(studies_dir)
    expected = {study_name(t, e, l, k) for (t, e, l) in CANDIDATES for k in range(N_OUTER)}
    jsons = {p.name[:-5] for p in studies_dir.glob("*.json")}
    tmps = sorted(p.name for p in studies_dir.glob("*.tmp"))
    if tmps:
        raise SelectionInputError(f"unfinished temp files present: {tmps[:3]}")
    if jsons != expected:
        raise SelectionInputError(f"study set mismatch: missing {sorted(expected - jsons)[:5]}, "
                                  f"extra {sorted(jsons - expected)[:5]}")
    fold_ids = None
    if split_path is not None:
        sp = pd.read_parquet(split_path, columns=["client_id", "split", "outer_fold"])
        fold_ids = {k: set(sp.loc[(sp.split == "train") & (sp.outer_fold == k), "client_id"]) for k in range(N_OUTER)}
    if truth is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import qini as QN
        tpos = pd.Series(np.arange(len(truth["ids"])), index=pd.Index(truth["ids"]))
    out, input_fp_by_k = {}, {}
    for (t, e, l) in CANDIDATES:
        for k in range(N_OUTER):
            name = study_name(t, e, l, k)
            jp = studies_dir / f"{name}.json"
            pp = studies_dir / f"{name}.outer_pred.parquet"
            r = json.loads(jp.read_text())
            if r.get("study") != name:
                raise SelectionInputError(f"{name}: study name field {r.get('study')!r}")
            fp = r.get("fingerprint") or {}
            if expected_code_fp is None:
                expected_code_fp = fp.get("code")
            if fp.get("code") != expected_code_fp:
                raise SelectionInputError(f"{name}: code fingerprint {fp.get('code')} != {expected_code_fp}")
            cfg = r.get("config") or {}
            exp = _expected_config(t, e, l, k)
            bad = {key: cfg.get(key) for key in exp if cfg.get(key) != exp[key]}
            if bad:
                raise SelectionInputError(f"{name}: config differs {sorted(bad)}")
            nt = r.get("n_trials") or {}
            if nt.get("complete") != N_TRIALS or nt.get("pruned", 0) != 0:
                raise SelectionInputError(f"{name}: n_trials {nt} (need complete == {N_TRIALS}, pruned == 0)")
            input_fp_by_k.setdefault(k, set()).add(fp.get("input"))
            if not pp.exists():
                raise SelectionInputError(f"{name}: outer_pred file missing")
            got = sha256_file(pp)
            if got != r.get("outer_pred_sha256"):
                raise SelectionInputError(f"{name}: outer_pred sha256 mismatch")
            for key in ("inner_best", "outer_qini", "optimism_inner_minus_outer"):
                if not isinstance(r.get(key), (int, float)) or not math.isfinite(r[key]):
                    raise SelectionInputError(f"{name}: {key} missing or not finite")
            if abs((r["inner_best"] - r["outer_qini"]) - r["optimism_inner_minus_outer"]) > 1e-12:
                raise SelectionInputError(f"{name}: optimism != inner_best - outer_qini")
            pred = pd.read_parquet(pp)
            if list(pred.columns) != ["client_id", "outer_fold", "uplift"]:
                raise SelectionInputError(f"{name}: outer_pred columns {list(pred.columns)}")
            if not (pred.outer_fold == k).all() or not pred.client_id.is_unique \
                    or not np.isfinite(pred.uplift.to_numpy(float)).all():
                raise SelectionInputError(f"{name}: outer_pred content check failed")
            if fold_ids is not None and set(pred.client_id) != fold_ids[k]:
                raise SelectionInputError(f"{name}: outer_pred client ids != split outer fold {k}")
            if truth is not None:
                pos = tpos.loc[pred.client_id.to_numpy()].to_numpy()
                q = QN.qini_auc(np.asarray(truth["y"])[pos], pred.uplift.to_numpy(float), np.asarray(truth["w"])[pos])
                if abs(q - r["outer_qini"]) > 1e-12:
                    raise SelectionInputError(f"{name}: outer Qini not reproduced from outer_pred")
            out[name] = {"track": t, "engine": e, "learner": l, "k": k,
                         "outer_qini": float(r["outer_qini"]), "inner_best": float(r["inner_best"]),
                         "optimism": float(r["optimism_inner_minus_outer"]),
                         "selected_outer": list(r.get("selected_outer") or []),
                         "selected_inner": {str(j): list(v) for j, v in (r.get("selected_inner") or {}).items()},
                         "json_sha256": sha256_file(jp), "outer_pred_sha256": got,
                         "input_fp": fp.get("input"), "config_fp": fp.get("config"), "code_fp": fp.get("code")}
    for k, s in input_fp_by_k.items():
        if len(s) != 1:
            raise SelectionInputError(f"outer fold {k}: {len(s)} different input fingerprints across candidates")
    return out


# ----------------------------------------------------------------------------- tables
def build_tables(studies: dict) -> dict:
    rows, opt_rows = [], []
    for (t, e, l) in CANDIDATES:
        q = [studies[study_name(t, e, l, k)]["outer_qini"] for k in range(N_OUTER)]
        o = [studies[study_name(t, e, l, k)]["optimism"] for k in range(N_OUTER)]
        rows.append({"candidate": cand_id(t, e, l), "track": t, "engine": e, "learner": l,
                     **{f"q_outer{k}": q[k] for k in range(N_OUTER)},
                     "mean": float(np.mean(q)), "sd": float(np.std(q, ddof=1))})
        opt_rows.append({"candidate": cand_id(t, e, l), "track": t, "engine": e, "learner": l,
                         **{f"optimism_outer{k}": o[k] for k in range(N_OUTER)}, "optimism_mean": float(np.mean(o))})
    qt = pd.DataFrame(rows)
    qt["rank"] = rank_desc(qt["mean"].to_numpy())
    ot = pd.DataFrame(opt_rows)

    # KL selection counts, compact track (selection depends on data only -> identical across the 14 compact
    # candidates at a given outer fold; asserted)
    compact = [(t, e, l) for (t, e, l) in CANDIDATES if t == "compact"]
    ref = {k: studies[study_name(*compact[0], k)] for k in range(N_OUTER)}
    for c in compact[1:]:
        for k in range(N_OUTER):
            s = studies[study_name(*c, k)]
            if s["selected_outer"] != ref[k]["selected_outer"] or s["selected_inner"] != ref[k]["selected_inner"]:
                raise SelectionInputError(f"KL selection differs across compact candidates at outer {k}: {c}")
    for k in range(N_OUTER):
        sel = ref[k]["selected_outer"]
        if len(sel) != 10 or not set(sel) <= set(LR.FEATURES_38) or len(set(sel)) != 10:
            raise SelectionInputError(f"compact selected_outer at outer {k} is not 10 distinct FEATURES_38")
    kl_rows = []
    for f in LR.FEATURES_38:
        folds = [k for k in range(N_OUTER) if f in ref[k]["selected_outer"]]
        n_inner = sum(f in v for k in range(N_OUTER) for v in ref[k]["selected_inner"].values())
        kl_rows.append({"feature": f, "n_outer_selected": len(folds),
                        "outer_folds": ";".join(map(str, folds)), "n_inner_selected": n_inner})
    kt = pd.DataFrame(kl_rows).sort_values(["n_outer_selected", "n_inner_selected", "feature"],
                                           ascending=[False, False, True], kind="mergesort").reset_index(drop=True)
    return {"qini": qt, "optimism": ot, "kl": kt}


def decide(qt: pd.DataFrame) -> dict:
    means = qt["mean"].to_numpy(float)
    top = float(means.max())
    tie_mask = (top - means) <= TIE_TOL
    tied = qt.loc[tie_mask, "candidate"].tolist()
    if len(tied) == 1:
        return {"status": STATUS_SELECTED, "selected": tied[0], "tie": []}
    return {"status": STATUS_TIE, "selected": None, "tie": tied}


def nb_aux(qt: pd.DataFrame, references: list[str]) -> dict:
    """Auxiliary NB intervals for each reference (the top candidate, or every tied top candidate)."""
    qcols = [f"q_outer{k}" for k in range(N_OUTER)]
    Q = qt.set_index("candidate")[qcols]
    out = {}
    for ref in references:
        single = nb_interval(Q.loc[ref].to_numpy(float))
        single["nb_flag"] = nb_flag(single)
        pairs = []
        for j in Q.index:
            if j == ref:
                continue
            d = Q.loc[ref].to_numpy(float) - Q.loc[j].to_numpy(float)
            iv = nb_interval(d)
            pairs.append({"other": j, "d": [float(x) for x in d], "d_mean": iv["mean"], "d_sd": iv["sd"],
                          "halfwidth": iv["halfwidth"], "lo": iv["lo"], "hi": iv["hi"]})
        out[ref] = {"single": single, "paired_vs_others": pairs}
    return {"rho": NB_RHO, "t_4_0975": NB_T, "two_sided": True, "K": N_OUTER, "variance": "sample, ddof=1",
            "formula_single": "mean +/- t*sqrt((1/K + rho) * s2_Q)",
            "formula_paired": "d_mean +/- t*sqrt((1/K + rho) * s2_d), d_k = Q(ref,k) - Q(j,k)",
            "nb_flag_rule": "lo <= 0 <= hi (closed) or mean <= 0",
            "no_p_values_no_holm": True, "by_reference": out}


# ----------------------------------------------------------------------------- run
OUTPUT_FILES = ("qini_table.csv", "optimism_table.csv", "kl_selection_counts.csv", "nb_aux.json")


def run_selection(studies_dir: Path, out_dir: Path, expected_code_fp: str | None = None,
                  split_path: Path | None = None, truth: dict | None = None,
                  nested_dir: Path = NESTED_DIR) -> dict:
    out_dir = Path(out_dir)
    assert_outside_nested(out_dir, nested_dir)
    studies = load_studies(studies_dir, expected_code_fp, split_path, truth)
    tabs = build_tables(studies)
    dec = decide(tabs["qini"])
    refs = [dec["selected"]] if dec["status"] == STATUS_SELECTED else list(dec["tie"])
    aux = nb_aux(tabs["qini"], refs)
    payload = {"qini_table.csv": csv_bytes(tabs["qini"]), "optimism_table.csv": csv_bytes(tabs["optimism"]),
               "kl_selection_counts.csv": csv_bytes(tabs["kl"]), "nb_aux.json": json_bytes(aux)}
    me = Path(__file__).resolve()
    result = {
        "stage": "selection", "status": dec["status"], "selected": dec["selected"],
        "tie": dec["tie"], "tie_tolerance": TIE_TOL,
        "choice_required": dec["status"] == STATUS_TIE,
        "nb_interval": ({"candidate": dec["selected"], **aux["by_reference"][dec["selected"]]["single"]}
               if dec["status"] == STATUS_SELECTED else None),
        "n_studies": len(studies), "n_trials_each": N_TRIALS,
        "code_fingerprint": next(iter({s["code_fp"] for s in studies.values()})),
        "input_fingerprint_by_outer": {str(k): next(iter({s["input_fp"] for s in studies.values() if s["k"] == k}))
                                       for k in range(N_OUTER)},
        "inputs": {name: {"json_sha256": s["json_sha256"], "outer_pred_sha256": s["outer_pred_sha256"]}
                   for name, s in sorted(studies.items())},
        "outer_qini_recomputed_from_outer_pred": truth is not None,
        "outer_pred_ids_checked_against_split": split_path is not None,
        "outputs_sha256": {k: sha256_bytes(v) for k, v in payload.items()},
        "script": {"path": str(me.relative_to(ROOT)) if ROOT in me.parents else str(me), "sha256": sha256_file(me)},
        "constants": {"nb_rho": NB_RHO, "nb_t": NB_T, "n_outer": N_OUTER, "sd_ddof": 1,
                      "rank": "competition rank by mean, ties within 1e-12 share rank"},
        "studies_dir": (str(Path(studies_dir).resolve().relative_to(ROOT))
                        if ROOT in Path(studies_dir).resolve().parents else str(studies_dir)),
        "column_notes": {"kl_selection_counts.csv": {
            "n_outer_selected": "number of outer folds (0..5) whose outer-k KL selection contains the feature",
            "n_inner_selected": "auxiliary column: number of inner selections (0..15)"}},
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    res_path = out_dir / "selection_result.json"
    if res_path.exists():           # re-run: must reproduce byte-for-byte, never overwrite
        old = json.loads(res_path.read_text())
        new_cmp = {k: v for k, v in json.loads(json_bytes(result)).items() if k != "created_at"}
        old_cmp = {k: v for k, v in old.items() if k != "created_at"}
        if new_cmp != old_cmp:
            raise RuntimeError("existing selection_result.json is not reproduced")
        for name, data in payload.items():
            p = out_dir / name
            if not p.exists() or p.read_bytes() != data:
                raise RuntimeError(f"existing {name} is not reproduced")
        return {"result_path": str(res_path), "result_sha256": sha256_file(res_path), "status": old["status"],
                "reused": True, "outputs_sha256": old["outputs_sha256"]}
    for name in OUTPUT_FILES:
        if (out_dir / name).exists():
            raise RuntimeError(f"{name} exists without selection_result.json; refusing to overwrite")
    for name in OUTPUT_FILES:
        atomic_write_bytes(out_dir / name, payload[name])
    atomic_write_bytes(res_path, json_bytes(result))      # written last
    return {"result_path": str(res_path), "result_sha256": sha256_file(res_path), "status": dec["status"],
            "reused": False, "outputs_sha256": result["outputs_sha256"]}


def verify_selection_dir(sel_dir: Path) -> dict:
    """Used by final.py: selection_result.json present and every listed output matches its SHA."""
    sel_dir = Path(sel_dir)
    res_path = sel_dir / "selection_result.json"
    r = json.loads(res_path.read_text())
    for name, sha in r["outputs_sha256"].items():
        if sha256_file(sel_dir / name) != sha:
            raise SelectionInputError(f"selection output {name} sha256 mismatch")
    files = {name: sha for name, sha in r["outputs_sha256"].items()}
    files["selection_result.json"] = sha256_file(res_path)
    return {"result": r, "files_sha256": files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies-dir", type=Path, default=DEFAULT_STUDIES_DIR)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--no-split-check", action="store_true", help="skip outer_pred client-id check against split")
    ap.add_argument("--no-recompute", action="store_true",
                    help="skip recomputing outer Qini from outer_pred + train labels (load_train_only)")
    a = ap.parse_args()
    truth = None
    if not a.no_recompute:
        import tune as TU
        ld = TU.load_train_only(verify_sha=True)
        truth = {"ids": ld["ids"], "w": ld["w"], "y": ld["y"]}
    r = run_selection(a.studies_dir, a.out_dir, split_path=None if a.no_split_check else SPLIT_PATH, truth=truth)
    # no metric values on screen
    print(json.dumps({"status": r["status"], "reused": r["reused"], "selection_result_sha256": r["result_sha256"],
                      "outputs_sha256": r["outputs_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
