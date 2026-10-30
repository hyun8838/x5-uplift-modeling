"""X5 train/holdout split and nested-CV fold assignment.

Input : data/processed/features_full38.parquet (client_id, treatment_flg only)
        data/og/retailhero-uplift/data/uplift_train.csv (client_id, target -> stratum label only)
Output: data/splits/split.parquet        client_id, split ('train'/'holdout'), outer_fold (train 0..4, holdout -1)
        data/splits/inner_folds.parquet  client_id, outer_fold k, inner_fold (0..2), long format:
                                            for each outer k, the train rows with outer_fold != k (rows = n_train x 4)
        + .meta.json for each

Rules
- stratum = 2*W + Y (W = treatment_flg, Y = target). target is dropped right after the stratum label is built;
  no output file contains target.
- Rows are sorted by client_id before any split so the result does not depend on input row order.
- 70/30: train_test_split(test_size=0.30, stratify=stratum, random_state=SEED).
- outer: StratifiedKFold(5, shuffle=True, random_state=SEED) on train.
- inner: for outer k, StratifiedKFold(3, shuffle=True, random_state=SEED + 1 + k) on the outer-k training part
  (train rows with outer_fold != k, sorted by client_id).
- holdout Y sealed: nothing about holdout target (values, rates, stratum counts) is printed or saved.
  The stratum vector of holdout rows exists in memory only as the stratify argument.
- Existing outputs are never overwritten. --out-dir writes elsewhere.

Usage:
  python modeling/split.py --smoke
  python modeling/split.py
  python modeling/split.py --out-dir <scratch dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
IN_TABLE = ROOT / "data/processed/features_full38.parquet"
IN_LABELS = ROOT / "data/og/retailhero-uplift/data/uplift_train.csv"
OUT_DIR = ROOT / "data/splits"
SEED = 20261009
TEST_SIZE = 0.30
N_OUTER, N_INNER = 5, 3
EXPECTED_CLIENTS = 200_039


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def inner_seed(k: int) -> int:
    return SEED + 1 + k


def make_splits(client_id: np.ndarray, stratum: np.ndarray, seed: int = SEED,
                test_size: float = TEST_SIZE, n_outer: int = N_OUTER, n_inner: int = N_INNER):
    """client_id must be sorted & unique. Returns (split_df, inner_df). stratum is used only as stratify input."""
    assert pd.Index(client_id).is_unique and pd.Index(client_id).is_monotonic_increasing
    idx = np.arange(len(client_id))
    tr_idx, _ = train_test_split(idx, test_size=test_size, stratify=stratum, random_state=seed, shuffle=True)
    tr_idx = np.sort(tr_idx)
    split = np.full(len(idx), "holdout", dtype=object)
    split[tr_idx] = "train"
    outer = np.full(len(idx), -1, dtype=np.int8)
    skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(skf.split(tr_idx, stratum[tr_idx])):
        outer[tr_idx[te]] = k
    split_df = pd.DataFrame({"client_id": client_id, "split": split, "outer_fold": outer})
    parts = []
    for k in range(n_outer):
        part = tr_idx[outer[tr_idx] != k]  # sorted by client_id
        inner = np.full(len(part), -1, dtype=np.int8)
        iskf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + 1 + k)
        for j, (_, te) in enumerate(iskf.split(part, stratum[part])):
            inner[te] = j
        assert (inner >= 0).all()
        parts.append(pd.DataFrame({"client_id": client_id[part], "outer_fold": np.int8(k), "inner_fold": inner}))
    inner_df = pd.concat(parts, ignore_index=True)
    inner_df["outer_fold"] = inner_df["outer_fold"].astype("int8")
    return split_df, inner_df


def load_inputs():
    feats = pd.read_parquet(IN_TABLE, columns=["client_id", "treatment_flg"])
    assert len(feats) == EXPECTED_CLIENTS and feats.client_id.is_unique
    lab = pd.read_csv(IN_LABELS, usecols=["client_id", "treatment_flg", "target"], dtype={"client_id": str})
    m = feats.merge(lab, on="client_id", how="left", validate="one_to_one", suffixes=("", "_csv"), indicator=True)
    assert (m["_merge"] == "both").all()
    assert (m.treatment_flg.astype(int) == m.treatment_flg_csv.astype(int)).all(), "W mismatch features vs csv"
    assert m.target.isin([0, 1]).all()
    m["stratum"] = (2 * m.treatment_flg.astype(int) + m.target.astype(int)).astype("int8")
    m = m.drop(columns=["target", "treatment_flg_csv", "_merge"])  # target discarded here
    del lab
    m = m.sort_values("client_id", kind="mergesort").reset_index(drop=True)
    return m


def run(out_dir: Path) -> None:
    started = time.time()
    out_split = out_dir / "split.parquet"
    out_inner = out_dir / "inner_folds.parquet"
    metas = [p.with_name(p.stem + ".meta.json") for p in (out_split, out_inner)]
    for p in (out_split, out_inner, *metas):
        if p.exists():
            raise FileExistsError(f"refusing to overwrite {p}")
    out_dir.mkdir(parents=True, exist_ok=True)
    m = load_inputs()
    assert "target" not in m.columns
    split_df, inner_df = make_splits(m.client_id.to_numpy(), m.stratum.to_numpy())
    assert "target" not in split_df.columns and "target" not in inner_df.columns
    for df, p in ((split_df, out_split), (inner_df, out_inner)):
        tmp = p.with_name(p.name + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, p)
    n_train = int((split_df.split == "train").sum())
    n_hold = int((split_df.split == "holdout").sum())
    w = m.treatment_flg.astype(int).to_numpy()
    tr = (split_df.split == "train").to_numpy()
    common = {
        "status": "split_no_target",
        "seed": SEED,
        "rules": {
            "row_order": "sorted by client_id (mergesort) before splitting",
            "stratum": "2*treatment_flg + target (0..3)",
            "train_holdout": f"sklearn train_test_split(test_size={TEST_SIZE}, stratify=stratum, random_state={SEED}, shuffle=True)",
            "outer": f"StratifiedKFold(n_splits={N_OUTER}, shuffle=True, random_state={SEED}) on train rows (client_id order), stratify=stratum",
            "inner": f"for outer k: StratifiedKFold(n_splits={N_INNER}, shuffle=True, random_state={SEED}+1+k) on train rows with outer_fold != k (client_id order), stratify=stratum",
            "inner_seeds": {str(k): inner_seed(k) for k in range(N_OUTER)},
        },
        "target_handling": "uplift_train.csv target read only to build stratum; target column dropped immediately; "
                           "no output contains target; no holdout target value/rate/stratum count computed for output",
        "target_in_output": False,
        "rows": {"total": len(split_df), "train": n_train, "holdout": n_hold,
                 "inner_long_rows": len(inner_df)},
        "treated": {"train": int(w[tr].sum()), "holdout": int(w[~tr].sum())},
        "input_sha256": {"features_full38.parquet": sha256_file(IN_TABLE),
                         "uplift_train.csv": sha256_file(IN_LABELS)},
        "script_sha256": sha256_file(Path(__file__)),
        "output_sha256": {"split.parquet": sha256_file(out_split),
                          "inner_folds.parquet": sha256_file(out_inner)},
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                    "sklearn": sklearn.__version__, "pyarrow": __import__("pyarrow").__version__},
        "elapsed_seconds": round(time.time() - started, 1),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    atomic_write_json(metas[0], {**common, "file": "split.parquet",
                                 "columns": list(split_df.columns)})
    atomic_write_json(metas[1], {**common, "file": "inner_folds.parquet",
                                 "columns": list(inner_df.columns)})
    print(json.dumps({"rows": common["rows"], "treated": common["treated"],
                      "output_sha256": common["output_sha256"]}, indent=2), flush=True)


def smoke() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    cid = np.array([f"{i:06d}" for i in range(n)], dtype=object)
    st = rng.integers(0, 4, n)
    s1, i1 = make_splits(cid, st)
    s2, i2 = make_splits(cid, st)
    assert s1.equals(s2) and i1.equals(i2)
    tr = s1.split.eq("train")
    assert tr.sum() == 700 and (s1.outer_fold[~tr] == -1).all() and s1.outer_fold[tr].between(0, 4).all()
    assert len(i1) == tr.sum() * 4
    for k in range(5):
        part = i1[i1.outer_fold == k]
        assert set(part.client_id) == set(s1.client_id[tr & (s1.outer_fold != k)])
        assert part.client_id.is_unique and part.inner_fold.between(0, 2).all()
    print("smoke OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    smoke() if args.smoke else run(args.out_dir)
