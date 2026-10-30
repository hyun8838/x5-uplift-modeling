"""Fixed 3-fold assignment over the whole train 70% for the
final re-tuning. Created before model selection.

Input : data/splits/split.parquet (train rows only) + train targets via tune.load_train_only
        (holdout target bytes are never parsed; see tune._read_train_targets)
Output: data/splits/final_folds.parquet  client_id, final_fold (0..2); train rows only (140,027)
        + final_folds.meta.json

Rules
- rows sorted by client_id (lexicographic string order, mergesort)
- StratifiedKFold(n_splits=3, shuffle=True, random_state=20261015), stratify = 2*W + Y (train rows only)
- seed 20261015 = 20261009 + 1 + 5, disjoint from the other seeds 20261009..20261014
- target dropped after the stratum label; no output contains target
- existing outputs are never overwritten; --out-dir writes elsewhere

Usage:
  python modeling/split_final_folds.py --smoke
  python modeling/split_final_folds.py
  python modeling/split_final_folds.py --out-dir <scratch dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modeling"))
SPLIT_PATH = ROOT / "data/splits/split.parquet"
OUT_DIR = ROOT / "data/splits"
OUT_NAME = "final_folds.parquet"
SEED = 20261015
N_FOLDS = 3
EXPECTED_TRAIN_ROWS = 140_027


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def content_sha256(df: pd.DataFrame) -> str:
    """Version-independent hash: client_id-sorted CSV bytes."""
    d = df.sort_values("client_id", kind="mergesort")
    return hashlib.sha256(d.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def make_final_folds(client_id: np.ndarray, stratum: np.ndarray, seed: int = SEED,
                     n_folds: int = N_FOLDS) -> pd.DataFrame:
    """client_id must be sorted & unique (train rows only). stratum used only as stratify input."""
    idx = pd.Index(client_id)
    assert idx.is_unique and idx.is_monotonic_increasing
    fold = np.full(len(client_id), -1, dtype=np.int8)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for j, (_, te) in enumerate(skf.split(np.zeros(len(client_id)), stratum)):
        fold[te] = j
    assert (fold >= 0).all()
    return pd.DataFrame({"client_id": client_id, "final_fold": fold})


def load_train():
    import tune as TU  # read-only reuse; verifies input SHAs and blocks holdout ids
    d = TU.load_train_only(verify_sha=True)
    ids, w, y = d["ids"], d["w"], d["y"]
    del d
    assert len(ids) == EXPECTED_TRAIN_ROWS
    stratum = (2 * w + y).astype(np.int8)
    del y  # target discarded here
    order = np.argsort(ids, kind="mergesort")
    return ids[order].astype(object), stratum[order], w[order]


def run(out_dir: Path) -> None:
    started = time.time()
    out = out_dir / OUT_NAME
    meta_p = out.with_name(out.stem + ".meta.json")
    for p in (out, meta_p):
        if p.exists():
            raise FileExistsError(f"refusing to overwrite {p}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ids, stratum, w = load_train()
    split = pd.read_parquet(SPLIT_PATH)
    holdout_ids = set(split.loc[split.split == "holdout", "client_id"])
    assert not set(ids) & holdout_ids
    df = make_final_folds(ids, stratum)
    assert list(df.columns) == ["client_id", "final_fold"]
    tmp = out.with_name(out.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    import tune as TU
    sizes = df.final_fold.value_counts().sort_index()
    meta = {
        "status": "final_folds_no_target",
        "purpose": "final re-tuning on whole train 70% (3-fold CV); fixed before model selection",
        "file": OUT_NAME,
        "columns": list(df.columns),
        "seed": SEED,
        "rules": {
            "rows": "split.parquet split=='train' only (140,027); holdout ids asserted absent",
            "row_order": "sorted by client_id (lexicographic string order, mergesort) before splitting",
            "stratum": "2*treatment_flg + target (0..3), train rows only",
            "folds": f"StratifiedKFold(n_splits={N_FOLDS}, shuffle=True, random_state={SEED}) on train rows, stratify=stratum",
            "seed_rule": "20261009 + 1 + 5; disjoint from split/outer 20261009 and inner 20261010..20261014",
            "target_loader": "tune.load_train_only (holdout target bytes not parsed)",
        },
        "target_in_output": False,
        "rows": {"train": len(df)},
        "fold_sizes": {str(k): int(v) for k, v in sizes.items()},
        "treated_train": int(w.sum()),
        "input_sha256": {"split.parquet": sha256_file(SPLIT_PATH),
                         "uplift_train.csv": sha256_file(TU.LABELS_PATH),
                         "features_full38.parquet": sha256_file(TU.FEATURES_PATH),
                         "inner_folds.parquet (sha-verified by loader, not used)": sha256_file(TU.INNER_PATH)},
        "script_sha256": {"split_final_folds.py": sha256_file(Path(__file__)),
                          "tune.py (reused loader)": sha256_file(Path(TU.__file__))},
        "output_sha256": {OUT_NAME: sha256_file(out)},
        "output_content_sha256": {OUT_NAME: content_sha256(df)},
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                    "sklearn": sklearn.__version__, "pyarrow": __import__("pyarrow").__version__},
        "elapsed_seconds": round(time.time() - started, 1),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    atomic_write_json(meta_p, meta)
    print(json.dumps({"rows": meta["rows"], "fold_sizes": meta["fold_sizes"],
                      "output_sha256": meta["output_sha256"]}, indent=2), flush=True)


def smoke() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    cid = np.array([f"{i:06d}" for i in range(n)], dtype=object)
    st = rng.integers(0, 4, n)
    a, b = make_final_folds(cid, st), make_final_folds(cid, st)
    assert a.equals(b) and a.final_fold.between(0, 2).all()
    assert sorted(a.final_fold.value_counts().tolist()) in ([333, 333, 334],)
    print("smoke OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    smoke() if args.smoke else run(args.out_dir)
