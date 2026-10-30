"""build_features.py chunk-size invariance and cutoff boundary cases.

Pass criteria:
  1. real data: rerun with --chunk-rows 400,000 and 5,000,000 (scratch outputs) -> output parquet SHA-256 equal to
     the canonical features_cut.parquet (built with 1,500,000). target is not read by the script.
  2. synthetic boundary run through the real run() path (module SOURCE patched to a scratch folder, pyarrow block
     size patched to 512 bytes so clients cross batch boundaries and the carry path is used), chunk_rows 1/3/1000:
     identical output SHA across chunk sizes, and
       - transaction exactly at cutoff: kept, days_since_purchase == 0, counted in 30d window,
         diagnostics train_rows_at_cutoff_exact == 2 (2 lines)
       - transaction 1 s after cutoff: excluded, train_rows_after_cutoff == 1
       - transaction exactly cutoff - 30 d: in 30d window; 1 s earlier: not in 30d, in 60d
       - transaction exactly cutoff - 90 d: in 90d window (spend_90d, product_lines_90d); 1 s earlier: not
       - client spread over many batches: purchase_count / total_spend equal to hand count
"""
from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "modeling/build_features.py"
CANON = ROOT / "data/processed/features_cut.parquet"
CANON_META = ROOT / "data/processed/features_cut.meta.json"
CUTOFF = "2019-03-18 23:40:03"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def real_rerun(log, scratch: Path) -> dict:
    canon_sha = json.loads(CANON_META.read_text())["output_sha256"]
    assert sha(CANON) == canon_sha
    res = {"canonical_sha256": canon_sha, "canonical_chunk_rows": json.loads(CANON_META.read_text())["chunk_rows"],
           "runs": {}}
    for c in (400_000, 5_000_000):
        out = scratch / f"real_chunk_{c}" / "features.parquet"
        if not out.exists():
            p = subprocess.run([sys.executable, str(SCRIPT), "--cutoff", CUTOFF, "--out", str(out),
                                "--chunk-rows", str(c)], capture_output=True, text=True)
            assert p.returncode == 0, p.stderr[-2000:]
        meta = json.loads(out.with_name("features.meta.json").read_text())
        s = sha(out)
        res["runs"][str(c)] = {"sha256": s, "equal": s == canon_sha, "parts": meta["diagnostics"]["parts"],
                               "elapsed_seconds": meta["elapsed_seconds_build"]}
        log(f"  real rerun chunk_rows={c:,}: parts={meta['diagnostics']['parts']} sha {s[:12]}... "
            f"equal to canonical {s == canon_sha} ({meta['elapsed_seconds_build']}s)")
    res["pass"] = all(r["equal"] for r in res["runs"].values())
    return res


def _write_synthetic(src: Path) -> dict:
    src.mkdir(parents=True, exist_ok=True)
    cut = pd.Timestamp(CUTOFF)
    f = lambda t: t.strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731
    d30, d90 = pd.Timedelta(days=30), pd.Timedelta(days=90)
    s1 = pd.Timedelta(seconds=1)
    rows = []

    def tx(cid, tid, ts, psum, lines):
        for k, prod in enumerate(lines):
            rows.append([cid, tid, f(ts), 1.0, 0.0, 0.0, 0.0, psum, "s1", prod, 1.0, 10.0, np.nan])

    # client a: at cutoff (2 lines), cutoff-30d exactly, cutoff-30d-1s, cutoff+1s (excluded)
    tx("a", "a1", cut, 100.0, ["p1", "p2"])
    tx("a", "a2", cut - d30, 20.0, ["p1"])
    tx("a", "a3", cut - d30 - s1, 7.0, ["p1"])
    tx("a", "a4", cut + s1, 999.0, ["p1"])
    # client b: cutoff-90d exactly, cutoff-90d-1s, plus one mid transaction
    tx("b", "b1", cut - d90, 30.0, ["p2"])
    tx("b", "b2", cut - d90 - s1, 40.0, ["p2"])
    tx("b", "b3", pd.Timestamp("2019-01-10 12:00:00"), 5.0, ["p1"])
    # client c: 15 transactions (crosses many 512-byte batches)
    for i in range(15):
        tx("c", f"c{i:02d}", pd.Timestamp("2018-12-01 10:00:00") + pd.Timedelta(days=5 * i), float(i + 1), ["p1", "p3"])
    # client z: not a train client, after cutoff
    tx("z", "z1", cut + pd.Timedelta(hours=1), 1.0, ["p1"])
    cols = ["client_id", "transaction_id", "transaction_datetime", "regular_points_received",
            "express_points_received", "regular_points_spent", "express_points_spent", "purchase_sum",
            "store_id", "product_id", "product_quantity", "trn_sum_from_iss", "trn_sum_from_red"]
    pur = pd.DataFrame(rows, columns=cols)
    pur.to_csv(src / "purchases.csv", index=False)
    pd.DataFrame({"client_id": ["a", "b", "c", "z"], "first_issue_date": ["2018-01-01 00:00:00"] * 4,
                  "first_redeem_date": [None] * 4, "age": [30, 40, 50, 60], "gender": ["F", "M", "U", "U"]}
                 ).to_csv(src / "clients.csv", index=False)
    pd.DataFrame({"product_id": ["p1", "p2", "p3"], "level_1": "x", "level_2": ["l1", "l2", "l1"], "level_3": "x",
                  "level_4": "x", "segment_id": 1, "brand_id": ["b1", "b2", None], "vendor_id": "v",
                  "netto": [0.5, 1.0, 0.2], "is_own_trademark": [0, 1, 0], "is_alcohol": [0, 0, 1]}
                 ).to_csv(src / "products.csv", index=False)
    pd.DataFrame({"client_id": ["a", "b", "c"], "treatment_flg": [1, 0, 1], "target": [0, 1, 0]}
                 ).to_csv(src / "uplift_train.csv", index=False)
    return {"raw_rows": len(pur)}


def synthetic_boundary(log, scratch: Path) -> dict:
    sys.path.insert(0, str(ROOT / "modeling"))
    bf = importlib.import_module("build_features")
    src = scratch / "synth_src"
    info = _write_synthetic(src)
    orig = (bf.SOURCE, bf.EXPECTED_RAW_ROWS, bf.EXPECTED_CLIENTS, bf.iter_purchases)
    base_iter = bf.iter_purchases
    shas, frames, metas = {}, {}, {}
    try:
        bf.SOURCE, bf.EXPECTED_RAW_ROWS, bf.EXPECTED_CLIENTS = src, info["raw_rows"], 3
        bf.iter_purchases = lambda path, block_size: base_iter(path, 512)
        for c in (1, 3, 1000):
            out = scratch / f"synth_chunk_{c}" / "features.parquet"
            if not out.exists():
                bf.run(pd.Timestamp(CUTOFF), out, c, None, write_meta=True)
            shas[c] = sha(out)
            frames[c] = pd.read_parquet(out).set_index("client_id")
            metas[c] = json.loads(out.with_name("features.meta.json").read_text())
    finally:
        bf.SOURCE, bf.EXPECTED_RAW_ROWS, bf.EXPECTED_CLIENTS, bf.iter_purchases = orig
    o = frames[1]
    dg = metas[1]["diagnostics"]
    a, b, c = o.loc["a"], o.loc["b"], o.loc["c"]
    checks = {
        "same_sha_across_chunk_rows": len(set(shas.values())) == 1,
        "multiple_parts_used": metas[1]["diagnostics"]["parts"] > 1,
        "at_cutoff_kept_days_since_purchase_0": a.days_since_purchase == 0.0 and a.purchase_count == 3,
        "after_cutoff_excluded": dg["train_rows_after_cutoff"] == 1 and a.total_spend == 127.0,
        "at_cutoff_exact_lines_2": dg["train_rows_at_cutoff_exact"] == 2,
        "exact_cutoff_minus_30d_in_30d": a.purchases_30d == 2 and a.spend_30d == 120.0,
        "minus_30d_minus_1s_in_60d_only": a.purchases_60d == 3 and a.spend_60d == 127.0,
        "exact_cutoff_minus_90d_in_90d": b.purchases_90d == 2 and b.spend_90d == 35.0 and b.product_lines_90d == 2,
        "carry_client_counts": c.purchase_count == 15 and c.total_spend == float(sum(range(1, 16)))
                               and c.product_lines == 30,
    }
    log(f"  synthetic boundary run: parts per chunk_rows {[metas[k]['diagnostics']['parts'] for k in (1, 3, 1000)]}, "
        f"diag {dg}")
    for k, v in checks.items():
        log(f"    {k}: {v}")
    return {"checks": {k: bool(v) for k, v in checks.items()}, "sha256": {str(k): v for k, v in shas.items()},
            "parts": {str(k): metas[k]["diagnostics"]["parts"] for k in shas}, "diagnostics": dg,
            "pass": bool(all(checks.values()))}


def run(log, scratch: Path, real: bool = True) -> dict:
    out = {"synthetic": synthetic_boundary(log, scratch)}
    if real:
        out["real"] = real_rerun(log, scratch)
    out["pass"] = out["synthetic"]["pass"] and (out["real"]["pass"] if real else True)
    return out
