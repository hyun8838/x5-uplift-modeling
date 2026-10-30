"""X5: build the 35-feature customer table (+client_id, treatment_flg) from raw CSVs in one pass.

Aggregation: product line -> (client_id, transaction_id) transaction -> client.

- uplift_train.csv is read with usecols=['client_id', 'treatment_flg'] only. target is never read.
- No preprocessing: age raw, gender raw string (F/M/U). Dummies are a model-stage step.
- Cutoff is an argument. Purchase rows are kept when transaction_datetime <= cutoff (inclusive);
  window features use transaction_datetime >= cutoff - N days.
- purchases.csv (~45.8M rows) is streamed with pyarrow; rows are filtered to train clients and the
  cutoff before conversion to pandas. The file is sorted by client_id; the last client of each
  buffer is carried to the next buffer so every client is summarized from its complete history.

Usage:
  python modeling/build_features.py --cutoff "2019-03-18 23:40:03" --out data/processed/features_cut.parquet
  python modeling/build_features.py --smoke   # synthetic unit checks
  python modeling/build_features.py --cutoff ... --out <scratch> --max-raw-rows 3000000   # sample run
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
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/og/retailhero-uplift/data"
INPUTS = ("clients.csv", "products.csv", "purchases.csv", "uplift_train.csv")
EXPECTED_RAW_ROWS = 45_786_568
EXPECTED_CLIENTS = 200_039
WINDOWS = (30, 60, 90)

COLUMNS = [
    "client_id", "treatment_flg",
    "shopping_days", "alcohol_share", "days_since_issue", "days_since_purchase",
    "express_points_earned", "express_points_used", "product_lines_90d", "product_lines",
    "lines_per_purchase", "own_brand_share", "zero_qty_share", "regular_points_earned",
    "regular_points_used", "spend_per_purchase", "spend_30d", "spend_60d", "spend_90d",
    "total_spend", "purchases_30d", "purchases_60d", "purchases_90d", "purchase_count",
    "category_count", "product_count", "store_count", "age", "gender",
    "median_netto", "median_iss_amount", "mean_red_amount",
    "purchase_gap_cv", "purchase_trend_30d", "purchase_amount_cv", "top_store_share", "brand_diversity",
]
TX_FIELDS = [
    "transaction_datetime", "purchase_sum", "store_id",
    "regular_points_received", "regular_points_spent",
    "express_points_received", "express_points_spent",
]
PURCHASE_TYPES = {
    "client_id": pa.string(), "transaction_id": pa.string(),
    "transaction_datetime": pa.timestamp("s"),
    "regular_points_received": pa.float64(), "express_points_received": pa.float64(),
    "regular_points_spent": pa.float64(), "express_points_spent": pa.float64(),
    "purchase_sum": pa.float64(), "store_id": pa.string(), "product_id": pa.string(),
    "product_quantity": pa.float64(), "trn_sum_from_iss": pa.float64(), "trn_sum_from_red": pa.float64(),
}


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


def load_dimensions(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    official = pd.read_csv(
        source / "uplift_train.csv", usecols=["client_id", "treatment_flg"],
        dtype={"client_id": str, "treatment_flg": "int8"},
    )
    assert list(official.columns) == ["client_id", "treatment_flg"]  # target never loaded
    assert official.client_id.is_unique and set(official.treatment_flg.unique()) <= {0, 1}
    clients = pd.read_csv(
        source / "clients.csv", usecols=["client_id", "first_issue_date", "age", "gender"],
        dtype={"client_id": str, "gender": str}, parse_dates=["first_issue_date"],
    )
    products = pd.read_csv(
        source / "products.csv",
        usecols=["product_id", "level_2", "brand_id", "netto", "is_own_trademark", "is_alcohol"],
        dtype={"product_id": str, "level_2": str, "brand_id": str},
    )
    assert clients.client_id.is_unique and products.product_id.is_unique
    assert clients.first_issue_date.notna().all()
    # missing level_2 -> "UNKNOWN", then UNKNOWN excluded from category_count
    products["level_2_div"] = products.level_2.fillna("UNKNOWN").mask(lambda s: s.eq("UNKNOWN"))
    products["is_own_line"] = products.is_own_trademark.astype("int8")
    products["is_alcohol_line"] = products.is_alcohol.astype("int8")
    products = products[["product_id", "level_2_div", "brand_id", "netto", "is_own_line", "is_alcohol_line"]]
    return official, clients, products


def summarize(lines: pd.DataFrame, products: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Per-client summary for clients whose full (pre-cutoff) history is in `lines`."""
    n_before = len(lines)
    lines = lines.merge(products, on="product_id", how="left", validate="many_to_one", indicator=True)
    if (lines["_merge"] != "both").any():
        raise AssertionError(f"product_id unmatched: {int((lines['_merge'] != 'both').sum())}")
    lines = lines.drop(columns="_merge")
    assert len(lines) == n_before

    # ---- transaction grain: (client_id, transaction_id); transaction fields must be constant
    gtx = lines.groupby(["client_id", "transaction_id"], sort=False, dropna=False)
    bad = gtx[TX_FIELDS].nunique(dropna=False).ne(1).any(axis=1)
    if bad.any():
        raise AssertionError(f"transaction fields differ within composite key: {int(bad.sum())}")
    tx = gtx[TX_FIELDS].first().reset_index()
    assert tx.regular_points_spent.max() <= 0 and tx.express_points_spent.max() <= 0

    tx["tx_day"] = tx.transaction_datetime.dt.floor("D")
    for d in WINDOWS:
        m = tx.transaction_datetime >= cutoff - pd.Timedelta(days=d)
        tx[f"in_{d}d"] = m.astype("int64")
        tx[f"spend_{d}d"] = tx.purchase_sum.where(m, 0.0)
    gt = tx.groupby("client_id", sort=False)
    out = gt.agg(
        purchase_count=("transaction_id", "size"),
        purchases_30d=("in_30d", "sum"),
        purchases_60d=("in_60d", "sum"),
        purchases_90d=("in_90d", "sum"),
        shopping_days=("tx_day", "nunique"),
        total_spend=("purchase_sum", "sum"),
        spend_30d=("spend_30d", "sum"),
        spend_60d=("spend_60d", "sum"),
        spend_90d=("spend_90d", "sum"),
        store_count=("store_id", "nunique"),
        last_tx=("transaction_datetime", "max"),
        regular_points_earned=("regular_points_received", "sum"),
        regular_points_used=("regular_points_spent", "sum"),
        express_points_earned=("express_points_received", "sum"),
        express_points_used=("express_points_spent", "sum"),
    )
    out["regular_points_used"] = -out["regular_points_used"]
    out["express_points_used"] = -out["express_points_used"]

    # gap / amount CV (rule: >=3 gaps or >=3 transactions, mean > 0, ddof=1)
    txs = tx.sort_values(["client_id", "transaction_datetime", "transaction_id"])
    txs["gap_days"] = txs.groupby("client_id", sort=False).transaction_datetime.diff().dt.total_seconds() / 86400
    gap = txs.groupby("client_id", sort=False).gap_days.agg(["count", "mean", "std"])
    out["purchase_gap_cv"] = (gap["std"] / gap["mean"]).where((gap["count"] >= 3) & (gap["mean"] > 0))
    amt = gt.purchase_sum.agg(["count", "mean", "std"])
    out["purchase_amount_cv"] = (amt["std"] / amt["mean"]).where((amt["count"] >= 3) & (amt["mean"] > 0))
    valid = tx.loc[tx.store_id.notna()]
    store_n = valid.groupby("client_id", sort=False).size()
    top_n = valid.groupby(["client_id", "store_id"], sort=False).size().groupby(level=0).max()
    out["top_store_share"] = top_n / store_n

    # ---- line grain
    lines["in_90d"] = (lines.transaction_datetime >= cutoff - pd.Timedelta(days=90)).astype("int64")
    lines["is_qty_zero_line"] = (lines.product_quantity == 0).astype("int8")
    gl = lines.groupby("client_id", sort=False)
    ls = gl.agg(
        product_lines=("product_id", "size"),
        product_lines_90d=("in_90d", "sum"),
        product_count=("product_id", "nunique"),
        category_count=("level_2_div", "nunique"),
        own_lines=("is_own_line", "sum"),
        alcohol_lines=("is_alcohol_line", "sum"),
        qty_zero_lines=("is_qty_zero_line", "sum"),
        median_netto=("netto", "median"),
        median_iss_amount=("trn_sum_from_iss", "median"),
        mean_red_amount=("trn_sum_from_red", "mean"),
    )
    unique = lines[["client_id", "product_id", "brand_id"]].drop_duplicates(["client_id", "product_id"])
    known = unique.loc[unique.brand_id.notna()]
    known_n = known.groupby("client_id", sort=False).size()
    brand_n = known.groupby("client_id", sort=False).brand_id.nunique()
    ls["brand_diversity"] = brand_n / known_n  # NaN when no product has a known brand

    out = out.join(ls, how="inner", validate="one_to_one")
    assert len(out) == lines.client_id.nunique() == tx.client_id.nunique()
    return out.reset_index()


def assemble(official: pd.DataFrame, clients: pd.DataFrame, summary: pd.DataFrame,
             cutoff: pd.Timestamp, allow_missing: bool = False) -> pd.DataFrame:
    out = official.merge(clients, on="client_id", how="left", validate="one_to_one", indicator=True)
    assert (out["_merge"] == "both").all(), "train client missing in clients.csv"
    out = out.drop(columns="_merge")
    out = out.merge(summary, on="client_id", how="left" if allow_missing else "inner", validate="one_to_one")
    if not allow_missing:
        assert len(out) == len(official), "train client without purchase history"
    out["days_since_issue"] = (cutoff - out.first_issue_date).dt.total_seconds() / 86400.0
    out["days_since_purchase"] = (cutoff - out.last_tx).dt.total_seconds() / 86400.0
    n_tx = out.purchase_count.replace(0, np.nan)
    out["spend_per_purchase"] = out.total_spend.div(n_tx)
    out["lines_per_purchase"] = out.product_lines.div(n_tx)
    n_lines = out.product_lines.replace(0, np.nan)
    out["own_brand_share"] = out.own_lines.div(n_lines)
    out["alcohol_share"] = out.alcohol_lines.div(n_lines)
    out["zero_qty_share"] = out.qty_zero_lines.div(n_lines)
    recent = out.purchases_30d.astype("float64")
    previous = out.purchases_60d.astype("float64") - recent
    assert previous.dropna().ge(0).all()
    out["purchase_trend_30d"] = np.log1p(recent) - np.log1p(previous)
    out = out[COLUMNS]
    assert out.client_id.is_unique and "target" not in out.columns
    assert not any("redeem" in c for c in out.columns)
    assert out.days_since_issue.min() > 0
    assert out.days_since_purchase.dropna().min() >= 0
    return out


def iter_purchases(path: Path, block_size: int):
    reader = pacsv.open_csv(
        path, read_options=pacsv.ReadOptions(block_size=block_size),
        convert_options=pacsv.ConvertOptions(include_columns=list(PURCHASE_TYPES), column_types=PURCHASE_TYPES),
    )
    for batch in reader:
        yield pa.Table.from_batches([batch])


def run(cutoff: pd.Timestamp, out_path: Path, chunk_rows: int, max_raw_rows: int | None,
        write_meta: bool) -> None:
    started = time.time()
    official, clients, products = load_dimensions(SOURCE)
    train_set = pa.array(sorted(official.client_id), type=pa.string())
    cutoff_pa = pa.scalar(cutoff.to_pydatetime(), type=pa.timestamp("s"))

    diag = {"raw_rows": 0, "train_rows": 0, "train_rows_after_cutoff": 0,
            "train_rows_at_cutoff_exact": 0, "selected_lines": 0, "parts": 0}
    pieces: list[pd.DataFrame] = []
    buffer: list[pa.Table] = []
    buffered = 0
    carry: pa.Table | None = None
    last_raw_client = None
    prev_done_client = None

    def flush(final: bool) -> None:
        nonlocal carry, buffer, buffered, prev_done_client
        parts = ([carry] if carry is not None else []) + buffer
        buffer, buffered = [], 0
        parts = [t for t in parts if t.num_rows]
        if not parts:
            carry = None
            return
        table = pa.concat_tables(parts)
        ids = table["client_id"]
        if final:
            complete, carry = table, None
        else:
            last = ids[-1]
            start = pc.index(ids, last).as_py()
            if start == 0:
                carry = table
                return
            complete, carry = table.slice(0, start), table.slice(start)
        df = complete.to_pandas()
        assert df.client_id.is_monotonic_increasing
        if prev_done_client is not None:
            assert df.client_id.iloc[0] > prev_done_client, "client split across completed parts"
        prev_done_client = df.client_id.iloc[-1]
        pieces.append(summarize(df, products, cutoff))
        diag["parts"] += 1
        print(f"part {diag['parts']:03d}: lines={len(df):,} clients={len(pieces[-1]):,} "
              f"raw_scanned={diag['raw_rows']:,} elapsed={time.time()-started:.0f}s", flush=True)

    for table in iter_purchases(SOURCE / "purchases.csv", 1 << 25):
        if max_raw_rows is not None and diag["raw_rows"] >= max_raw_rows:
            break
        ids = table["client_id"]
        # source sort order check (raw, before filtering)
        first = ids[0].as_py()
        if last_raw_client is not None:
            assert first >= last_raw_client, "purchases.csv not sorted by client_id"
        last_raw_client = ids[-1].as_py()
        diag["raw_rows"] += table.num_rows
        table = table.filter(pc.is_in(ids, value_set=train_set))
        diag["train_rows"] += table.num_rows
        ts = table["transaction_datetime"]
        assert ts.null_count == 0
        diag["train_rows_after_cutoff"] += int(pc.sum(pc.greater(ts, cutoff_pa)).as_py() or 0)
        diag["train_rows_at_cutoff_exact"] += int(pc.sum(pc.equal(ts, cutoff_pa)).as_py() or 0)
        table = table.filter(pc.less_equal(ts, cutoff_pa))
        diag["selected_lines"] += table.num_rows
        buffer.append(table)
        buffered += table.num_rows
        if buffered >= chunk_rows:
            flush(final=False)
    flush(final=True)

    summary = pd.concat(pieces, ignore_index=True)
    assert summary.client_id.is_unique
    smoke_run = max_raw_rows is not None
    if smoke_run:
        official = official.loc[official.client_id.isin(summary.client_id)].reset_index(drop=True)
    else:
        assert diag["raw_rows"] == EXPECTED_RAW_ROWS, diag["raw_rows"]
        assert len(official) == EXPECTED_CLIENTS
    final = assemble(official, clients, summary, cutoff)
    assert int(final.product_lines.sum()) == diag["selected_lines"]
    final = final.sort_values("client_id", ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite {out_path}")
    tmp = out_path.with_name(out_path.name + ".tmp")
    final.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    elapsed = round(time.time() - started, 1)
    print(f"DONE rows={len(final):,} cols={final.shape[1]} elapsed={elapsed}s", flush=True)
    if not write_meta:
        return
    meta = {
        "status": "features_no_target",
        "rows": len(final), "n_columns": final.shape[1], "columns": list(final.columns),
        "dtypes": {c: str(t) for c, t in final.dtypes.items()},
        "cutoff": str(cutoff),
        "cutoff_rule": "purchase line kept if transaction_datetime <= cutoff; window Nd = transaction_datetime >= cutoff - N days",
        "target_read": False,
        "input_sha256": {name: sha256_file(SOURCE / name) for name in INPUTS},
        "script_sha256": sha256_file(Path(__file__)),
        "output_sha256": sha256_file(out_path),
        "diagnostics": diag,
        "missing": {c: int(v) for c, v in final.isna().sum().items() if v},
        "chunk_rows": chunk_rows, "random_seed": None,
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__,
                    "numpy": np.__version__, "pyarrow": pa.__version__},
        "elapsed_seconds_build": elapsed,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    meta_path = out_path.with_name(out_path.stem + ".meta.json")
    if meta_path.exists():
        raise FileExistsError(f"refusing to overwrite {meta_path}")
    atomic_write_json(meta_path, meta)


def smoke() -> None:
    cutoff = pd.Timestamp("2019-03-19 00:00:00")
    products = pd.DataFrame({
        "product_id": ["p1", "p2", "p3"], "level_2": ["a", None, "b"], "brand_id": ["b1", None, "b1"],
        "netto": [0.5, 1.0, np.nan], "is_own_trademark": [1, 0, 0], "is_alcohol": [0, 1, 0],
    })
    products["level_2_div"] = products.level_2.fillna("UNKNOWN").mask(lambda s: s.eq("UNKNOWN"))
    products["is_own_line"] = products.is_own_trademark.astype("int8")
    products["is_alcohol_line"] = products.is_alcohol.astype("int8")
    products = products[["product_id", "level_2_div", "brand_id", "netto", "is_own_line", "is_alcohol_line"]]
    t = pd.to_datetime
    lines = pd.DataFrame({
        "client_id": ["a", "a", "a", "a", "b"],
        "transaction_id": ["t1", "t1", "t2", "t3", "t1"],
        "transaction_datetime": t(["2019-03-18 12:00:00"] * 2 + ["2018-12-01 00:00:00", "2019-02-10 00:00:00", "2019-03-18 12:00:00"]),
        "purchase_sum": [100.0, 100.0, 50.0, 30.0, 40.0],
        "store_id": ["s1", "s1", "s2", "s1", "s2"],
        "product_id": ["p1", "p2", "p3", "p1", "p1"],
        "product_quantity": [0.0, 1.0, 1.0, 2.0, 0.0],
        "regular_points_received": [1.0, 1.0, 0.5, 0.0, 2.0],
        "regular_points_spent": [0.0, 0.0, 0.0, 0.0, -3.0],
        "express_points_received": [0.0] * 5, "express_points_spent": [0.0] * 5,
        "trn_sum_from_iss": [10.0, 20.0, 30.0, 40.0, 5.0],
        "trn_sum_from_red": [np.nan, 4.0, np.nan, 6.0, np.nan],
    })
    s = summarize(lines, products, cutoff)
    official = pd.DataFrame({"client_id": ["a", "b"], "treatment_flg": np.array([0, 1], dtype="int8")})
    clients = pd.DataFrame({"client_id": ["a", "b"], "first_issue_date": t(["2018-01-01"] * 2),
                            "age": [30, 40], "gender": ["F", "U"]})
    o = assemble(official, clients, s, cutoff).set_index("client_id")
    a = o.loc["a"]
    assert a.total_spend == 180.0 and a.purchase_count == 3 and a.product_lines == 4
    assert a.spend_30d == 100.0 and a.purchases_30d == 1 and a.purchases_60d == 2 and a.purchases_90d == 2
    assert a.spend_90d == 130.0 and a.product_lines_90d == 3
    assert a.category_count == 2 and a.product_count == 3 and a.store_count == 2
    assert a.zero_qty_share == 0.25 and a.own_brand_share == 0.5 and a.alcohol_share == 0.25
    assert a.median_netto == 0.5 and a.median_iss_amount == 25.0 and a.mean_red_amount == 5.0
    assert a.brand_diversity == 0.5  # known-brand unique products p1,p3 -> 1 brand / 2
    assert abs(a.top_store_share - 2 / 3) < 1e-12
    assert np.isnan(a.purchase_gap_cv) and abs(a.purchase_amount_cv - np.std([100, 50, 30], ddof=1) / 60) < 1e-12
    assert abs(a.purchase_trend_30d - (np.log1p(1) - np.log1p(1))) < 1e-12
    assert a.days_since_purchase == 0.5
    b = o.loc["b"]
    assert b.total_spend == 40.0 and b.regular_points_used == 3.0 and np.isnan(b.mean_red_amount)
    print("smoke OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--chunk-rows", type=int, default=1_500_000)
    ap.add_argument("--max-raw-rows", type=int, default=None, help="sample run: stop after ~N raw rows")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        assert args.cutoff and args.out
        run(pd.Timestamp(args.cutoff), args.out.resolve(), args.chunk_rows, args.max_raw_rows,
            write_meta=args.max_raw_rows is None)
