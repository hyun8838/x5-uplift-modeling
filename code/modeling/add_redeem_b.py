"""X5: add the three cutoff-based redeem columns to the canonical 35-feature table.

Input : data/processed/features_cut.parquet (200,039 x 37, cutoff 2019-03-18 23:40:03)
        data/og/retailhero-uplift/data/clients.csv (client_id, first_issue_date, first_redeem_date only)
Output: data/processed/features_full38.parquet (+ .meta.json)

- target is never read; uplift_train.csv is not opened (treatment_flg comes from the input table).
- Boundary is inclusive like build_features.py: redeem counts when first_redeem_date <= cutoff.
    redeem_by_cutoff        = 1 if first_redeem_date <= cutoff else 0 (later or missing -> 0)
    days_redeem_to_cutoff   = (cutoff - first_redeem_date) / 86400 s, NaN when redeem_by_cutoff == 0
    issue_redeem_delay_cens = (first_redeem_date - first_issue_date) / 86400 s, NaN when redeem_by_cutoff == 0
- Raw redeem columns (first_redeem_date etc.) are not written. first_issue_date is not written either
  (days_since_issue is already in the input table).
- Existing outputs are never overwritten.

Usage:
  python modeling/add_redeem_b.py --smoke
  python modeling/add_redeem_b.py
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

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/og/retailhero-uplift/data"
IN_TABLE = ROOT / "data/processed/features_cut.parquet"
OUT_TABLE = ROOT / "data/processed/features_full38.parquet"
CUTOFF = pd.Timestamp("2019-03-18 23:40:03")
EXPECTED_CLIENTS = 200_039
NEW_COLUMNS = ["redeem_by_cutoff", "days_redeem_to_cutoff", "issue_redeem_delay_cens"]


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


def redeem_b(issue: pd.Series, redeem: pd.Series, cutoff: pd.Timestamp) -> pd.DataFrame:
    ok = (redeem <= cutoff).fillna(False).astype(bool)  # NaT compares False
    to_cut = (cutoff - redeem).dt.total_seconds() / 86400.0
    delay = (redeem - issue).dt.total_seconds() / 86400.0
    return pd.DataFrame({
        "redeem_by_cutoff": ok.astype("int8"),
        "days_redeem_to_cutoff": to_cut.where(ok),
        "issue_redeem_delay_cens": delay.where(ok),
    }, index=redeem.index)


def build(base: pd.DataFrame, clients: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    assert base.client_id.is_unique and clients.client_id.is_unique
    assert "target" not in base.columns and not any("redeem" in c for c in base.columns)
    m = base[["client_id"]].merge(clients, on="client_id", how="left", validate="one_to_one", indicator=True)
    assert (m["_merge"] == "both").all(), "client missing in clients.csv"
    assert (m.client_id.to_numpy() == base.client_id.to_numpy()).all()
    assert m.first_issue_date.notna().all()
    new = redeem_b(m.first_issue_date, m.first_redeem_date, cutoff)
    out = pd.concat([base.reset_index(drop=True), new.reset_index(drop=True)], axis=1)
    assert len(out) == len(base)
    assert out.columns.tolist() == base.columns.tolist() + NEW_COLUMNS
    # consistency
    r = out.redeem_by_cutoff.eq(1)
    assert out.days_redeem_to_cutoff.notna().eq(r).all() and out.issue_redeem_delay_cens.notna().eq(r).all()
    assert out.days_redeem_to_cutoff.dropna().ge(0).all()
    return out


def run() -> None:
    started = time.time()
    base = pd.read_parquet(IN_TABLE)
    assert base.shape == (EXPECTED_CLIENTS, 37), base.shape
    clients = pd.read_csv(
        SOURCE / "clients.csv", usecols=["client_id", "first_issue_date", "first_redeem_date"],
        dtype={"client_id": str}, parse_dates=["first_issue_date", "first_redeem_date"],
    )
    out = build(base, clients, CUTOFF)
    r_all = clients.set_index("client_id").loc[base.client_id, "first_redeem_date"]
    diag = {
        "n_redeem_le_cutoff": int(out.redeem_by_cutoff.sum()),
        "n_redeem_after_cutoff": int((r_all > CUTOFF).sum()),
        "n_redeem_missing": int(r_all.isna().sum()),
        "n_redeem_exact_cutoff": int((r_all == CUTOFF).sum()),
        "first_redeem_date_max": str(r_all.max()),
        "issue_redeem_delay_cens_min": float(out.issue_redeem_delay_cens.min()),
        "n_issue_redeem_delay_cens_negative": int(out.issue_redeem_delay_cens.lt(0).sum()),
    }
    for p in (OUT_TABLE, OUT_TABLE.with_name(OUT_TABLE.stem + ".meta.json")):
        if p.exists():
            raise FileExistsError(f"refusing to overwrite {p}")
    tmp = OUT_TABLE.with_name(OUT_TABLE.name + ".tmp")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, OUT_TABLE)
    elapsed = round(time.time() - started, 1)
    print(f"DONE rows={len(out):,} cols={out.shape[1]} elapsed={elapsed}s", flush=True)
    print(json.dumps(diag, indent=2), flush=True)
    meta = {
        "status": "features_full38_no_target",
        "rows": len(out), "n_columns": out.shape[1], "columns": list(out.columns),
        "dtypes": {c: str(t) for c, t in out.dtypes.items()},
        "cutoff": str(CUTOFF),
        "cutoff_rule": "redeem_by_cutoff = first_redeem_date <= cutoff (inclusive); later or missing -> 0, "
                       "and the two day columns are NaN",
        "added_columns": NEW_COLUMNS,
        "raw_redeem_columns_written": False,
        "target_read": False,
        "input_sha256": {"features_cut.parquet": sha256_file(IN_TABLE),
                         "clients.csv": sha256_file(SOURCE / "clients.csv")},
        "script_sha256": sha256_file(Path(__file__)),
        "output_sha256": sha256_file(OUT_TABLE),
        "diagnostics": diag,
        "missing": {c: int(v) for c, v in out.isna().sum().items() if v},
        "random_seed": None,
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "elapsed_seconds_build": elapsed,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    atomic_write_json(OUT_TABLE.with_name(OUT_TABLE.stem + ".meta.json"), meta)


def smoke() -> None:
    t = pd.to_datetime
    cut = pd.Timestamp("2019-03-18 23:40:03")
    base = pd.DataFrame({"client_id": list("abcd"), "treatment_flg": np.array([0, 1, 0, 1], dtype="int8")})
    clients = pd.DataFrame({
        "client_id": list("dcba"),
        "first_issue_date": t(["2018-01-01"] * 4),
        "first_redeem_date": t(["2019-03-18 23:40:03", None, "2019-03-19 00:00:00", "2019-03-17 23:40:03"]),
    })
    o = build(base, clients, cut).set_index("client_id")
    assert o.redeem_by_cutoff.tolist() == [1, 0, 0, 1]  # a: before, b: after, c: missing, d: exact (inclusive)
    assert o.loc["a", "days_redeem_to_cutoff"] == 1.0 and o.loc["d", "days_redeem_to_cutoff"] == 0.0
    assert np.isnan(o.loc["b", "days_redeem_to_cutoff"]) and np.isnan(o.loc["c", "issue_redeem_delay_cens"])
    expect = (t("2019-03-17 23:40:03") - t("2018-01-01")).total_seconds() / 86400
    assert abs(o.loc["a", "issue_redeem_delay_cens"] - expect) < 1e-12
    assert o.index.tolist() == list("abcd")
    print("smoke OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    smoke() if args.smoke else run()
