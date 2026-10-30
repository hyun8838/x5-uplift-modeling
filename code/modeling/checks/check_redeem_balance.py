"""Balance check of the three cutoff-based redeem columns (add_redeem_b.py) on data/processed/features_full38.parquet.

SMD ignoring missing; treatment-prediction AUC with out-of-fold XGBoost (StratifiedKFold(5, shuffle,
random_state=20261008), XGB(n_estimators=300, max_depth=3, lr=0.1, hist, n_jobs=8, random_state=0)),
null AUC SE = sqrt((n1+n0+1)/(12 n1 n0)).

- target is never read. treatment_flg comes from the table.
- The redeem columns are taken from the table, not recomputed. The raw redeem columns (positive control only)
  are read from clients.csv and are used only inside this check.
- Pass rule: every redeem column |SMD| < 0.1 AND |z| < 3 for both "base35+redeem3" and "redeem3 only".

Usage: python modeling/checks/check_redeem_balance.py > checks/redeem_balance.out.txt
"""
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "data/processed/features_full38.parquet"
CLIENTS = ROOT / "data/og/retailhero-uplift/data/clients.csv"
OUT_JSON = ROOT / "checks/redeem_balance.json"
REDEEM_COLS = ["redeem_by_cutoff", "days_redeem_to_cutoff", "issue_redeem_delay_cens"]
SEED_CV, SEED_XGB = 20261008, 0
SMD_MAX, Z_MAX = 0.1, 3.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


started = time.time()
f = pd.read_parquet(TABLE)
assert f.shape == (200_039, 40) and "target" not in f.columns
assert f.columns[-3:].tolist() == REDEEM_COLS
w = f.treatment_flg.to_numpy()

cl = pd.read_csv(CLIENTS, usecols=["client_id", "first_issue_date", "first_redeem_date"],
                 dtype={"client_id": str}, parse_dates=["first_issue_date", "first_redeem_date"])
m = f[["client_id"]].merge(cl, on="client_id", how="left", validate="one_to_one")
assert (m.client_id.to_numpy() == f.client_id.to_numpy()).all()
r, iss = m.first_redeem_date, m.first_issue_date
raw = pd.DataFrame({  # raw (uncensored) redeem columns, positive control
    "redeem_isna": r.isna().astype(int),
    "redeem_unixtime": (r - pd.Timestamp("1970-01-01")).dt.total_seconds(),
    "issue_redeem_delay": (r - iss).dt.total_seconds() / 86400,
})
cen = f[REDEEM_COLS].astype(float)
base = f.drop(columns=["client_id", "treatment_flg"] + REDEEM_COLS)
assert base.shape[1] == 35
base = pd.get_dummies(base, columns=["gender"], dtype=float)


def smd(x):
    x = pd.Series(x, dtype=float)
    a, b = x[w == 1], x[w == 0]
    s = math.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / s if s > 0 else 0.0


res = {"smd": {}, "missing_rate": {}}
print(f"table={TABLE.name} rows={len(f):,} n_treat={int(w.sum()):,} n_ctrl={int((1-w).sum()):,}")
print("SMD (missing ignored)")
for df in (cen, raw):
    for c in df:
        res["smd"][c] = smd(df[c]); res["missing_rate"][c] = float(df[c].isna().mean())
        print(f"  {c:26s} {res['smd'][c]:+.4f}  missing={res['missing_rate'][c]:.3f}")

ok = f.redeem_by_cutoff.eq(1).to_numpy()
p0 = w.mean()


def gz(mask):
    n = int(mask.sum())
    return (w[mask].mean() - p0) / math.sqrt(p0 * (1 - p0) / n), n


z1, n1g = gz(ok); z0, n0g = gz(~ok)
res["group"] = {"by_cutoff_1": {"z": z1, "n": n1g}, "by_cutoff_0": {"z": z0, "n": n0g}}
print("group z: by_cutoff=1 %+.2f n=%d | by_cutoff=0 %+.2f n=%d" % (z1, n1g, z0, n0g))

n1, n0 = int(w.sum()), int((1 - w).sum())
se = math.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
skf = StratifiedKFold(5, shuffle=True, random_state=SEED_CV)


def cf_auc(X):
    oof = np.zeros(len(w))
    for tr, te in skf.split(X, w):
        mdl = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, tree_method="hist",
                                n_jobs=8, random_state=SEED_XGB)
        mdl.fit(X.iloc[tr], w[tr]); oof[te] = mdl.predict_proba(X.iloc[te])[:, 1]
    return roc_auc_score(w, oof)


print(f"null AUC SE approx {se:.4f}")
res["auc"] = {}
for name, X in [("base35", base),
                ("base35+redeem3", pd.concat([base, cen], axis=1)),
                ("redeem3 only", cen),
                ("base35+raw3 (positive control)", pd.concat([base, raw], axis=1))]:
    t0 = time.time(); a = cf_auc(X)
    res["auc"][name] = {"auc": a, "z": (a - 0.5) / se, "n_features": X.shape[1], "seconds": round(time.time() - t0, 1)}
    print(f"  {name:32s} p={X.shape[1]:2d} AUC={a:.4f}  z_vs_0.5={(a-0.5)/se:+.2f}  ({res['auc'][name]['seconds']}s)", flush=True)

crit = {f"|SMD| {c} < {SMD_MAX}": abs(res["smd"][c]) < SMD_MAX for c in REDEEM_COLS}
crit[f"|z| base35+redeem3 < {Z_MAX}"] = abs(res["auc"]["base35+redeem3"]["z"]) < Z_MAX
crit[f"|z| redeem3 only < {Z_MAX}"] = abs(res["auc"]["redeem3 only"]["z"]) < Z_MAX
verdict = "PASS" if all(crit.values()) else "FAIL"
pc_ok = res["auc"]["base35+raw3 (positive control)"]["z"] >= Z_MAX
print("criteria:")
for k, v in crit.items():
    print(f"  {k:28s} {'ok' if v else 'NG'}")
print(f"positive control detected (z >= {Z_MAX}): {pc_ok}")
print(f"VERDICT: {verdict}")
elapsed = round(time.time() - started, 1)
print(f"elapsed {elapsed}s")

res.update({
    "criteria": crit, "verdict": verdict, "positive_control_detected": pc_ok,
    "null_auc_se": se, "n_treat": n1, "n_ctrl": n0,
    "seeds": {"StratifiedKFold": SEED_CV, "XGBClassifier": SEED_XGB},
    "xgb_params": {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.1, "tree_method": "hist", "n_jobs": 8},
    "inputs_sha256": {TABLE.name: sha256_file(TABLE), CLIENTS.name: sha256_file(CLIENTS)},
    "script_sha256": sha256_file(Path(__file__)),
    "target_read": False,
    "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                "xgboost": xgb.__version__, "sklearn": sklearn.__version__},
    "elapsed_seconds": elapsed, "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
})
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
assert not OUT_JSON.exists(), f"refusing to overwrite {OUT_JSON}"
tmp = OUT_JSON.with_name(OUT_JSON.name + ".tmp")
tmp.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str) + "\n")
os.replace(tmp, OUT_JSON)
