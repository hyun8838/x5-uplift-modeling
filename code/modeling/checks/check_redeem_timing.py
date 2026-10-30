"""Timing of first_redeem_date relative to the purchase window, by treatment share.

first_redeem_date in clients.csv extends past the last purchase date; this check shows how treatment share
differs between customers whose first redeem falls inside / after the window / is missing, and whether
points-spending in purchases.csv is consistent with it. target is not read.

Usage: python modeling/checks/check_redeem_timing.py data/og/retailhero-uplift/data > checks/redeem_timing.out.txt
"""
import sys, math
import pyarrow.csv as pc, pyarrow.compute as pcc, pandas as pd
D = sys.argv[1]
pur = pc.read_csv(f"{D}/purchases.csv", convert_options=pc.ConvertOptions(
    include_columns=["client_id","transaction_id","transaction_datetime","regular_points_spent","express_points_spent"]))
print("purchases rows", pur.num_rows)
ts = pur["transaction_datetime"]
print("window", pcc.min(ts), pcc.max(ts))
df = pur.to_pandas(); del pur
df["spent"] = (df.regular_points_spent != 0) | (df.express_points_spent != 0)
g = df.groupby("client_id").agg(first_tx=("transaction_datetime","min"), last_tx=("transaction_datetime","max"))
sp = df[df.spent].groupby("client_id")["transaction_datetime"].min().rename("first_spent_tx")
g = g.join(sp)
cl = pd.read_csv(f"{D}/clients.csv", parse_dates=["first_issue_date","first_redeem_date"])
tr = pd.read_csv(f"{D}/uplift_train.csv", usecols=["client_id","treatment_flg"])  # target not read
m = tr.merge(cl, on="client_id", how="left").merge(g, on="client_id", how="left")
end = ts_end = df.transaction_datetime.max()
p0 = m.treatment_flg.mean(); N = len(m)
def grp(name, mask):
    n = mask.sum(); r = m.loc[mask,"treatment_flg"].mean()
    z = (r - p0) / math.sqrt(p0*(1-p0)/n)
    print(f"{name:40s} n={n:7d} ({n/N:6.2%}) treat={r:.4f} z={z:+.2f}")
print("train N", N, "treat share", round(p0,5))
grp("redeem <= window end", m.first_redeem_date <= end)
grp("redeem > window end", m.first_redeem_date > end)
grp("redeem missing", m.first_redeem_date.isna())
print("first_issue max", cl.first_issue_date.max(), "| redeem max", cl.first_redeem_date.max())
# consistency: points spent in purchases vs first_redeem_date
a = m[m.first_redeem_date > end]
print("redeem>end & spent points within window:", a.first_spent_tx.notna().sum(), "of", len(a))
b = m[m.first_redeem_date.isna()]
print("redeem missing & spent points within window:", b.first_spent_tx.notna().sum(), "of", len(b))
c = m[m.first_redeem_date <= end]
print("redeem<=end & first_spent_tx < first_redeem (by >1 day):", (c.first_spent_tx < c.first_redeem_date - pd.Timedelta(days=1)).sum(), "of", len(c))
print("redeem<=end & no points spent in window:", c.first_spent_tx.isna().sum())
