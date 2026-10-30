"""X5 score drivers: which features the final uplift score relies on.
The final R-learner predicts uplift with one XGBoost regressor (models["tau"]); mean |SHAP| (xgboost
pred_contribs) over the train rows, gender dummies summed. total_gain shares are stored as a check.
Descriptive only: shows what the score uses, not why a customer responds. Holdout is not read.
Output: {out_dir}/score_drivers.json, figure figs/score_drivers.png (top 12)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import learners as LR  # noqa: E402
import tune as TU  # noqa: E402

ROOT = TU.ROOT
MODEL = ROOT / "results/final/final_model.pkl"
GENDER = ["gender_F", "gender_M", "gender_U"]


def shares(model, X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    booster = model.models["tau"].get_booster()
    full = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    assert np.allclose(full.sum(axis=1), model.predict_uplift(X), atol=1e-5)   # contributions add up to the score
    shap = pd.Series(np.abs(full[:, :-1]).mean(axis=0), index=X.columns)
    gain = pd.Series(booster.get_score(importance_type="total_gain")).reindex(X.columns).fillna(0.0)
    out = []
    for s in (shap, gain):
        s = s.copy()
        s["gender"] = s[GENDER].sum()
        s = s.drop(GENDER)
        out.append((s / s.sum()).sort_values(ascending=False))
    return out[0], out[1]


def main(out_dir: Path, fig_path: Path | None) -> None:
    model = pickle.load(open(MODEL, "rb"))
    assert model.learner == "r"
    tr = TU.load_train_only()
    X = LR.prepare_matrix(tr["frame"], list(tr["frame"].columns))[model.columns].reset_index(drop=True)
    shap, gain = shares(model, X)
    res = {"model": "final_model.pkl, R-learner final stage (XGBRegressor)", "rows": int(len(X)),
           "metric": "mean |SHAP| share, gender dummies summed", "shap_share": shap.to_dict(),
           "gain_share_check": gain.to_dict(),
           "top12_overlap_shap_vs_gain": len(set(shap.index[:12]) & set(gain.index[:12]))}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "score_drivers.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "gain_share_check"}, indent=2))
    if fig_path:
        plot(shap.head(12), fig_path)


NAMES = {
    "express_points_used": "익스프레스 포인트 사용", "age": "나이", "days_since_issue": "카드 발급 후 경과일",
    "spend_30d": "30일 구매액", "product_lines": "구매 품목 줄 수", "mean_red_amount": "포인트 사용액 평균",
    "purchase_amount_cv": "구매액 변동계수", "spend_per_purchase": "구매당 금액", "median_netto": "상품 중량 중앙값",
    "spend_90d": "90일 구매액", "regular_points_earned": "일반 포인트 적립", "brand_diversity": "브랜드 다양성",
}


def plot(top: pd.Series, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    top = top[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.barh([NAMES.get(k, k) for k in top.index], top.values * 100, color="#C44E52", height=0.62)
    for i, v in enumerate(top.values * 100):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8.5, color="#333333")
    ax.set_xlabel("평균 |SHAP| 비중 (%)")
    ax.set_title("SHAP 기반 변수 중요도", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/post_score_drivers")
    ap.add_argument("--fig", type=Path, default=None)
    a = ap.parse_args()
    main(a.out_dir, a.fig)
