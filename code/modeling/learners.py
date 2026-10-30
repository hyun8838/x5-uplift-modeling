"""X5 uplift learners.

7 learners x 2 engines, one interface (28 candidates = 7 x 2 engines x 2 tracks):

    model = fit(learner, X, w, y, params, engine, seed, n_jobs=1)
    tau = model.predict_uplift(X_new)

Rules:
- No propensity model. e = mean(w) of the training data is used as a constant (randomized experiment).
  It enters X (stage-2 blend), R (residual w - e), DR (IPW terms), 4-class (P(z)/e).
- All sub-models of one learner share one parameter set. Classification of Y (and the 4-class / class
  transformation targets) uses logistic / softmax loss; pseudo-outcome and residual regressions use squared error.
- Parameter mapping, common meaning across engines:
      depth            XGB max_depth           | LGB max_depth = depth, num_leaves = 2**depth
      learning_rate    learning_rate           | learning_rate
      n_estimators     n_estimators            | n_estimators
      reg_lambda       reg_lambda (L2)         | reg_lambda (lambda_l2)
      min_leaf_hessian min_child_weight        | min_child_weight (= min_sum_hessian_in_leaf)
      subsample        subsample               | subsample (= bagging_fraction), subsample_freq (= bagging_freq) = 1
      colsample        colsample_bytree        | colsample_bytree (= feature_fraction)
  LGB min_child_samples (= min_data_in_leaf) is fixed to LGB_MIN_DATA_IN_LEAF = 1:
  XGBoost has no row-count leaf constraint, so with 1 the leaf size is governed by min_leaf_hessian
  in both engines. The argument `lgb_min_data_in_leaf` defaults to 1; passing None
  explicitly leaves LightGBM's own default (20) and is used only by the diagnostic probe in tests.
- Preprocessing (`prepare_matrix`) is row-wise only (no statistic learned from data): gender F/M/U -> 3 dummies,
  age raw, other columns cast to float, missing left as NaN (engine default handling). Safe inside any fold.

Implementation:
- R and DR need out-of-sample nuisance predictions: m(x)=E[Y|x] (R) and m0, m1 (DR) are cross-fitted with
  StratifiedKFold(CROSSFIT_FOLDS=3, stratify 2w+y, random_state=seed) inside the learner's training data,
  using the shared parameter set. X uses the standard Künzel construction (m0 fitted on controls is applied to
  treated rows and vice versa, which is already out-of-group), so no cross-fitting there.
- Sample weights (R: (w-e)^2) are rescaled to mean 1 so that min_leaf_hessian and
  reg_lambda keep the same scale as unweighted learners. Rescaling by a constant does not change the minimizer
  of the weighted loss without regularization.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

LEARNERS = ("s", "t", "x", "r", "dr", "four_class", "lai")
ENGINES = ("xgboost", "lightgbm")
PARAM_KEYS = ("depth", "learning_rate", "n_estimators", "reg_lambda", "min_leaf_hessian", "subsample", "colsample")
GENDER_LEVELS = ("F", "M", "U")
CROSSFIT_FOLDS = 3
LGB_MIN_DATA_IN_LEAF = 1
TREATMENT_COL = "_treatment"

# 38 features of data/processed/features_full38.parquet (gender counted once, before dummies)
FEATURES_38 = [
    "shopping_days", "alcohol_share", "days_since_issue", "days_since_purchase",
    "express_points_earned", "express_points_used", "product_lines_90d", "product_lines",
    "lines_per_purchase", "own_brand_share", "zero_qty_share", "regular_points_earned",
    "regular_points_used", "spend_per_purchase", "spend_30d", "spend_60d", "spend_90d",
    "total_spend", "purchases_30d", "purchases_60d", "purchases_90d", "purchase_count",
    "category_count", "product_count", "store_count", "age", "gender",
    "median_netto", "median_iss_amount", "mean_red_amount",
    "purchase_gap_cv", "purchase_trend_30d", "purchase_amount_cv", "top_store_share", "brand_diversity",
    "redeem_by_cutoff", "days_redeem_to_cutoff", "issue_redeem_delay_cens",
]
assert len(FEATURES_38) == 38 and len(set(FEATURES_38)) == 38


# ----------------------------------------------------------------------------- preprocessing
def prepare_matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Row-wise transform only. gender -> gender_F/M/U (all three kept), everything else float."""
    assert len(cols) == len(set(cols)), "duplicate columns"
    forbidden = {"client_id", "treatment_flg", "target", TREATMENT_COL}
    assert not forbidden & set(cols), forbidden & set(cols)
    out = {}
    for c in cols:
        if c == "gender":
            g = frame[c].astype(object)
            assert g.notna().all(), "gender missing"
            bad = set(g.unique()) - set(GENDER_LEVELS)
            assert not bad, f"unknown gender level {bad}"
            for lv in GENDER_LEVELS:
                out[f"gender_{lv}"] = (g == lv).to_numpy(dtype=float)
        else:
            assert pd.api.types.is_numeric_dtype(frame[c].dtype), c
            out[c] = frame[c].to_numpy(dtype=float)
    x = pd.DataFrame(out, index=frame.index)
    assert not np.isinf(x.to_numpy()).any(), "inf in features"
    return x


# ----------------------------------------------------------------------------- engines
def check_params(params: dict) -> dict:
    missing = set(PARAM_KEYS) - set(params)
    extra = set(params) - set(PARAM_KEYS)
    assert not missing and not extra, (missing, extra)
    p = dict(params)
    p["depth"] = int(p["depth"])
    p["n_estimators"] = int(p["n_estimators"])
    assert 1 <= p["depth"] <= 16 and p["n_estimators"] >= 1
    assert 0 < p["subsample"] <= 1 and 0 < p["colsample"] <= 1
    return p


def engine_kwargs(engine: str, params: dict, seed: int, n_jobs: int,
                  lgb_min_data_in_leaf: int | None = LGB_MIN_DATA_IN_LEAF) -> dict:
    p = check_params(params)
    if engine == "xgboost":
        return dict(
            n_estimators=p["n_estimators"], max_depth=p["depth"], learning_rate=p["learning_rate"],
            reg_lambda=p["reg_lambda"], min_child_weight=p["min_leaf_hessian"],
            subsample=p["subsample"], colsample_bytree=p["colsample"],
            tree_method="hist", random_state=seed, n_jobs=n_jobs, verbosity=0,
        )
    if engine == "lightgbm":
        kw = dict(
            n_estimators=p["n_estimators"], max_depth=p["depth"], num_leaves=2 ** p["depth"],
            learning_rate=p["learning_rate"], reg_lambda=p["reg_lambda"],
            min_child_weight=p["min_leaf_hessian"],
            subsample=p["subsample"], subsample_freq=1, colsample_bytree=p["colsample"],
            random_state=seed, n_jobs=n_jobs, verbosity=-1, deterministic=True, force_row_wise=True,
        )
        if lgb_min_data_in_leaf is not None:
            kw["min_child_samples"] = int(lgb_min_data_in_leaf)
        return kw
    raise ValueError(engine)


@dataclass
class EngineSpec:
    engine: str
    params: dict
    seed: int
    n_jobs: int = 1
    lgb_min_data_in_leaf: int | None = LGB_MIN_DATA_IN_LEAF

    def classifier(self, num_class: int = 2):
        kw = engine_kwargs(self.engine, self.params, self.seed, self.n_jobs, self.lgb_min_data_in_leaf)
        if self.engine == "xgboost":
            if num_class == 2:
                kw.update(objective="binary:logistic", eval_metric="logloss")
            else:
                kw.update(objective="multi:softprob", num_class=num_class, eval_metric="mlogloss")
            return XGBClassifier(**kw)
        kw.update(objective="binary" if num_class == 2 else "multiclass")
        if num_class != 2:
            kw["num_class"] = num_class
        return LGBMClassifier(**kw)

    def regressor(self):
        kw = engine_kwargs(self.engine, self.params, self.seed, self.n_jobs, self.lgb_min_data_in_leaf)
        if self.engine == "xgboost":
            return XGBRegressor(objective="reg:squarederror", **kw)
        return LGBMRegressor(objective="regression", **kw)


def _prob(model, x: pd.DataFrame) -> np.ndarray:
    assert list(model.classes_) == [0, 1]
    p = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    assert np.isfinite(p).all()
    return p


def _pred(model, x: pd.DataFrame) -> np.ndarray:
    v = np.asarray(model.predict(x), dtype=float)
    assert np.isfinite(v).all()
    return v


def _mean_one(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    assert (weights > 0).all()
    return weights / weights.mean()


def _crossfit_splits(w: np.ndarray, y: np.ndarray, seed: int):
    skf = StratifiedKFold(CROSSFIT_FOLDS, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(w)), 2 * w + y))


# ----------------------------------------------------------------------------- learner
@dataclass
class UpliftModel:
    learner: str
    spec: EngineSpec
    e: float = float("nan")
    columns: list = field(default_factory=list)
    models: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, w, y) -> "UpliftModel":
        w = np.asarray(w).astype(int)
        y = np.asarray(y).astype(int)
        assert len(X) == len(w) == len(y)
        assert set(np.unique(w)) == {0, 1} and set(np.unique(y)) <= {0, 1}
        assert TREATMENT_COL not in X.columns
        X = X.reset_index(drop=True)
        self.columns = list(X.columns)
        self.e = float(w.mean())
        e, sp, L = self.e, self.spec, self.learner
        t, c = w == 1, w == 0
        if L == "s":
            self.models["m"] = sp.classifier().fit(X.assign(**{TREATMENT_COL: w.astype(float)}), y)
        elif L == "t":
            self.models["m0"] = sp.classifier().fit(X[c], y[c])
            self.models["m1"] = sp.classifier().fit(X[t], y[t])
        elif L == "x":
            m0 = sp.classifier().fit(X[c], y[c])
            m1 = sp.classifier().fit(X[t], y[t])
            d1 = y[t] - _prob(m0, X[t])        # treated: observed - imputed control outcome
            d0 = _prob(m1, X[c]) - y[c]        # control: imputed treated outcome - observed
            self.models["tau1"] = sp.regressor().fit(X[t], d1)
            self.models["tau0"] = sp.regressor().fit(X[c], d0)
            self.info["stage1_models"] = 2     # m0/m1 not needed for prediction
        elif L == "r":
            m_hat = np.full(len(y), np.nan)
            for tr, va in _crossfit_splits(w, y, sp.seed):
                m_hat[va] = _prob(sp.classifier().fit(X.iloc[tr], y[tr]), X.iloc[va])
            assert np.isfinite(m_hat).all()
            den = w - e
            pseudo = (y - m_hat) / den
            self.models["tau"] = sp.regressor().fit(X, pseudo, sample_weight=_mean_one(den ** 2))
        elif L == "dr":
            m0_hat = np.full(len(y), np.nan)
            m1_hat = np.full(len(y), np.nan)
            for tr, va in _crossfit_splits(w, y, sp.seed):
                tr0, tr1 = tr[c[tr]], tr[t[tr]]
                m0_hat[va] = _prob(sp.classifier().fit(X.iloc[tr0], y[tr0]), X.iloc[va])
                m1_hat[va] = _prob(sp.classifier().fit(X.iloc[tr1], y[tr1]), X.iloc[va])
            assert np.isfinite(m0_hat).all() and np.isfinite(m1_hat).all()
            pseudo = m1_hat - m0_hat + w / e * (y - m1_hat) - (1 - w) / (1 - e) * (y - m0_hat)
            self.models["tau"] = sp.regressor().fit(X, pseudo)
        elif L == "four_class":
            z = 2 * w + y
            assert set(np.unique(z)) == {0, 1, 2, 3}
            m = sp.classifier(num_class=4).fit(X, z)
            assert list(m.classes_) == [0, 1, 2, 3]
            self.models["m"] = m
        elif L == "lai":
            z = np.where(w == 1, y, 1 - y)     # Z = YW + (1-Y)(1-W)
            self.models["m"] = sp.classifier().fit(X, z, sample_weight=None)
        else:
            raise ValueError(L)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        assert list(X.columns) == self.columns, "column mismatch"
        X = X.reset_index(drop=True)
        e, L, M = self.e, self.learner, self.models
        if L == "s":
            tau = (_prob(M["m"], X.assign(**{TREATMENT_COL: 1.0}))
                   - _prob(M["m"], X.assign(**{TREATMENT_COL: 0.0})))
        elif L == "t":
            tau = _prob(M["m1"], X) - _prob(M["m0"], X)
        elif L == "x":
            # Künzel: tau = g(x) tau0(x) + (1 - g(x)) tau1(x), g = propensity -> constant e
            tau = e * _pred(M["tau0"], X) + (1 - e) * _pred(M["tau1"], X)
        elif L in ("r", "dr"):
            tau = _pred(M["tau"], X)
        elif L == "four_class":
            p = M["m"].predict_proba(X)
            assert p.shape[1] == 4 and np.isfinite(p).all()
            # P(Y=1|T,x) = P(z=3|x)/e, P(Y=1|C,x) = P(z=1|x)/(1-e) with constant e
            tau = p[:, 3] / e - p[:, 1] / (1 - e)
        elif L == "lai":
            tau = 2 * _prob(M["m"], X) - 1
        else:
            raise ValueError(L)
        tau = np.asarray(tau, dtype=float)
        assert tau.shape == (len(X),) and np.isfinite(tau).all()
        return tau

    def n_submodels(self) -> int:
        k = CROSSFIT_FOLDS
        return {"s": 1, "t": 2, "x": 4, "r": k + 1, "dr": 2 * k + 1,
                "four_class": 1, "lai": 1}[self.learner]


def fit(learner: str, X: pd.DataFrame, w, y, params: dict, engine: str, seed: int,
        n_jobs: int = 1, lgb_min_data_in_leaf: int | None = LGB_MIN_DATA_IN_LEAF) -> UpliftModel:
    assert learner in LEARNERS and engine in ENGINES
    spec = EngineSpec(engine, check_params(params), int(seed), int(n_jobs), lgb_min_data_in_leaf)
    return UpliftModel(learner, spec).fit(X, w, y)
