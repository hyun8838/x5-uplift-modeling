"""Synthetic uplift data with known individual effect tau(x) (unit tests only).

Y ~ Bernoulli(sigmoid(b(x) + W * g(x))), W ~ Bernoulli(p_treat), true tau(x) = sigmoid(b+g) - sigmoid(b).
Feature roles (by name prefix):
  prog_*  : affect only the baseline b(x) (pure prognostic, no effect modification)
  mod_*   : affect only g(x) (effect modifiers)
  gender  : F/M/U string, effect modifier (U shifts g)
  age     : raw integer-like with NaN and outliers, effect modifier (age > 50)
  noise_* : unrelated
scenario 'mixed'    : g has both signs (about 40% of customers have tau < 0)
scenario 'positive' : g >= 0 for everyone, heterogeneous size
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make(n: int, seed: int, scenario: str = "mixed", n_noise: int = 4, p_treat: float = 0.5):
    rng = np.random.default_rng(seed)
    d = {}
    for i in range(4):
        d[f"prog_{i}"] = rng.normal(size=n)
    d["mod_0"] = rng.normal(size=n)
    d["mod_1"] = rng.uniform(size=n)
    d["gender"] = rng.choice(np.array(["F", "M", "U"], dtype=object), size=n, p=[0.37, 0.17, 0.46])
    age = np.round(rng.normal(45, 12, size=n))
    out = rng.uniform(size=n) < 0.005
    age[out] = rng.choice([-7491.0, 1852.0, 0.0], size=out.sum())
    age[rng.uniform(size=n) < 0.10] = np.nan
    d["age"] = age
    for i in range(n_noise):
        d[f"noise_{i}"] = rng.normal(size=n) if i % 3 else rng.poisson(3, size=n).astype(float)
    x = pd.DataFrame(d)
    # 10% missing in one prognostic and one modifier column (engine default missing handling)
    x.loc[rng.uniform(size=n) < 0.10, "prog_3"] = np.nan
    x.loc[rng.uniform(size=n) < 0.05, "mod_1"] = np.nan

    p3 = x.prog_3.fillna(0).to_numpy()
    b = 0.3 + 1.0 * x.prog_0 + 0.7 * x.prog_1 - 0.6 * x.prog_2 + 0.5 * p3
    m1 = x.mod_1.fillna(0.5).to_numpy()
    old = (np.nan_to_num(age, nan=0.0) > 50) & (age < 120)
    u = (x.gender == "U").to_numpy()
    if scenario == "mixed":
        g = 0.6 * x.mod_0.to_numpy() + 0.5 * (m1 - 0.5) + 0.3 * u + 0.25 * old - 0.2
    elif scenario == "positive":
        g = 0.15 + 0.6 * _sigmoid(2.0 * x.mod_0.to_numpy()) + 0.4 * m1 + 0.3 * u + 0.2 * old
    else:
        raise ValueError(scenario)
    b = np.asarray(b, float)
    w = (rng.uniform(size=n) < p_treat).astype(int)
    tau = _sigmoid(b + g) - _sigmoid(b)
    p = _sigmoid(b + w * g)
    y = (rng.uniform(size=n) < p).astype(int)
    return x, w, y, tau
