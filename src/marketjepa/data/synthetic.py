"""Générateur de données OHLCV synthétiques (GBM à changement de régime).

Sert aux tests, aux démonstrations et à valider la boucle d'entraînement sans
données réelles. Les régimes (calme / volatil) suivent une chaîne de Markov.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(
    n_steps: int,
    seed: int | None = 0,
    start_price: float = 100.0,
    substeps: int = 8,
    freq: str = "h",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Régimes : (drift annuel, vol annuelle) par pas, convertis en paramètres par sous-pas.
    regimes = np.array([[0.05, 0.10], [-0.02, 0.35]])
    switch = np.array([[0.98, 0.02], [0.05, 0.95]])
    steps_per_year = 24 * 365 if freq == "h" else 252
    dt = 1.0 / (steps_per_year * substeps)

    regime = 0
    price = start_price
    opens = np.empty(n_steps)
    highs = np.empty(n_steps)
    lows = np.empty(n_steps)
    closes = np.empty(n_steps)
    volumes = np.empty(n_steps)

    for t in range(n_steps):
        regime = rng.choice(2, p=switch[regime])
        mu, sigma = regimes[regime]
        z = rng.standard_normal(substeps)
        increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        path = price * np.exp(np.cumsum(increments))
        opens[t] = price
        highs[t] = max(price, path.max())
        lows[t] = min(price, path.min())
        closes[t] = path[-1]
        price = closes[t]
        abs_ret = abs(np.log(closes[t] / opens[t]))
        volumes[t] = np.exp(10.0 + 20.0 * abs_ret + 0.3 * rng.standard_normal())

    index = pd.date_range("2020-01-01", periods=n_steps, freq=freq)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=index,
    )
