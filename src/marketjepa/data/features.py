"""Transformation OHLCV -> features stationnaires."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

FEATURE_NAMES = (
    "log_return",  # log(C_t / C_{t-1})
    "log_range",  # log(H_t / L_t)
    "log_body",  # log(C_t / O_t)
    "log_gap",  # log(O_t / C_{t-1})
    "log_volume_change",  # log(V_t / V_{t-1})
)

_EPS = 1e-8


def _check_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    missing = [c for c in OHLCV_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"Colonnes OHLCV manquantes: {missing}")
    return df.rename(columns={cols[c]: c for c in OHLCV_COLUMNS})


def ohlcv_to_features(df: pd.DataFrame) -> np.ndarray:
    """Convertit un DataFrame OHLCV (T lignes) en matrice de features (T-1, 5).

    Les features sont des log-ratios, donc approximativement stationnaires et
    indépendantes du niveau de prix. La première ligne est perdue (différence).
    """
    df = _check_columns(df)
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    lo = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    v = df["volume"].to_numpy(dtype=np.float64)

    if len(c) < 2:
        raise ValueError("Au moins 2 lignes sont nécessaires pour calculer des rendements")
    if np.any(c <= 0) or np.any(o <= 0) or np.any(h <= 0) or np.any(lo <= 0):
        raise ValueError("Les prix doivent être strictement positifs")

    prev_c = c[:-1]
    prev_v = v[:-1]
    feats = np.stack(
        [
            np.log(c[1:] / prev_c),
            np.log(h[1:] / lo[1:]),
            np.log(c[1:] / o[1:]),
            np.log(o[1:] / prev_c),
            np.log((v[1:] + _EPS) / (prev_v + _EPS)),
        ],
        axis=1,
    )
    return feats.astype(np.float32)


@dataclass(frozen=True)
class ZScoreStats:
    mean: np.ndarray
    std: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, list[float]]) -> ZScoreStats:
        return cls(np.asarray(d["mean"], dtype=np.float32), np.asarray(d["std"], dtype=np.float32))


def fit_zscore(x: np.ndarray) -> ZScoreStats:
    """Calcule moyenne/écart-type par feature. À ajuster sur le jeu d'entraînement seulement."""
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.where(std < _EPS, 1.0, std).astype(np.float32)
    return ZScoreStats(mean=mean, std=std)


def apply_zscore(x: np.ndarray, stats: ZScoreStats) -> np.ndarray:
    return ((x - stats.mean) / stats.std).astype(np.float32)
