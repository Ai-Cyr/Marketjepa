"""Datasets PyTorch de fenêtres glissantes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from marketjepa.config import DataConfig
from marketjepa.data.features import ZScoreStats, apply_zscore, fit_zscore, ohlcv_to_features


class WindowDataset(Dataset):
    """Découpe une ou plusieurs séries (T_i, F) en fenêtres (window_len, F).

    Les fenêtres ne chevauchent jamais deux séries : chaque actif est traité
    indépendamment, ce qui permet d'entraîner sur un panier d'actifs.
    """

    def __init__(
        self,
        series: np.ndarray | Sequence[np.ndarray],
        window_len: int,
        stride: int = 1,
    ) -> None:
        if isinstance(series, np.ndarray):
            series = [series]
        if window_len <= 0 or stride <= 0:
            raise ValueError("window_len et stride doivent être > 0")

        self.series = [np.ascontiguousarray(s, dtype=np.float32) for s in series]
        self.window_len = window_len
        self.stride = stride

        n_feat = {s.shape[1] for s in self.series}
        if len(n_feat) != 1:
            raise ValueError(f"Toutes les séries doivent avoir le même nombre de features: {n_feat}")
        self.num_features = n_feat.pop()

        index: list[tuple[int, int]] = []
        for a, s in enumerate(self.series):
            n_windows = (len(s) - window_len) // stride + 1
            index.extend((a, i * stride) for i in range(max(n_windows, 0)))
        if not index:
            raise ValueError(
                f"Aucune fenêtre possible: window_len={window_len} > longueur des séries"
            )
        self._index = index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        a, start = self._index[idx]
        window = self.series[a][start : start + self.window_len]
        return torch.from_numpy(window)


def build_dataset(
    frames: pd.DataFrame | Sequence[pd.DataFrame],
    cfg: DataConfig,
    stats: ZScoreStats | None = None,
) -> tuple[WindowDataset, ZScoreStats | None]:
    """Pipeline complet OHLCV -> features -> normalisation -> fenêtres.

    Si `stats` est None et que la normalisation est active, les statistiques sont
    ajustées sur l'ensemble des données fournies (à réserver au jeu d'entraînement).
    """
    if isinstance(frames, pd.DataFrame):
        frames = [frames]
    feats = [ohlcv_to_features(df) for df in frames]

    for f in feats:
        if f.shape[1] != cfg.num_features:
            raise ValueError(
                f"num_features={cfg.num_features} attendu, {f.shape[1]} features calculées"
            )

    if cfg.normalize == "zscore":
        if stats is None:
            stats = fit_zscore(np.concatenate(feats, axis=0))
        feats = [apply_zscore(f, stats) for f in feats]
    else:
        stats = None

    return WindowDataset(feats, window_len=cfg.window_len, stride=cfg.stride), stats
