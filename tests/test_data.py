import numpy as np
import pytest
import torch

from marketjepa.config import DataConfig
from marketjepa.data import (
    FEATURE_NAMES,
    WindowDataset,
    apply_zscore,
    build_dataset,
    fit_zscore,
    generate_synthetic_ohlcv,
    ohlcv_to_features,
)


def test_synthetic_ohlcv_is_consistent():
    df = generate_synthetic_ohlcv(200, seed=1)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 200
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["volume"] > 0).all()
    # Chaque bougie s'ouvre au close précédent
    np.testing.assert_allclose(df["open"].to_numpy()[1:], df["close"].to_numpy()[:-1])


def test_synthetic_is_reproducible():
    a = generate_synthetic_ohlcv(50, seed=3)
    b = generate_synthetic_ohlcv(50, seed=3)
    assert a.equals(b)


def test_features_shape_and_values():
    df = generate_synthetic_ohlcv(100, seed=0)
    f = ohlcv_to_features(df)
    assert f.shape == (99, len(FEATURE_NAMES))
    assert f.dtype == np.float32
    assert np.isfinite(f).all()
    # log_range >= 0 par construction
    assert (f[:, 1] >= 0).all()
    # log_gap nul puisque open == close précédent
    np.testing.assert_allclose(f[:, 3], 0.0, atol=1e-6)


def test_features_rejects_missing_columns():
    df = generate_synthetic_ohlcv(10).drop(columns=["volume"])
    with pytest.raises(ValueError, match="manquantes"):
        ohlcv_to_features(df)


def test_zscore():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=3.0, scale=2.0, size=(1000, 4)).astype(np.float32)
    stats = fit_zscore(x)
    z = apply_zscore(x, stats)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-4)


def test_window_dataset_single_series():
    series = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
    ds = WindowDataset(series, window_len=5, stride=3)
    assert len(ds) == 6
    w = ds[1]
    assert isinstance(w, torch.Tensor)
    assert w.shape == (5, 3)
    np.testing.assert_array_equal(w.numpy(), series[3:8])


def test_window_dataset_multi_series_never_crosses_boundary():
    a = np.zeros((10, 2), dtype=np.float32)
    b = np.ones((7, 2), dtype=np.float32)
    ds = WindowDataset([a, b], window_len=4, stride=1)
    assert len(ds) == 7 + 4
    for i in range(len(ds)):
        w = ds[i].numpy()
        assert (w == w[0, 0]).all()


def test_window_dataset_errors():
    with pytest.raises(ValueError):
        WindowDataset(np.zeros((3, 2), dtype=np.float32), window_len=5)
    with pytest.raises(ValueError):
        WindowDataset([np.zeros((10, 2)), np.zeros((10, 3))], window_len=4)


def test_build_dataset_pipeline():
    cfg = DataConfig(context_len=16, target_len=4, num_features=5, stride=1)
    frames = [generate_synthetic_ohlcv(60, seed=i) for i in range(2)]
    ds, stats = build_dataset(frames, cfg)
    assert stats is not None and stats.mean.shape == (5,)
    assert len(ds) == 2 * (59 - 20 + 1)
    assert ds[0].shape == (20, 5)

    # Réutilisation des statistiques (jeu de validation)
    ds_val, stats_val = build_dataset(generate_synthetic_ohlcv(40, seed=9), cfg, stats=stats)
    assert stats_val is stats
