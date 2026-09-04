import pytest

from marketjepa.config import DataConfig, MarketJEPAConfig, ModelConfig, TrainConfig


@pytest.fixture
def small_cfg() -> MarketJEPAConfig:
    """Configuration minuscule pour des tests rapides sur CPU."""
    return MarketJEPAConfig(
        data=DataConfig(context_len=16, target_len=4, num_features=5, stride=2),
        model=ModelConfig(
            patch_len=2,
            d_model=16,
            n_heads=2,
            n_layers=1,
            dim_feedforward=32,
            predictor_dim=8,
            predictor_depth=1,
            dropout=0.0,
        ),
        train=TrainConfig(
            batch_size=8,
            epochs=1,
            lr=1e-3,
            warmup_steps=2,
            log_every=1000,
            seed=0,
            device="cpu",
        ),
    )
