"""MarketJEPA : Joint-Embedding Predictive Architecture pour séries temporelles de marché."""

from marketjepa.config import DataConfig, MarketJEPAConfig, ModelConfig, TrainConfig, load_config
from marketjepa.models.jepa import MarketJEPA

__all__ = [
    "DataConfig",
    "MarketJEPA",
    "MarketJEPAConfig",
    "ModelConfig",
    "TrainConfig",
    "load_config",
]

__version__ = "0.1.0"
