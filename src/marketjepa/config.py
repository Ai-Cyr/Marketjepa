"""Configuration typée du projet (chargeable depuis YAML)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    context_len: int = 64
    target_len: int = 16
    num_features: int = 5
    stride: int = 1
    normalize: str = "zscore"

    @property
    def window_len(self) -> int:
        return self.context_len + self.target_len


@dataclass
class ModelConfig:
    patch_len: int = 4
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dim_feedforward: int = 256
    predictor_dim: int = 64
    predictor_depth: int = 2
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_steps: int = 100
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 1.0
    masking: str = "future"
    num_target_blocks: int = 2
    seed: int = 42
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    log_every: int = 50
    num_workers: int = 0


@dataclass
class MarketJEPAConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        p = self.model.patch_len
        if p <= 0:
            raise ValueError("model.patch_len doit être > 0")
        if self.data.context_len % p != 0 or self.data.target_len % p != 0:
            raise ValueError(
                "data.context_len et data.target_len doivent être des multiples de "
                f"model.patch_len (patch_len={p}, context_len={self.data.context_len}, "
                f"target_len={self.data.target_len})"
            )
        if self.model.d_model % self.model.n_heads != 0:
            raise ValueError("model.d_model doit être divisible par model.n_heads")
        if self.train.masking not in {"future", "block"}:
            raise ValueError("train.masking doit valoir 'future' ou 'block'")
        if self.data.normalize not in {"zscore", "none"}:
            raise ValueError("data.normalize doit valoir 'zscore' ou 'none'")

    @property
    def num_patches(self) -> int:
        return self.data.window_len // self.model.patch_len

    @property
    def num_context_patches(self) -> int:
        return self.data.context_len // self.model.patch_len

    @property
    def num_target_patches(self) -> int:
        return self.data.target_len // self.model.patch_len

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MarketJEPAConfig:
        raw = raw or {}
        return cls(
            data=_build(DataConfig, raw.get("data", {})),
            model=_build(ModelConfig, raw.get("model", {})),
            train=_build(TrainConfig, raw.get("train", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> MarketJEPAConfig:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls.from_dict(raw)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, allow_unicode=True)


def _build(cls: type, section: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(section) - known
    if unknown:
        raise ValueError(f"Clés inconnues pour {cls.__name__}: {sorted(unknown)}")
    return cls(**section)


def load_config(path: str | Path | None = None) -> MarketJEPAConfig:
    """Charge la configuration depuis un YAML, ou renvoie la configuration par défaut."""
    if path is None:
        return MarketJEPAConfig()
    return MarketJEPAConfig.from_yaml(path)
