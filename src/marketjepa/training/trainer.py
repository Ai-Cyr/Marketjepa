"""Boucle d'entraînement JEPA : optimisation du contexte/prédicteur + EMA de la cible."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from marketjepa.config import MarketJEPAConfig
from marketjepa.models.jepa import MarketJEPA
from marketjepa.training.masking import MaskSpec

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: MarketJEPA,
        masker: Callable[[], MaskSpec],
        cfg: MarketJEPAConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.masker = masker
        self.cfg = cfg
        self.device = device

        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        self.step = 0
        self.total_steps = 0
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------ #
    def _lr_at(self, step: int) -> float:
        warmup = self.cfg.train.warmup_steps
        base = self.cfg.train.lr
        if step < warmup:
            return base * (step + 1) / max(warmup, 1)
        if self.total_steps <= warmup:
            return base
        progress = (step - warmup) / max(self.total_steps - warmup, 1)
        return base * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    def _momentum_at(self, step: int) -> float:
        m0, m1 = self.cfg.train.ema_momentum_start, self.cfg.train.ema_momentum_end
        if self.total_steps <= 1:
            return m0
        frac = min(step / (self.total_steps - 1), 1.0)
        return m0 + (m1 - m0) * frac

    # ------------------------------------------------------------------ #
    def train_step(self, batch: torch.Tensor) -> float:
        self.model.train()
        x = batch.to(self.device, non_blocking=True)
        mask = self.masker().to(self.device)

        for group in self.optimizer.param_groups:
            group["lr"] = self._lr_at(self.step)

        pred, target = self.model(x, mask.context_idx, mask.target_idx)
        loss = self.model.loss(pred, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.model.update_target_encoder(self._momentum_at(self.step))

        self.step += 1
        return float(loss.item())

    def fit(self, loader: DataLoader, epochs: int | None = None) -> list[dict[str, float]]:
        epochs = epochs or self.cfg.train.epochs
        self.total_steps = epochs * len(loader)
        log_every = max(self.cfg.train.log_every, 1)

        for epoch in range(epochs):
            t0 = time.time()
            running, count = 0.0, 0
            for batch in loader:
                loss = self.train_step(batch)
                running += loss
                count += 1
                if self.step % log_every == 0:
                    logger.info(
                        "epoch %d step %d loss %.5f lr %.2e m %.4f",
                        epoch,
                        self.step,
                        loss,
                        self._lr_at(self.step - 1),
                        self._momentum_at(self.step - 1),
                    )
            record = {
                "epoch": float(epoch),
                "loss": running / max(count, 1),
                "seconds": time.time() - t0,
            }
            self.history.append(record)
            logger.info("epoch %d terminée: loss moyenne %.5f", epoch, record["loss"])
        return self.history

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "config": self.cfg.to_dict(),
            "history": self.history,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.step = int(payload.get("step", 0))
        self.history = list(payload.get("history", []))
        return payload
