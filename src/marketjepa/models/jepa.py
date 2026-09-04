"""MarketJEPA : encodeur de contexte, encodeur cible (EMA) et prédicteur latent."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from marketjepa.config import ModelConfig
from marketjepa.models.embedding import gather_tokens
from marketjepa.models.encoder import TimeSeriesEncoder
from marketjepa.models.predictor import Predictor


class MarketJEPA(nn.Module):
    """Prédit, dans l'espace latent, les représentations des patchs cibles à partir du contexte.

    - `context_encoder` : entraîné par rétropropagation, ne voit que les patchs de contexte.
    - `target_encoder` : copie mise à jour par moyenne mobile exponentielle (EMA), voit
      la fenêtre entière et fournit les cibles (sans gradient).
    - `predictor` : réseau léger qui prédit les cibles depuis le contexte + positions.
    """

    def __init__(self, num_features: int, cfg: ModelConfig, max_patches: int = 4096) -> None:
        super().__init__()
        self.cfg = cfg
        self.context_encoder = TimeSeriesEncoder(
            num_features=num_features,
            patch_len=cfg.patch_len,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            max_patches=max_patches,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(
            d_model=cfg.d_model,
            predictor_dim=cfg.predictor_dim,
            n_heads=cfg.n_heads,
            depth=cfg.predictor_depth,
            dropout=cfg.dropout,
            max_patches=max_patches,
        )

    # ------------------------------------------------------------------ #
    def forward(
        self, x: torch.Tensor, ctx_idx: torch.Tensor, tgt_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Renvoie (prédictions, cibles), chacune de forme (B, Nt, D)."""
        with torch.no_grad():
            h = self.target_encoder(x)
            h = F.layer_norm(h, (h.shape[-1],))
            target = gather_tokens(h, tgt_idx)
        ctx = self.context_encoder(x, ctx_idx)
        pred = self.predictor(ctx, ctx_idx, tgt_idx)
        return pred, target

    @staticmethod
    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """EMA : θ_target ← m·θ_target + (1−m)·θ_context."""
        for p_t, p_c in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters(), strict=True
        ):
            p_t.mul_(momentum).add_(p_c.detach(), alpha=1.0 - momentum)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Représentations de patchs (B, N, D) via l'encodeur cible (usage aval)."""
        return self.target_encoder(x)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Représentation globale (B, D) d'une fenêtre : moyenne des patchs."""
        return self.encode(x).mean(dim=1)
