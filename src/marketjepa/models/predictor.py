"""Prédicteur latent : contexte encodé + tokens masqués -> représentations cibles."""

from __future__ import annotations

import torch
from torch import nn

from marketjepa.models.embedding import SinusoidalPositionalEncoding


class Predictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        predictor_dim: int,
        n_heads: int,
        depth: int,
        dropout: float = 0.1,
        max_patches: int = 4096,
    ) -> None:
        super().__init__()
        self.proj_in = nn.Linear(d_model, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos = SinusoidalPositionalEncoding(predictor_dim, max_len=max_patches)
        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=n_heads,
            dim_feedforward=predictor_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(predictor_dim)
        self.proj_out = nn.Linear(predictor_dim, d_model)

    def forward(
        self, ctx: torch.Tensor, ctx_idx: torch.Tensor, tgt_idx: torch.Tensor
    ) -> torch.Tensor:
        """ctx : (B, Nc, D) ; ctx_idx : (Nc,) ou (B, Nc) ; tgt_idx : (Nt,) ou (B, Nt)."""
        b = ctx.shape[0]
        x = self.proj_in(ctx) + self.pos(ctx_idx)
        nt = tgt_idx.shape[-1]
        mask = self.mask_token.expand(b, nt, -1) + self.pos(tgt_idx)
        z = torch.cat([x, mask], dim=1)
        z = self.blocks(z)
        z = self.norm(z[:, -nt:])
        return self.proj_out(z)
