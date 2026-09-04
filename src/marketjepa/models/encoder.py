"""Encodeur Transformer pour séquences de patchs temporels."""

from __future__ import annotations

import torch
from torch import nn

from marketjepa.models.embedding import PatchEmbedding, SinusoidalPositionalEncoding, gather_tokens


class TimeSeriesEncoder(nn.Module):
    """Encode une fenêtre (B, T, F) en représentations de patchs (B, N, D).

    Si `idx` est fourni, seuls les patchs sélectionnés sont encodés (l'encodeur
    de contexte ne voit jamais les cibles, comme dans I-JEPA).
    """

    def __init__(
        self,
        num_features: int,
        patch_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        max_patches: int = 4096,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_embed = PatchEmbedding(num_features, patch_len, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_patches)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.patch_embed(x)
        n = tokens.shape[1]
        tokens = tokens + self.pos(torch.arange(n, device=tokens.device))
        if idx is not None:
            tokens = gather_tokens(tokens, idx)
        h = self.blocks(tokens)
        return self.norm(h)
