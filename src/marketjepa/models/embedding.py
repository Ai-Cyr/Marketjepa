"""Tokenisation des séries temporelles en patchs + encodage positionnel."""

from __future__ import annotations

import math

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """(B, T, F) -> (B, T // patch_len, d_model) via projection linéaire des patchs."""

    def __init__(self, num_features: int, patch_len: int, d_model: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.patch_len = patch_len
        self.proj = nn.Linear(num_features * patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        if f != self.num_features:
            raise ValueError(f"{self.num_features} features attendues, reçu {f}")
        if t % self.patch_len != 0:
            raise ValueError(f"T={t} doit être un multiple de patch_len={self.patch_len}")
        n = t // self.patch_len
        x = x.reshape(b, n, self.patch_len * f)
        return self.proj(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Encodage positionnel sinusoïdal fixe, indexable par positions arbitraires."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model doit être pair pour l'encodage sinusoïdal")
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """positions : (N,) ou (B, N) d'indices entiers -> (N, D) ou (B, N, D)."""
        return self.pe[positions]


def gather_tokens(tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Sélectionne des tokens par indice.

    tokens : (B, N, D) ; idx : (K,) partagé par le batch ou (B, K) par échantillon.
    """
    if idx.dim() == 1:
        return tokens.index_select(1, idx)
    if idx.dim() == 2:
        expanded = idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        return tokens.gather(1, expanded)
    raise ValueError("idx doit être de dimension 1 ou 2")
