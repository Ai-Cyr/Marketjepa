"""Stratégies de masquage : quels patchs forment le contexte, lesquels sont les cibles."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaskSpec:
    context_idx: torch.Tensor  # (Nc,) indices de patchs visibles
    target_idx: torch.Tensor  # (Nt,) indices de patchs à prédire

    def to(self, device: torch.device | str) -> MaskSpec:
        return MaskSpec(self.context_idx.to(device), self.target_idx.to(device))


class FutureMasker:
    """Contexte = passé, cible = futur. Masque déterministe adapté à la prévision."""

    def __init__(self, num_patches: int, num_target_patches: int) -> None:
        if not 0 < num_target_patches < num_patches:
            raise ValueError("0 < num_target_patches < num_patches requis")
        self.num_patches = num_patches
        self.num_target_patches = num_target_patches

    def __call__(self, generator: torch.Generator | None = None) -> MaskSpec:
        split = self.num_patches - self.num_target_patches
        return MaskSpec(
            context_idx=torch.arange(0, split),
            target_idx=torch.arange(split, self.num_patches),
        )


class BlockMasker:
    """Masque I-JEPA : blocs contigus aléatoires en cible, le reste en contexte.

    Les cibles peuvent se trouver n'importe où dans la fenêtre (passé compris),
    ce qui force l'encodeur à modéliser la structure complète de la série.
    """

    def __init__(
        self,
        num_patches: int,
        num_blocks: int = 2,
        block_size_range: tuple[int, int] = (2, 4),
        min_context: int = 2,
    ) -> None:
        lo, hi = block_size_range
        if not 1 <= lo <= hi < num_patches:
            raise ValueError("block_size_range invalide")
        if num_blocks < 1:
            raise ValueError("num_blocks doit être >= 1")
        self.num_patches = num_patches
        self.num_blocks = num_blocks
        self.block_size_range = block_size_range
        self.min_context = min_context

    def __call__(self, generator: torch.Generator | None = None) -> MaskSpec:
        lo, hi = self.block_size_range
        n = self.num_patches
        for _ in range(100):
            is_target = torch.zeros(n, dtype=torch.bool)
            for _ in range(self.num_blocks):
                size = int(torch.randint(lo, hi + 1, (1,), generator=generator))
                start = int(torch.randint(0, n - size + 1, (1,), generator=generator))
                is_target[start : start + size] = True
            if (~is_target).sum() >= self.min_context and is_target.any():
                break
        else:
            raise RuntimeError("Impossible de générer un masque respectant min_context")
        return MaskSpec(
            context_idx=torch.nonzero(~is_target, as_tuple=False).squeeze(1),
            target_idx=torch.nonzero(is_target, as_tuple=False).squeeze(1),
        )


def build_masker(
    strategy: str,
    num_patches: int,
    num_target_patches: int,
    num_blocks: int = 2,
) -> FutureMasker | BlockMasker:
    if strategy == "future":
        return FutureMasker(num_patches, num_target_patches)
    if strategy == "block":
        size = max(1, num_target_patches // num_blocks)
        hi = min(num_patches - 1, max(size, 1))
        lo = max(1, min(size, hi))
        return BlockMasker(num_patches, num_blocks=num_blocks, block_size_range=(lo, hi))
    raise ValueError(f"Stratégie de masquage inconnue: {strategy}")
