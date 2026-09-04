import pytest
import torch

from marketjepa.training.masking import BlockMasker, FutureMasker, build_masker


def test_future_masker_is_causal():
    m = FutureMasker(num_patches=10, num_target_patches=3)
    spec = m()
    assert spec.context_idx.tolist() == list(range(7))
    assert spec.target_idx.tolist() == [7, 8, 9]
    assert spec.context_idx.max() < spec.target_idx.min()


def test_future_masker_validates():
    with pytest.raises(ValueError):
        FutureMasker(num_patches=4, num_target_patches=4)


def test_block_masker_partition():
    g = torch.Generator().manual_seed(0)
    m = BlockMasker(num_patches=12, num_blocks=2, block_size_range=(2, 3), min_context=3)
    for _ in range(20):
        spec = m(g)
        union = torch.cat([spec.context_idx, spec.target_idx]).sort().values
        assert union.tolist() == list(range(12))
        assert len(spec.context_idx) >= 3
        assert len(spec.target_idx) >= 2


def test_build_masker():
    assert isinstance(build_masker("future", 10, 2), FutureMasker)
    assert isinstance(build_masker("block", 10, 4, num_blocks=2), BlockMasker)
    with pytest.raises(ValueError):
        build_masker("random", 10, 2)
