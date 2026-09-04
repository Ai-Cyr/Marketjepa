import torch

from marketjepa.models import MarketJEPA, PatchEmbedding, SinusoidalPositionalEncoding
from marketjepa.models.embedding import gather_tokens
from marketjepa.training.masking import FutureMasker


def test_patch_embedding_shape():
    pe = PatchEmbedding(num_features=5, patch_len=4, d_model=16)
    out = pe(torch.randn(3, 20, 5))
    assert out.shape == (3, 5, 16)


def test_positional_encoding_indexing():
    pos = SinusoidalPositionalEncoding(d_model=8, max_len=32)
    assert pos(torch.arange(5)).shape == (5, 8)
    assert pos(torch.zeros(2, 5, dtype=torch.long)).shape == (2, 5, 8)


def test_gather_tokens_shared_and_per_sample():
    tokens = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    shared = gather_tokens(tokens, torch.tensor([1, 3]))
    assert torch.equal(shared, tokens[:, [1, 3]])
    per = gather_tokens(tokens, torch.tensor([[0, 1], [2, 3]]))
    assert torch.equal(per[0], tokens[0, [0, 1]])
    assert torch.equal(per[1], tokens[1, [2, 3]])


def test_jepa_forward_shapes(small_cfg):
    cfg = small_cfg
    model = MarketJEPA(num_features=cfg.data.num_features, cfg=cfg.model)
    x = torch.randn(4, cfg.data.window_len, cfg.data.num_features)
    spec = FutureMasker(cfg.num_patches, cfg.num_target_patches)()
    pred, target = model(x, spec.context_idx, spec.target_idx)
    assert pred.shape == (4, cfg.num_target_patches, cfg.model.d_model)
    assert target.shape == pred.shape
    assert not target.requires_grad
    loss = model.loss(pred, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.embed(x).shape == (4, cfg.model.d_model)


def test_target_encoder_has_no_grad_and_ema_moves(small_cfg):
    cfg = small_cfg
    model = MarketJEPA(num_features=cfg.data.num_features, cfg=cfg.model)
    assert all(not p.requires_grad for p in model.target_encoder.parameters())

    with torch.no_grad():
        for p in model.context_encoder.parameters():
            p.add_(1.0)
    before = [p.clone() for p in model.target_encoder.parameters()]
    model.update_target_encoder(momentum=0.5)
    after = list(model.target_encoder.parameters())
    for b, a, c in zip(before, after, model.context_encoder.parameters(), strict=True):
        torch.testing.assert_close(a, 0.5 * b + 0.5 * c)
