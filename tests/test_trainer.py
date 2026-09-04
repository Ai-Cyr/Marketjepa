import torch
from torch.utils.data import DataLoader

from marketjepa.data import build_dataset, generate_synthetic_ohlcv
from marketjepa.models import MarketJEPA
from marketjepa.training import Trainer, build_masker
from marketjepa.utils import set_seed


def _make_trainer(cfg, masking="future"):
    set_seed(cfg.train.seed)
    ds, _ = build_dataset(generate_synthetic_ohlcv(400, seed=0), cfg.data)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)
    model = MarketJEPA(num_features=cfg.data.num_features, cfg=cfg.model)
    masker = build_masker(masking, cfg.num_patches, cfg.num_target_patches)
    return Trainer(model, masker, cfg, torch.device("cpu")), loader


def test_training_smoke_and_checkpoint(small_cfg, tmp_path):
    trainer, loader = _make_trainer(small_cfg)
    history = trainer.fit(loader, epochs=2)
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in history)
    assert trainer.step == 2 * len(loader)

    path = trainer.save_checkpoint(tmp_path / "ckpt.pt")
    assert path.exists()

    fresh, _ = _make_trainer(small_cfg)
    payload = fresh.load_checkpoint(path)
    assert fresh.step == trainer.step
    assert payload["config"]["data"]["context_len"] == small_cfg.data.context_len
    for a, b in zip(fresh.model.parameters(), trainer.model.parameters(), strict=True):
        torch.testing.assert_close(a, b)


def test_block_masking_trains(small_cfg):
    small_cfg.train.masking = "block"
    trainer, loader = _make_trainer(small_cfg, masking="block")
    history = trainer.fit(loader, epochs=1)
    assert torch.isfinite(torch.tensor(history[0]["loss"]))


def test_lr_and_momentum_schedules(small_cfg):
    trainer, loader = _make_trainer(small_cfg)
    trainer.total_steps = 10
    lrs = [trainer._lr_at(s) for s in range(10)]
    assert lrs[0] < lrs[1]  # warmup
    assert lrs[-1] < lrs[2]  # décroissance cosinus
    ms = [trainer._momentum_at(s) for s in range(10)]
    assert ms[0] == small_cfg.train.ema_momentum_start
    assert abs(ms[-1] - small_cfg.train.ema_momentum_end) < 1e-9
