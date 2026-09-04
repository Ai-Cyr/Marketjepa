"""Point d'entrée `marketjepa-train`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from marketjepa.config import load_config
from marketjepa.data.dataset import build_dataset
from marketjepa.data.synthetic import generate_synthetic_ohlcv
from marketjepa.models.jepa import MarketJEPA
from marketjepa.training.masking import build_masker
from marketjepa.training.trainer import Trainer
from marketjepa.utils import resolve_device, set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pré-entraînement MarketJEPA")
    p.add_argument("--config", type=Path, default=None, help="Fichier YAML de configuration")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--data", type=Path, nargs="+", help="CSV OHLCV (un par actif)")
    src.add_argument(
        "--synthetic-steps", type=int, default=None, help="Générer N pas de données synthétiques"
    )
    p.add_argument("--epochs", type=int, default=None, help="Écrase train.epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Écrase train.batch_size")
    p.add_argument("--out", type=Path, default=None, help="Chemin du checkpoint final")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size

    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    if args.data:
        frames = [pd.read_csv(path, index_col=0, parse_dates=True) for path in args.data]
    else:
        n = args.synthetic_steps or 10_000
        frames = [generate_synthetic_ohlcv(n, seed=cfg.train.seed)]
    dataset, stats = build_dataset(frames, cfg.data)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.train.num_workers,
    )

    model = MarketJEPA(num_features=cfg.data.num_features, cfg=cfg.model)
    masker = build_masker(
        cfg.train.masking,
        num_patches=cfg.num_patches,
        num_target_patches=cfg.num_target_patches,
        num_blocks=cfg.train.num_target_blocks,
    )
    trainer = Trainer(model, masker, cfg, device)
    logging.getLogger(__name__).info(
        "device=%s fenêtres=%d patchs=%d paramètres=%d",
        device,
        len(dataset),
        cfg.num_patches,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    trainer.fit(loader)

    out = args.out or Path(cfg.train.checkpoint_dir) / "marketjepa.pt"
    trainer.save_checkpoint(out, extra={"zscore": stats.to_dict() if stats else None})
    logging.getLogger(__name__).info("checkpoint sauvegardé: %s", out)
    return out


if __name__ == "__main__":
    main()
