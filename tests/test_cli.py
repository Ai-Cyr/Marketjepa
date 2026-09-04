import torch

from marketjepa.cli import main


def test_cli_synthetic_end_to_end(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
data: {context_len: 16, target_len: 4, num_features: 5, stride: 4}
model: {patch_len: 2, d_model: 16, n_heads: 2, n_layers: 1, dim_feedforward: 32,
        predictor_dim: 8, predictor_depth: 1, dropout: 0.0}
train: {batch_size: 8, epochs: 1, warmup_steps: 1, log_every: 1000, device: cpu}
""",
        encoding="utf-8",
    )
    out = main(
        [
            "--config",
            str(cfg_path),
            "--synthetic-steps",
            "200",
            "--out",
            str(tmp_path / "model.pt"),
            "--quiet",
        ]
    )
    payload = torch.load(out, weights_only=False)
    assert "model" in payload and payload["zscore"] is not None
