import pytest

from marketjepa.config import DataConfig, MarketJEPAConfig, ModelConfig, load_config


def test_default_config_is_valid():
    cfg = MarketJEPAConfig()
    assert cfg.num_patches == 20
    assert cfg.num_context_patches == 16
    assert cfg.num_target_patches == 4


def test_patch_len_must_divide_lengths():
    with pytest.raises(ValueError):
        MarketJEPAConfig(
            data=DataConfig(context_len=10, target_len=4), model=ModelConfig(patch_len=4)
        )


def test_yaml_roundtrip(tmp_path):
    cfg = MarketJEPAConfig(data=DataConfig(context_len=32, target_len=8))
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    loaded = load_config(path)
    assert loaded.to_dict() == cfg.to_dict()


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="Clés inconnues"):
        MarketJEPAConfig.from_dict({"data": {"foo": 1}})


def test_default_yaml_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.data.num_features == 5
