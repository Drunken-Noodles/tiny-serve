import pytest

from tiny_serve.config import QwenConfig


def make_qwen3_config() -> QwenConfig:
    return QwenConfig(
        vocab_size=151_936,
        hidden_size=1024,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=3072,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_position_embeddings=32_768,
        tie_word_embeddings=True,
        use_qk_norm=True,
        attention_bias=False,
    )


def test_qwen3_derived_attention_sizes() -> None:
    config = make_qwen3_config()

    assert config.num_kv_groups == 2
    assert config.q_projection_size == 2048
    assert config.kv_projection_size == 1024


def test_invalid_gqa_config_fails_early() -> None:
    config = QwenConfig(
        vocab_size=100,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=6,
        head_dim=8,
        intermediate_size=256,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        use_qk_norm=False,
        attention_bias=False,
    )

    with pytest.raises(ValueError, match="divisible"):
        config.validate()
