import torch

from elenkhos_serve.config import QwenConfig
from elenkhos_serve.model.layers import RotaryEmbedding
from elenkhos_serve.model.qwen import QwenDecoderLayer


def make_tiny_qwen3_config() -> QwenConfig:
    return QwenConfig(
        vocab_size=100,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=16,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        use_qk_norm=True,
        attention_bias=False,
    )


def make_position_embeddings(
    config: QwenConfig,
    batch_size: int,
    sequence_length: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    rope = RotaryEmbedding(
        head_dim=config.head_dim,
        theta=config.rope_theta,
    )

    position_ids = torch.arange(
        sequence_length,
        dtype=torch.long,
    ).unsqueeze(0).expand(batch_size, -1)

    return rope(position_ids, dtype=dtype)


def test_decoder_layer_keeps_shape_and_dtype() -> None:
    torch.manual_seed(0)

    config = make_tiny_qwen3_config()
    layer = QwenDecoderLayer(config, layer_idx=0)

    x = torch.randn(
        2,
        3,
        config.hidden_size,
        dtype=torch.float32,
    )

    cos, sin = make_position_embeddings(
        config=config,
        batch_size=2,
        sequence_length=3,
        dtype=x.dtype,
    )

    y = layer(
        hidden_states=x,
        position_embeddings=(cos, sin),
    )

    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_decoder_layer_has_qwen_style_state_dict_names() -> None:
    config = make_tiny_qwen3_config()
    layer = QwenDecoderLayer(config, layer_idx=0)

    keys = set(layer.state_dict().keys())

    assert "input_layernorm.weight" in keys
    assert "self_attn.q_proj.weight" in keys
    assert "self_attn.k_proj.weight" in keys
    assert "self_attn.v_proj.weight" in keys
    assert "self_attn.o_proj.weight" in keys
    assert "self_attn.q_norm.weight" in keys
    assert "self_attn.k_norm.weight" in keys
    assert "post_attention_layernorm.weight" in keys
    assert "mlp.gate_proj.weight" in keys
    assert "mlp.up_proj.weight" in keys
    assert "mlp.down_proj.weight" in keys


def test_decoder_layer_future_token_does_not_change_earlier_outputs() -> None:
    torch.manual_seed(0)

    config = make_tiny_qwen3_config()
    layer = QwenDecoderLayer(config, layer_idx=0).eval()

    x = torch.randn(1, 4, config.hidden_size)

    cos, sin = make_position_embeddings(
        config=config,
        batch_size=1,
        sequence_length=4,
        dtype=x.dtype,
    )

    output_before = layer(
        hidden_states=x,
        position_embeddings=(cos, sin),
    )

    changed_x = x.clone()
    changed_x[:, -1, :] += 100.0

    output_after = layer(
        hidden_states=changed_x,
        position_embeddings=(cos, sin),
    )

    torch.testing.assert_close(
        output_before[:, :-1, :],
        output_after[:, :-1, :],
        rtol=1e-5,
        atol=1e-6,
    )
