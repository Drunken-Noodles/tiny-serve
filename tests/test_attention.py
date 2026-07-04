import torch

from elenkhos_serve.config import QwenConfig
from elenkhos_serve.model.attention import (
    QwenAttention,
    repeat_kv,
)
from elenkhos_serve.model.layers import RotaryEmbedding


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


def test_repeat_kv_repeats_each_kv_head() -> None:
    x = torch.tensor(
        [
            [
                [[10.0]],
                [[20.0]],
            ]
        ]
    )

    repeated = repeat_kv(x, n_rep=2)

    expected = torch.tensor(
        [
            [
                [[10.0]],
                [[10.0]],
                [[20.0]],
                [[20.0]],
            ]
        ]
    )

    torch.testing.assert_close(repeated, expected)


def test_qwen_attention_output_shape_and_parameter_names() -> None:
    config = make_tiny_qwen3_config()
    attention = QwenAttention(config, layer_idx=0)

    x = torch.randn(2, 3, config.hidden_size)

    cos, sin = make_position_embeddings(
        config=config,
        batch_size=2,
        sequence_length=3,
        dtype=x.dtype,
    )

    y = attention(
        hidden_states=x,
        position_embeddings=(cos, sin),
    )

    assert y.shape == x.shape

    assert set(attention.state_dict().keys()) == {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "q_norm.weight",
        "k_norm.weight",
    }


def test_qk_norm_normalizes_each_head_dimension() -> None:
    torch.manual_seed(0)

    config = make_tiny_qwen3_config()
    attention = QwenAttention(config, layer_idx=0)

    x = torch.randn(2, 3, config.hidden_size)

    q, k, _ = attention._project_qkv(x)

    q_rms_squared = q.float().pow(2).mean(dim=-1)
    k_rms_squared = k.float().pow(2).mean(dim=-1)

    torch.testing.assert_close(
        q_rms_squared,
        torch.ones_like(q_rms_squared),
        rtol=1e-4,
        atol=1e-4,
    )

    torch.testing.assert_close(
        k_rms_squared,
        torch.ones_like(k_rms_squared),
        rtol=1e-4,
        atol=1e-4,
    )


def test_attention_cannot_see_future_tokens() -> None:
    torch.manual_seed(0)

    config = make_tiny_qwen3_config()
    attention = QwenAttention(config, layer_idx=0).eval()

    x = torch.randn(1, 4, config.hidden_size)

    cos, sin = make_position_embeddings(
        config=config,
        batch_size=1,
        sequence_length=4,
        dtype=x.dtype,
    )

    output_before = attention(
        hidden_states=x,
        position_embeddings=(cos, sin),
    )

    changed_x = x.clone()

    # Only change the final token.
    changed_x[:, -1, :] = changed_x[:, -1, :] + 100.0

    output_after = attention(
        hidden_states=changed_x,
        position_embeddings=(cos, sin),
    )

    # Positions 0, 1, 2 must not be affected by token 3.
    torch.testing.assert_close(
        output_before[:, :-1, :],
        output_after[:, :-1, :],
        rtol=1e-5,
        atol=1e-6,
    )
