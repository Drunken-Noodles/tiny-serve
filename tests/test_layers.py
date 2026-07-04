import math

import torch
import torch.nn.functional as F

from tiny_serve.model.layers import (
    RMSNorm,
    RotaryEmbedding,
    SwiGLUMLP,
    apply_rope,
    rotate_half,
)


def test_rmsnorm_keeps_shape_and_dtype() -> None:
    layer = RMSNorm(hidden_size=8)

    x = torch.randn(2, 3, 8, dtype=torch.float32)
    y = layer(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_rmsnorm_matches_manual_calculation() -> None:
    layer = RMSNorm(hidden_size=4, eps=1e-6)

    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor([1.0, 2.0, 0.5, -1.0])
        )

    x = torch.tensor(
        [[[3.0, 4.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    y = layer(x)

    rms = torch.sqrt(
        torch.tensor((3.0**2 + 4.0**2) / 4 + 1e-6)
    )

    expected = torch.tensor(
        [[[3.0 / rms, 2.0 * 4.0 / rms, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        y,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_rmsnorm_zero_input_has_no_nan() -> None:
    layer = RMSNorm(hidden_size=8)

    x = torch.zeros(2, 3, 8)
    y = layer(x)

    assert not torch.isnan(y).any()
    torch.testing.assert_close(y, torch.zeros_like(y))


def test_rmsnorm_state_dict_contains_only_weight() -> None:
    layer = RMSNorm(hidden_size=8)

    assert set(layer.state_dict().keys()) == {"weight"}


def test_swiglu_keeps_shape_and_dtype() -> None:
    layer = SwiGLUMLP(
        hidden_size=8,
        intermediate_size=24,
    )

    x = torch.randn(2, 3, 8, dtype=torch.float32)
    y = layer(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_swiglu_matches_manual_calculation() -> None:
    layer = SwiGLUMLP(
        hidden_size=2,
        intermediate_size=3,
    )

    gate_weight = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, -1.0],
        ]
    )

    up_weight = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ]
    )

    down_weight = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, -1.0],
        ]
    )

    with torch.no_grad():
        layer.gate_proj.weight.copy_(gate_weight)
        layer.up_proj.weight.copy_(up_weight)
        layer.down_proj.weight.copy_(down_weight)

    x = torch.tensor(
        [[[1.0, 2.0]]],
        dtype=torch.float32,
    )

    gate = F.silu(x @ gate_weight.T)
    up = x @ up_weight.T
    expected = (gate * up) @ down_weight.T

    actual = layer(x)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_swiglu_state_dict_names_match_qwen_style() -> None:
    layer = SwiGLUMLP(
        hidden_size=8,
        intermediate_size=24,
    )

    assert set(layer.state_dict().keys()) == {
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    }


def test_rotate_half_matches_expected_layout() -> None:
    x = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0]]]]
    )

    rotated = rotate_half(x)

    expected = torch.tensor(
        [[[[-3.0, -4.0, 1.0, 2.0]]]]
    )

    torch.testing.assert_close(rotated, expected)


def test_rope_returns_expected_shapes_and_dtype() -> None:
    rope = RotaryEmbedding(
        head_dim=8,
        theta=10_000.0,
    )

    position_ids = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
        ],
        dtype=torch.long,
    )

    cos, sin = rope(
        position_ids,
        dtype=torch.float32,
    )

    assert cos.shape == (2, 3, 8)
    assert sin.shape == (2, 3, 8)
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32


def test_rope_position_zero_does_not_change_q_or_k() -> None:
    rope = RotaryEmbedding(
        head_dim=4,
        theta=10_000.0,
    )

    q = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0]]]]
    )

    k = torch.tensor(
        [[[[5.0, 6.0, 7.0, 8.0]]]]
    )

    position_ids = torch.zeros(
        (1, 1),
        dtype=torch.long,
    )

    cos, sin = rope(
        position_ids,
        dtype=q.dtype,
    )

    q_rotated, k_rotated = apply_rope(
        q,
        k,
        cos,
        sin,
    )

    torch.testing.assert_close(q_rotated, q)
    torch.testing.assert_close(k_rotated, k)


def test_rope_position_one_matches_manual_rotation() -> None:
    rope = RotaryEmbedding(
        head_dim=4,
        theta=10_000.0,
    )

    q = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0]]]]
    )

    position_ids = torch.tensor(
        [[1]],
        dtype=torch.long,
    )

    cos, sin = rope(
        position_ids,
        dtype=q.dtype,
    )

    q_rotated, _ = apply_rope(
        q,
        q,
        cos,
        sin,
    )

    angle_0 = 1.0
    angle_1 = 0.01

    expected = torch.tensor(
        [[[
            [
                1.0 * math.cos(angle_0) - 3.0 * math.sin(angle_0),
                2.0 * math.cos(angle_1) - 4.0 * math.sin(angle_1),
                3.0 * math.cos(angle_0) + 1.0 * math.sin(angle_0),
                4.0 * math.cos(angle_1) + 2.0 * math.sin(angle_1),
            ]
        ]]],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        q_rotated,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_rope_has_no_trainable_or_checkpoint_weights() -> None:
    rope = RotaryEmbedding(
        head_dim=8,
        theta=10_000.0,
    )

    assert list(rope.parameters()) == []
    assert "inv_freq" not in rope.state_dict()
