import torch

from tiny_serve.config import QwenConfig
from tiny_serve.model import (
    QwenForCausalLM,
    QwenModel,
)


def make_tiny_qwen3_config() -> QwenConfig:
    return QwenConfig(
        vocab_size=32,
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


def test_qwen_model_returns_final_hidden_states() -> None:
    torch.manual_seed(0)

    config = make_tiny_qwen3_config()
    model = QwenModel(config).eval()

    input_ids = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=torch.long,
    )

    hidden_states = model(input_ids)

    assert hidden_states.shape == (
        2,
        3,
        config.hidden_size,
    )

    assert hidden_states.dtype == torch.float32


def test_qwen_model_default_positions_match_explicit_positions() -> None:
    torch.manual_seed(0)

    config = make_tiny_qwen3_config()
    model = QwenModel(config).eval()

    input_ids = torch.tensor(
        [[1, 2, 3, 4]],
        dtype=torch.long,
    )

    output_default = model(input_ids)

    position_ids = torch.tensor(
        [[0, 1, 2, 3]],
        dtype=torch.long,
    )

    output_explicit = model(
        input_ids=input_ids,
        position_ids=position_ids,
    )

    torch.testing.assert_close(
        output_default,
        output_explicit,
        rtol=1e-5,
        atol=1e-6,
    )


def test_qwen_causal_lm_returns_vocabulary_logits() -> None:
    torch.manual_seed(0)

    config = make_tiny_qwen3_config()
    model = QwenForCausalLM(config).eval()

    input_ids = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=torch.long,
    )

    logits = model(input_ids)

    assert logits.shape == (
        2,
        3,
        config.vocab_size,
    )


def test_lm_head_and_embedding_share_the_same_weight() -> None:
    config = make_tiny_qwen3_config()
    model = QwenForCausalLM(config)

    assert (
        model.lm_head.weight.data_ptr()
        == model.model.embed_tokens.weight.data_ptr()
    )


def test_qwen_model_has_huggingface_style_weight_names() -> None:
    config = make_tiny_qwen3_config()
    model = QwenForCausalLM(config)

    keys = set(model.state_dict().keys())

    expected_keys = {
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    }

    assert expected_keys.issubset(keys)
