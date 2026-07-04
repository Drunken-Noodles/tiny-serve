from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from elenkhos_serve.config import QwenConfig
from elenkhos_serve.model import (
    QwenForCausalLM,
    load_hf_weights,
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


def write_tiny_checkpoint(
    model: QwenForCausalLM,
    checkpoint_dir: Path,
) -> dict[str, torch.Tensor]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    source_state: dict[str, torch.Tensor] = {}

    for name, tensor in model.state_dict().items():
        # In a real tied-weight checkpoint, lm_head.weight may not exist
        # because embed_tokens.weight already owns those exact values.
        if name == "lm_head.weight":
            continue

        source_state[name] = tensor.detach().cpu().clone()

    save_file(
        source_state,
        str(checkpoint_dir / "model.safetensors"),
    )

    return source_state


def test_loader_copies_all_non_alias_weights(
    tmp_path: Path,
) -> None:
    torch.manual_seed(123)

    config = make_tiny_qwen3_config()

    source_model = QwenForCausalLM(config)
    source_state = write_tiny_checkpoint(
        source_model,
        tmp_path,
    )

    torch.manual_seed(999)

    target_model = QwenForCausalLM(config)

    report = load_hf_weights(
        target_model,
        tmp_path,
    )

    target_state = target_model.state_dict()

    for name, expected_tensor in source_state.items():
        torch.testing.assert_close(
            target_state[name],
            expected_tensor,
        )

    assert report.source_tensor_count == len(source_state)
    assert report.loaded_tensor_count == len(source_state)

    assert (
        target_model.lm_head.weight.data_ptr()
        == target_model.model.embed_tokens.weight.data_ptr()
    )


def test_loader_rejects_unknown_checkpoint_weight(
    tmp_path: Path,
) -> None:
    config = make_tiny_qwen3_config()
    source_model = QwenForCausalLM(config)

    source_state = write_tiny_checkpoint(
        source_model,
        tmp_path,
    )

    source_state["not_a_real_qwen_weight"] = torch.zeros(1)

    save_file(
        source_state,
        str(tmp_path / "model.safetensors"),
    )

    target_model = QwenForCausalLM(config)

    try:
        load_hf_weights(target_model, tmp_path)
    except ValueError as error:
        assert "does not expect" in str(error)
    else:
        raise AssertionError(
            "Expected loader to reject unknown weight."
        )
