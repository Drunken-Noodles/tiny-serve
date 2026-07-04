from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from tiny_serve.model.qwen import QwenForCausalLM


@dataclass(frozen=True)
class LoadReport:
    """
    A record of what the loader did.

    We keep this because later, when parity fails, we want evidence:
    which tensors loaded, which aliases were skipped, and how many
    source tensors existed.
    """

    checkpoint_dir: Path
    source_tensor_count: int
    loaded_tensor_names: tuple[str, ...]
    skipped_tied_aliases: tuple[str, ...]

    @property
    def loaded_tensor_count(self) -> int:
        return len(self.loaded_tensor_names)


def find_safetensor_files(
    checkpoint_dir: str | Path,
) -> list[Path]:
    """
    Support both:

    1. One-file checkpoints:
       model.safetensors

    2. Sharded checkpoints:
       model.safetensors.index.json
       model-00001-of-00002.safetensors
       model-00002-of-00002.safetensors
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}"
        )

    index_path = checkpoint_dir / "model.safetensors.index.json"

    if index_path.exists():
        raw_index = json.loads(
            index_path.read_text(encoding="utf-8")
        )

        weight_map = raw_index.get("weight_map")

        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(
                f"Invalid safetensors index file: {index_path}"
            )

        shard_names = sorted(set(weight_map.values()))
        shard_paths = [
            checkpoint_dir / shard_name
            for shard_name in shard_names
        ]

        missing_shards = [
            path
            for path in shard_paths
            if not path.exists()
        ]

        if missing_shards:
            raise FileNotFoundError(
                "Missing checkpoint shard files:\n"
                + "\n".join(
                    f"- {path}"
                    for path in missing_shards
                )
            )

        return shard_paths

    single_file = checkpoint_dir / "model.safetensors"

    if single_file.exists():
        return [single_file]

    all_safetensors = sorted(
        checkpoint_dir.glob("*.safetensors")
    )

    if not all_safetensors:
        raise FileNotFoundError(
            "Could not find model.safetensors or any "
            f"*.safetensors files in {checkpoint_dir}"
        )

    return all_safetensors


def _get_source_tensor_names(
    safetensor_files: list[Path],
) -> set[str]:
    """
    Read only names first.

    We do this before copying any values so that we can fail early if
    the checkpoint and our model disagree.
    """
    names: set[str] = set()

    for file_path in safetensor_files:
        with safe_open(
            str(file_path),
            framework="pt",
            device="cpu",
        ) as checkpoint:
            for name in checkpoint.keys():
                if name in names:
                    raise ValueError(
                        "The same tensor name appears in multiple "
                        f"safetensor files: {name}"
                    )

                names.add(name)

    return names


def _get_tied_aliases(
    model: QwenForCausalLM,
) -> dict[str, str]:
    """
    A tied weight may only appear once in safetensors.

    Qwen uses:
        lm_head.weight
        ==
        model.embed_tokens.weight

    If lm_head.weight is absent in checkpoint, loading embed_tokens.weight
    is enough because both point to the same PyTorch Parameter.
    """
    if not model.config.tie_word_embeddings:
        return {}

    embedding_weight = model.model.embed_tokens.weight
    lm_head_weight = model.lm_head.weight

    if embedding_weight.data_ptr() != lm_head_weight.data_ptr():
        raise RuntimeError(
            "Expected tied embedding and LM-head weights, "
            "but they are different Parameters."
        )

    return {
        "lm_head.weight": "model.embed_tokens.weight",
    }


def load_hf_weights(
    model: QwenForCausalLM,
    checkpoint_dir: str | Path,
) -> LoadReport:
    """
    Strictly load Hugging Face safetensors into our owned Qwen model.

    Rules:
    - Every required target tensor must exist in checkpoint.
    - Every checkpoint tensor must be expected by our model.
    - Every tensor shape must match exactly.
    - Tied lm_head.weight may be omitted by checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)
    safetensor_files = find_safetensor_files(
        checkpoint_dir
    )

    source_names = _get_source_tensor_names(
        safetensor_files
    )

    target_state = model.state_dict()
    target_names = set(target_state.keys())

    tied_aliases = _get_tied_aliases(model)
    alias_target_names = set(tied_aliases.keys())

    required_target_names = (
        target_names - alias_target_names
    )

    missing_target_names = (
        required_target_names - source_names
    )

    unexpected_source_names = (
        source_names - target_names
    )

    if missing_target_names:
        formatted = "\n".join(
            f"- {name}"
            for name in sorted(missing_target_names)
        )

        raise ValueError(
            "Checkpoint is missing required model weights:\n"
            f"{formatted}"
        )

    if unexpected_source_names:
        formatted = "\n".join(
            f"- {name}"
            for name in sorted(unexpected_source_names)
        )

        raise ValueError(
            "Checkpoint contains weights our model does not expect:\n"
            f"{formatted}"
        )

    loaded_names: list[str] = []

    with torch.no_grad():
        for file_path in safetensor_files:
            with safe_open(
                str(file_path),
                framework="pt",
                device="cpu",
            ) as checkpoint:
                for name in checkpoint.keys():
                    source_tensor = checkpoint.get_tensor(name)
                    target_tensor = target_state[name]

                    if source_tensor.shape != target_tensor.shape:
                        raise ValueError(
                            f"Shape mismatch for '{name}': "
                            f"checkpoint has "
                            f"{tuple(source_tensor.shape)}, "
                            f"but our model expects "
                            f"{tuple(target_tensor.shape)}."
                        )

                    target_tensor.copy_(
                        source_tensor.to(
                            device=target_tensor.device,
                            dtype=target_tensor.dtype,
                        )
                    )

                    loaded_names.append(name)

    return LoadReport(
        checkpoint_dir=checkpoint_dir,
        source_tensor_count=len(source_names),
        loaded_tensor_names=tuple(sorted(loaded_names)),
        skipped_tied_aliases=tuple(
            sorted(alias_target_names - source_names)
        ),
    )
