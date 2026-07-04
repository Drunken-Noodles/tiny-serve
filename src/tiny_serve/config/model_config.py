from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QwenConfig:
    """
    The architecture settings required to build a Qwen-style decoder-only model.

    This class is intentionally small. It contains only values that our owned
    model implementation actually needs.
    """

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    intermediate_size: int

    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int

    tie_word_embeddings: bool
    use_qk_norm: bool
    attention_bias: bool

    @property
    def num_kv_groups(self) -> int:
        """
        Number of query-head groups that share one KV head.

        Example for Qwen3-0.6B:
        16 query heads / 8 KV heads = 2 query heads per KV head.
        """
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def q_projection_size(self) -> int:
        """Output width of q_proj."""
        return self.num_attention_heads * self.head_dim

    @property
    def kv_projection_size(self) -> int:
        """Output width of k_proj and v_proj."""
        return self.num_key_value_heads * self.head_dim

    def validate(self) -> None:
        """
        Fail early if the model configuration is internally inconsistent.
        """
        problems: list[str] = []

        if self.vocab_size <= 0:
            problems.append("vocab_size must be positive.")

        if self.hidden_size <= 0:
            problems.append("hidden_size must be positive.")

        if self.num_hidden_layers <= 0:
            problems.append("num_hidden_layers must be positive.")

        if self.num_attention_heads <= 0:
            problems.append("num_attention_heads must be positive.")

        if self.num_key_value_heads <= 0:
            problems.append("num_key_value_heads must be positive.")

        if self.head_dim <= 0:
            problems.append("head_dim must be positive.")

        if self.intermediate_size <= 0:
            problems.append("intermediate_size must be positive.")

        if self.num_attention_heads % self.num_key_value_heads != 0:
            problems.append(
                "num_attention_heads must be divisible by num_key_value_heads "
                "for Grouped Query Attention."
            )

        if self.max_position_embeddings <= 0:
            problems.append("max_position_embeddings must be positive.")

        if self.rms_norm_eps <= 0:
            problems.append("rms_norm_eps must be positive.")

        if self.rope_theta <= 0:
            problems.append("rope_theta must be positive.")

        if problems:
            joined = "\n- ".join(problems)
            raise ValueError(f"Invalid QwenConfig:\n- {joined}")

    @classmethod
    def from_hf_config(cls, config_path: str | Path) -> "QwenConfig":
        """
        Load the model values we need from Hugging Face config.json.
        """
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Could not find config file: {path}")

        raw = json.loads(path.read_text(encoding="utf-8"))

        config = cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=int(raw["hidden_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(
                raw.get("num_key_value_heads", raw["num_attention_heads"])
            ),
            head_dim=int(
                raw.get(
                    "head_dim",
                    raw["hidden_size"] // raw["num_attention_heads"],
                )
            ),
            intermediate_size=int(raw["intermediate_size"]),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            rope_theta=float(raw.get("rope_theta", 1_000_000.0)),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", True)),

            # Qwen3 config.json does not explicitly store use_qk_norm,
            # but Qwen3 architecture uses q_norm and k_norm.
            use_qk_norm=bool(
                raw.get(
                    "use_qk_norm",
                    raw.get("model_type") == "qwen3",
                )
            ),

            attention_bias=bool(raw.get("attention_bias", False)),
        )

        config.validate()
        return config
