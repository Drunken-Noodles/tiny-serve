from __future__ import annotations

import torch
from torch import nn

from elenkhos_serve.config import QwenConfig
from elenkhos_serve.model.attention import QwenAttention
from elenkhos_serve.model.layers import (
    RMSNorm,
    RotaryEmbedding,
    SwiGLUMLP,
)


class QwenDecoderLayer(nn.Module):
    """
    One Qwen-style decoder layer.

    Structure:

        hidden_states
        ↓
        RMSNorm
        ↓
        Self-Attention
        ↓
        Residual add
        ↓
        RMSNorm
        ↓
        SwiGLU MLP
        ↓
        Residual add
    """

    def __init__(
        self,
        config: QwenConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()

        if not 0 <= layer_idx < config.num_hidden_layers:
            raise ValueError(
                f"layer_idx must be between 0 and "
                f"{config.num_hidden_layers - 1}, but got {layer_idx}."
            )

        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.self_attn = QwenAttention(
            config=config,
            layer_idx=layer_idx,
        )

        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.mlp = SwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class QwenModel(nn.Module):
    """
    The Qwen transformer backbone.

    Input:
        token IDs

    Output:
        final hidden states

    This class does not convert hidden states into vocabulary logits.
    That job belongs to QwenForCausalLM.
    """

    def __init__(self, config: QwenConfig) -> None:
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # [vocab_size, hidden_size]
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )

        # ModuleList is required.
        # A normal Python list would not register these layers as model modules.
        self.layers = nn.ModuleList(
            [
                QwenDecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # RoPE has no trainable checkpoint weight.
        # It generates cos/sin from position IDs.
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            theta=config.rope_theta,
        )

    def _default_position_ids(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create positions for a normal prompt prefill.

        Example:
            input_ids shape: [2, 4]

        Output:
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ]

        Later, KV cache decoding will start positions after past length.
        """
        batch_size, sequence_length = input_ids.shape

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
            dtype=torch.long,
        )

        return positions.unsqueeze(0).expand(batch_size, -1)

    def _validate_input_ids(
        self,
        input_ids: torch.Tensor,
    ) -> None:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch_size, sequence_length], "
                f"but got {tuple(input_ids.shape)}."
            )

        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "input_ids must use an integer dtype such as torch.long, "
                f"but got {input_ids.dtype}."
            )

        if input_ids.numel() == 0:
            raise ValueError("input_ids cannot be empty.")

        min_token_id = int(input_ids.min().item())
        max_token_id = int(input_ids.max().item())

        if min_token_id < 0:
            raise ValueError(
                f"Token IDs must be non-negative, but found {min_token_id}."
            )

        if max_token_id >= self.vocab_size:
            raise ValueError(
                f"Token ID {max_token_id} is outside vocab_size "
                f"{self.vocab_size}."
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:
                [batch_size, sequence_length]

            position_ids:
                Optional [batch_size, sequence_length].
                If omitted, positions begin at zero.

            attention_mask:
                Optional additive attention mask.
                Current MVP supports the 4D additive mask format only:
                [batch or 1, 1 or heads, sequence, sequence].

        Returns:
            Final hidden states:
            [batch_size, sequence_length, hidden_size]
        """
        self._validate_input_ids(input_ids)

        input_ids = input_ids.long()

        if position_ids is None:
            position_ids = self._default_position_ids(input_ids)
        else:
            if position_ids.shape != input_ids.shape:
                raise ValueError(
                    "position_ids must have the same shape as input_ids. "
                    f"Got position_ids={tuple(position_ids.shape)} and "
                    f"input_ids={tuple(input_ids.shape)}."
                )

            position_ids = position_ids.to(
                device=input_ids.device,
                dtype=torch.long,
            )

        # [B, S] -> [B, S, H]
        hidden_states = self.embed_tokens(input_ids)

        # Shared RoPE values for all layers.
        cos, sin = self.rotary_emb(
            position_ids=position_ids,
            dtype=hidden_states.dtype,
        )

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                position_embeddings=(cos, sin),
                attention_mask=attention_mask,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class QwenForCausalLM(nn.Module):
    """
    Complete causal language model.

    Input:
        token IDs

    Output:
        vocabulary logits for each token position
    """

    def __init__(self, config: QwenConfig) -> None:
        super().__init__()

        self.config = config
        self.model = QwenModel(config)

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        if config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self) -> None:
        """
        Make LM head and token embedding use the exact same Parameter object.
        """
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            logits:
            [batch_size, sequence_length, vocab_size]
        """
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        return self.lm_head(hidden_states)
