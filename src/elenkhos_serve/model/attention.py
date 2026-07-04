from __future__ import annotations

import torch
from torch import nn

from elenkhos_serve.config import QwenConfig
from elenkhos_serve.model.layers import RMSNorm, apply_rope


def repeat_kv(
    hidden_states: torch.Tensor,
    n_rep: int,
) -> torch.Tensor:
    """
    Repeat each KV head so it can match the number of Query heads.

    Input:
        [batch, kv_heads, sequence, head_dim]

    Output:
        [batch, kv_heads * n_rep, sequence, head_dim]
    """
    if hidden_states.ndim != 4:
        raise ValueError(
            "hidden_states must have shape "
            "[batch, kv_heads, sequence, head_dim]."
        )

    if n_rep <= 0:
        raise ValueError(
            f"n_rep must be positive, but got {n_rep}."
        )

    if n_rep == 1:
        return hidden_states

    batch_size, kv_heads, sequence_length, head_dim = hidden_states.shape

    expanded = hidden_states[:, :, None, :, :].expand(
        batch_size,
        kv_heads,
        n_rep,
        sequence_length,
        head_dim,
    )

    return expanded.reshape(
        batch_size,
        kv_heads * n_rep,
        sequence_length,
        head_dim,
    )


class QwenAttention(nn.Module):
    """
    Qwen-style causal self-attention.

    Current scope:
    - QK-norm
    - RoPE
    - Grouped Query Attention
    - full-sequence causal attention

    Deliberately not included yet:
    - KV cache
    - paged KV cache
    - batching scheduler
    - SDPA backend switching
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

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_kv_groups
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(
            self.hidden_size,
            config.q_projection_size,
            bias=config.attention_bias,
        )

        self.k_proj = nn.Linear(
            self.hidden_size,
            config.kv_projection_size,
            bias=config.attention_bias,
        )

        self.v_proj = nn.Linear(
            self.hidden_size,
            config.kv_projection_size,
            bias=config.attention_bias,
        )

        self.o_proj = nn.Linear(
            config.q_projection_size,
            self.hidden_size,
            bias=config.attention_bias,
        )

        if config.use_qk_norm:
            self.q_norm: nn.Module = RMSNorm(
                self.head_dim,
                eps=config.rms_norm_eps,
            )

            self.k_norm: nn.Module = RMSNorm(
                self.head_dim,
                eps=config.rms_norm_eps,
            )
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project hidden states into Q, K, V and split into attention heads.

        Returns:
            q: [batch, q_heads,  sequence, head_dim]
            k: [batch, kv_heads, sequence, head_dim]
            v: [batch, kv_heads, sequence, head_dim]
        """
        batch_size, sequence_length, _ = hidden_states.shape

        q = self.q_proj(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        )

        k = self.k_proj(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )

        v = self.v_proj(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )

        # RMSNorm acts on the final dimension: head_dim.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # [B, S, H, D] -> [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    @staticmethod
    def _make_causal_mask(
        sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Return an additive causal mask shaped [1, 1, S, S].

        Allowed positions get 0.
        Future positions get a very negative number.
        """
        future_positions = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        mask = torch.zeros(
            sequence_length,
            sequence_length,
            device=device,
            dtype=dtype,
        )

        mask = mask.masked_fill(
            future_positions,
            torch.finfo(dtype).min,
        )

        return mask.unsqueeze(0).unsqueeze(0)

    def _eager_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Readable reference attention implementation.

        Inputs:
            query_states: [B, heads, S, D]
            key_states:   [B, heads, S, D]
            value_states: [B, heads, S, D]

        Returns:
            [B, heads, S, D]
        """
        _, _, sequence_length, _ = query_states.shape

        attention_scores = (
            torch.matmul(
                query_states,
                key_states.transpose(-2, -1),
            )
            * self.scaling
        )

        causal_mask = self._make_causal_mask(
            sequence_length=sequence_length,
            device=attention_scores.device,
            dtype=attention_scores.dtype,
        )

        attention_scores = attention_scores + causal_mask

        if attention_mask is not None:
            if attention_mask.ndim != 4:
                raise ValueError(
                    "attention_mask must have shape "
                    "[batch or 1, 1 or heads, sequence, sequence]."
                )

            expected_last_dims = (
                sequence_length,
                sequence_length,
            )

            if attention_mask.shape[-2:] != expected_last_dims:
                raise ValueError(
                    "attention_mask has incorrect final dimensions. "
                    f"Expected {expected_last_dims}, "
                    f"but got {tuple(attention_mask.shape[-2:])}."
                )

            attention_scores = attention_scores + attention_mask.to(
                device=attention_scores.device,
                dtype=attention_scores.dtype,
            )

        # Match the stable attention pattern:
        # softmax in float32, then return to the model dtype.
        attention_probs = torch.softmax(
            attention_scores,
            dim=-1,
            dtype=torch.float32,
        ).to(query_states.dtype)

        return torch.matmul(attention_probs, value_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:
                [batch, sequence, hidden_size]

            position_embeddings:
                (cos, sin), each [batch, sequence, head_dim]

            attention_mask:
                Optional additive mask. Zero means allowed;
                a very negative value means blocked.

        Returns:
            [batch, sequence, hidden_size]
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape "
                "[batch, sequence, hidden_size]."
            )

        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden_size {self.hidden_size}, "
                f"but got {hidden_states.shape[-1]}."
            )

        q, k, v = self._project_qkv(hidden_states)

        cos, sin = position_embeddings
        q, k = apply_rope(q, k, cos, sin)

        # GQA:
        # K/V go from [B, 8, S, D] to [B, 16, S, D] for Qwen3-0.6B.
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attention_output = self._eager_attention(
            query_states=q,
            key_states=k,
            value_states=v,
            attention_mask=attention_mask,
        )

        # [B, heads, S, D] -> [B, S, heads, D]
        attention_output = attention_output.transpose(1, 2).contiguous()

        batch_size, sequence_length, _, _ = attention_output.shape

        attention_output = attention_output.reshape(
            batch_size,
            sequence_length,
            self.num_attention_heads * self.head_dim,
        )

        return self.o_proj(attention_output)
