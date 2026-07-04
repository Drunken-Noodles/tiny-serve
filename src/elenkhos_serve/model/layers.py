import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization used by Qwen-style models.

    This layer normalizes the final dimension of the input tensor.
    For hidden states, that final dimension is usually hidden_size.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, but got {hidden_size}."
            )

        if eps <= 0:
            raise ValueError(
                f"eps must be positive, but got {eps}."
            )

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states:
                Tensor shaped [batch_size, sequence_length, hidden_size].

        Returns:
            Tensor with the same shape and dtype as hidden_states.
        """
        input_dtype = hidden_states.dtype

        # Compute RMS in float32 for numerical stability.
        hidden_states_fp32 = hidden_states.float()

        variance = hidden_states_fp32.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        normalized = hidden_states_fp32 * torch.rsqrt(
            variance + self.eps
        )

        return self.weight * normalized.to(input_dtype)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.weight.numel()}, "
            f"eps={self.eps}"
        )


class SwiGLUMLP(nn.Module):
    """
    Qwen-style SwiGLU feed-forward network.

    Input:
        [batch_size, sequence_length, hidden_size]

    Output:
        [batch_size, sequence_length, hidden_size]
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, but got {hidden_size}."
            )

        if intermediate_size <= 0:
            raise ValueError(
                "intermediate_size must be positive, "
                f"but got {intermediate_size}."
            )

        # Qwen checkpoint names use these exact layer names.
        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states:
                [batch_size, sequence_length, hidden_size]

        Returns:
            [batch_size, sequence_length, hidden_size]
        """
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)

        hidden_states = gate * up

        return self.down_proj(hidden_states)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.gate_proj.in_features}, "
            f"intermediate_size={self.gate_proj.out_features}"
        )


class RotaryEmbedding(nn.Module):
    """
    Creates RoPE cosine and sine values for a sequence of token positions.

    This module has no trainable parameters. Its values are fully determined
    by head_dim, rope_theta, and position_ids.
    """

    def __init__(
        self,
        head_dim: int,
        theta: float,
    ) -> None:
        super().__init__()

        if head_dim <= 0:
            raise ValueError(
                f"head_dim must be positive, but got {head_dim}."
            )

        if head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even because RoPE rotates pairs of values, "
                f"but got {head_dim}."
            )

        if theta <= 0:
            raise ValueError(
                f"theta must be positive, but got {theta}."
            )

        self.head_dim = head_dim
        self.theta = theta

        # Shape: [head_dim / 2]
        #
        # For head_dim=128:
        # arange values are [0, 2, 4, ..., 126].
        dimension_indices = torch.arange(
            0,
            head_dim,
            2,
            dtype=torch.float32,
        )

        inv_freq = 1.0 / (
            theta ** (dimension_indices / head_dim)
        )

        # A buffer moves automatically with model.to(device), but it is not
        # trainable and is not expected inside Hugging Face checkpoint weights.
        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            position_ids:
                Integer tensor shaped [batch_size, sequence_length].

            dtype:
                Output dtype, normally q.dtype or k.dtype.

        Returns:
            cos, sin:
                Both shaped [batch_size, sequence_length, head_dim].
        """
        if position_ids.ndim != 2:
            raise ValueError(
                "position_ids must have shape [batch_size, sequence_length], "
                f"but got shape {tuple(position_ids.shape)}."
            )

        # We do frequency math in float32 for numerical stability.
        positions = position_ids.to(
            device=self.inv_freq.device,
            dtype=torch.float32,
        )

        # [batch, seq, 1] * [1, 1, head_dim/2]
        # -> [batch, seq, head_dim/2]
        frequencies = positions.unsqueeze(-1) * self.inv_freq.view(
            1,
            1,
            -1,
        )

        # Qwen/Hugging Face uses the "half-split" RoPE layout:
        # [f0, f1, ..., fN, f0, f1, ..., fN]
        angles = torch.cat(
            (frequencies, frequencies),
            dim=-1,
        )

        cos = angles.cos().to(dtype)
        sin = angles.sin().to(dtype)

        return cos, sin

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, "
            f"theta={self.theta}"
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Transform:
        [x1, x2] -> [-x2, x1]

    The split happens across the final dimension.
    """
    head_dim = x.shape[-1]

    if head_dim % 2 != 0:
        raise ValueError(
            "The final dimension must be even for RoPE rotation, "
            f"but got {head_dim}."
        )

    first_half = x[..., : head_dim // 2]
    second_half = x[..., head_dim // 2 :]

    return torch.cat(
        (-second_half, first_half),
        dim=-1,
    )


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to Q and K.

    Expected shapes:
        q:   [batch, q_heads,  seq, head_dim]
        k:   [batch, kv_heads, seq, head_dim]
        cos: [batch,          seq, head_dim]
        sin: [batch,          seq, head_dim]
    """
    if q.ndim != 4:
        raise ValueError(
            f"q must have 4 dimensions, but got shape {tuple(q.shape)}."
        )

    if k.ndim != 4:
        raise ValueError(
            f"k must have 4 dimensions, but got shape {tuple(k.shape)}."
        )

    if cos.shape != sin.shape:
        raise ValueError(
            "cos and sin must have the same shape, but got "
            f"{tuple(cos.shape)} and {tuple(sin.shape)}."
        )

    if cos.ndim != 3:
        raise ValueError(
            "cos and sin must have shape [batch, seq, head_dim], "
            f"but got {tuple(cos.shape)}."
        )

    expected_batch = q.shape[0]
    expected_seq = q.shape[2]
    expected_head_dim = q.shape[3]

    if cos.shape != (
        expected_batch,
        expected_seq,
        expected_head_dim,
    ):
        raise ValueError(
            "cos/sin shape must match q's batch, sequence, and head_dim. "
            f"Expected {(expected_batch, expected_seq, expected_head_dim)}, "
            f"but got {tuple(cos.shape)}."
        )

    if k.shape[0] != expected_batch:
        raise ValueError("q and k must have the same batch size.")

    if k.shape[2] != expected_seq:
        raise ValueError("q and k must have the same sequence length.")

    if k.shape[3] != expected_head_dim:
        raise ValueError("q and k must have the same head_dim.")

    # [batch, seq, head_dim]
    # -> [batch, 1, seq, head_dim]
    #
    # The 1 broadcasts over q_heads or kv_heads.
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)

    return q_rotated, k_rotated
