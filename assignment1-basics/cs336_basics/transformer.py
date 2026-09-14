"""Transformer building blocks for Assignment 1.

This module is intentionally organized from small components to the complete
language model:

    RMSNorm -> RoPE -> scaled dot-product attention
    -> causal multi-head attention -> TransformerBlock -> TransformerLM

The public class names and tensor shapes follow the assignment handout.  The
methods below are scaffolding for the implementation work; each TODO should
be implemented and tested independently before assembling the full model.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    """Bias-free linear layer with assignment-compatible weight layout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        with torch.no_grad():
            nn.init.trunc_normal_(
                self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std
            )

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.transpose(-1, -2)


class Embedding(nn.Module):
    """Token embedding lookup."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        with torch.no_grad():
            nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root mean square normalization without mean centering."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """Position-wise SwiGLU feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        gate = torch.nn.functional.silu(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)


class RotaryPositionalEmbedding(nn.Module):
    """Apply rotary position embeddings to ``(..., sequence, d_k)`` tensors."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        if theta <= 0:
            raise ValueError("theta must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        pair_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inverse_frequencies = theta ** (-pair_indices / d_k)
        angles = positions[:, None] * inverse_frequencies[None, :]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """Rotate x using positions shaped ``(..., sequence_length)``."""
        if x.shape[-1] != self.d_k:
            raise ValueError(f"expected last dimension {self.d_k}, got {x.shape[-1]}")
        if x.ndim < 2 or token_positions.ndim < 1:
            raise ValueError(
                "x must have a sequence dimension and token_positions must be non-empty"
            )
        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError("token_positions sequence length must match x")
        if token_positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_positions must contain integer positions")
        if torch.any(token_positions < 0) or torch.any(
            token_positions >= self.max_seq_len
        ):
            raise ValueError("token_positions are outside the RoPE range")

        while token_positions.ndim < x.ndim - 1:
            token_positions = token_positions.unsqueeze(-2)
        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)
        pairs = x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        even = pairs[..., 0]
        odd = pairs[..., 1]
        rotated = torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos),
            dim=-1,
        )
        return rotated.reshape_as(x)


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute masked scaled dot-product attention.

    Q, K, and V may have arbitrary shared leading batch dimensions.  The last
    two dimensions are ``(queries, d_k)``, ``(keys, d_k)``, and
    ``(keys, d_v)`` respectively.  A boolean mask uses True for allowed
    attention positions.
    """
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError("Q and K must have the same head dimension")
    if K.shape[-2] != V.shape[-2]:
        raise ValueError("K and V must have the same sequence length")
    if mask is not None and mask.dtype != torch.bool:
        raise TypeError("attention mask must have boolean dtype")

    scores = Q @ K.transpose(-1, -2)
    scores = scores / math.sqrt(Q.shape[-1])

    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    probabilities = torch.softmax(scores, dim=-1)
    return probabilities @ V


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float = 10_000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = (
            RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device=device)
            if max_seq_len is not None
            else None
        )

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        batch_size, seq_len, d_model = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.d_head).transpose(
            -3, -2
        )
        k = k.reshape(batch_size, seq_len, self.num_heads, self.d_head).transpose(
            -3, -2
        )
        v = v.reshape(batch_size, seq_len, self.num_heads, self.d_head).transpose(
            -3, -2
        )

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).expand(
                    batch_size, -1
                )
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        attn_weights = scaled_dot_product_attention(q, k, v, mask=mask).transpose(1, 2)
        attn_weights = attn_weights.reshape(batch_size, seq_len, d_model)

        return self.output_proj(attn_weights)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: attention followed by SwiGLU."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10_000.0,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        # Pre-norm attention sub-layer:
        # x -> RMSNorm -> self-attention -> residual addition
        x = x + self.attn(self.ln1(x), token_positions)

        # Pre-norm feed-forward sub-layer:
        # x -> RMSNorm -> SwiGLU -> residual addition
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10_000.0,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                context_length,
                theta=rope_theta,
                eps=eps,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: Tensor) -> Tensor:
        if in_indices.ndim != 2:
            raise ValueError("in_indices must have shape (batch_size, sequence_length)")
        if in_indices.shape[-1] > self.context_length:
            raise ValueError("input sequence exceeds context_length")

        batch_size, seq_len = in_indices.shape
        token_positions = torch.arange(
            seq_len,
            device=in_indices.device,
        ).expand(batch_size, -1)

        out = self.token_embeddings(in_indices)
        for layer in self.layers:
            out = layer(out, token_positions)
        out = self.ln_final(out)
        return self.lm_head(out)
