"""Numerically stable neural-network utility functions for Assignment 1."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    """Apply numerically stable softmax along ``dim``.

    The output has the same shape as ``x`` and sums to one along ``dim``.
    Implement this using the max-subtraction trick rather than relying on a
    probability-space computation that can overflow.
    """
    max_value = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_value)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute mean cross-entropy from unnormalized logits.

    ``inputs`` has shape ``(..., vocab_size)`` and ``targets`` has shape
    ``(...)``.  The function should use a log-sum-exp formulation so that
    large logits remain numerically stable.
    """
    log_probs = inputs - torch.logsumexp(inputs, dim=-1, keepdim=True)
    return -log_probs[
        torch.arange(inputs.shape[0], device=inputs.device), targets
    ].mean()


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """Clip the global L2 norm of all available parameter gradients in place.

    Parameters without gradients should be ignored.  If the current global
    norm is already at most ``max_l2_norm``, this function should do nothing.
    """
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")

    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return

    squared_norm = sum(torch.sum(p.grad.square()) for p in parameters)  # type: ignore
    norm = torch.sqrt(squared_norm)  # type: ignore

    if norm > max_l2_norm:
        scale = max_l2_norm / (norm + 1e-6)
        with torch.no_grad():
            for parameter in parameters:
                parameter.grad.mul_(scale)  # type: ignore
