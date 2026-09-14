"""Tokenized dataset utilities for the Assignment 1 training loop."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray


def get_batch(
    dataset: NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample input/next-token batches from a 1D token array.

    ``dataset`` may be a regular NumPy array or an ``np.memmap``.  The output
    tensors should have shape ``(batch_size, context_length)`` and be placed
    on ``device``.  This function is intentionally the only place where
    random contiguous training windows are sampled.
    """
    if dataset.ndim != 1:
        raise ValueError("dataset must be a 1D token array")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if len(dataset) <= context_length:
        raise ValueError("dataset must contain more tokens than context_length")

    # The exclusive upper bound guarantees i + context_length is a valid
    # target index.  Therefore i ranges from 0 through n - context_length - 1.
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    offsets = np.arange(context_length)
    input_indices = starts[:, None] + offsets[None, :]

    inputs = torch.as_tensor(dataset[input_indices], dtype=torch.long, device=device)
    targets = torch.as_tensor(dataset[input_indices + 1], dtype=torch.long, device=device)
    return inputs, targets
