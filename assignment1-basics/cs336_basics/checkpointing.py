"""Checkpoint save/load scaffolding for model training."""

from __future__ import annotations

import os
from typing import BinaryIO, IO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Save model state, optimizer state, and iteration to ``out``."""
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise ValueError("iteration must be a non-negative integer")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore a checkpoint and return the saved iteration number."""
    checkpoint = torch.load(src, weights_only=True)
    required = {"model_state_dict", "optimizer_state_dict", "iteration"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("invalid checkpoint format")

    iteration = checkpoint["iteration"]
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise ValueError("checkpoint iteration must be a non-negative integer")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return iteration
