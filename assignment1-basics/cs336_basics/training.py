"""Training-loop scaffolding for the decoder-only Transformer."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
import numpy as np

from .data import get_batch
from .nn_utils import cross_entropy, gradient_clipping
from .optimizer import get_lr_cosine_schedule, get_lr_wsd_schedule
from .checkpointing import save_checkpoint


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
    *,
    max_grad_norm: float | None = None,
) -> Tensor:
    """Run one forward, loss, backward, optional clipping, and optimizer step."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=inputs.device.type == "cuda"
    ):
        logits = model(inputs)
        loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss.backward()

    if max_grad_norm is not None:
        gradient_clipping(model.parameters(), max_grad_norm)

    optimizer.step()
    return loss.detach()


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataset,
    *,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    num_batches: int,
) -> float:
    """Estimate mean loss over randomly sampled validation batches."""
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        total_loss += loss.item()
    if was_training:
        model.train()
    return total_loss / num_batches


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset,
    *,
    lr_schedule: str,
    wsd_decay_start_iters: int | None,
    num_iterations: int,
    batch_size: int,
    context_length: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    max_grad_norm: float | None = None,
    valid_dataset=None,
    device: str | torch.device = "cpu",
    start_iteration: int = 0,
    eval_interval: int | None = None,
    eval_batches: int = 0,
    log_interval: int = 20,
    checkpoint_path: str | Path | None = None,
) -> list[dict[str, float]]:
    """Run configurable training and return logged metrics.

    The implementation should update the optimizer learning rate each
    iteration, sample batches from NumPy arrays or memmaps, periodically
    evaluate/log metrics, and optionally save checkpoints.
    """
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")
    if start_iteration < 0:
        raise ValueError("start_iteration cannot be negative")
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if (eval_interval is None) != (eval_batches == 0):
        raise ValueError(
            "set both eval_interval and a positive eval_batches, or neither"
        )
    if eval_interval is not None and eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    if eval_batches < 0:
        raise ValueError("eval_batches cannot be negative")
    if valid_dataset is None and eval_interval is not None:
        raise ValueError("valid_dataset is required when evaluation is enabled")

    metrics: list[dict[str, float]] = []
    for local_iteration in range(num_iterations):
        iteration = start_iteration + local_iteration
        completed_iteration = iteration + 1
        if lr_schedule == "cosine":
            lr = get_lr_cosine_schedule(
                iteration,
                max_learning_rate,
                min_learning_rate,
                warmup_iters,
                cosine_cycle_iters,
            )
        else:
            lr = get_lr_wsd_schedule(
                iteration,
                max_learning_rate,
                min_learning_rate,
                warmup_iters,
                wsd_decay_start_iters,
                cosine_cycle_iters
            )
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, targets = get_batch(
            train_dataset, batch_size, context_length, device=device
        )
        loss = train_step(
            model=model,
            optimizer=optimizer,
            inputs=inputs,
            targets=targets,
            max_grad_norm=max_grad_norm,
        )
        should_log = completed_iteration % log_interval == 0
        if should_log:
            record = {
                "iteration": float(completed_iteration),
                "train_loss": loss.item(),
                "learning_rate": lr,
            }
        should_evaluate = (
            valid_dataset is not None
            and eval_interval is not None
            and completed_iteration % eval_interval == 0
        )
        if should_evaluate:
            valid_loss = evaluate_loss(
                model,
                valid_dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
                num_batches=eval_batches,
            )
            record["valid_loss"] = valid_loss
            if checkpoint_path is not None:
                checkpoint_target = Path(checkpoint_path)
                checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    model, optimizer, completed_iteration, checkpoint_target
                )

        if completed_iteration % log_interval == 0 or should_evaluate:
            metrics.append(record)
            suffix = (
                f" valid_loss={record['valid_loss']:.4f}"
                if "valid_loss" in record
                else ""
            )
            print(
                f"step={completed_iteration} lr={lr:.3e} "
                f"train_loss={loss.item():.4f} "
                f"{suffix}"
            )
    if checkpoint_path is not None:
        checkpoint_target = Path(checkpoint_path)
        checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model,
            optimizer,
            iteration=start_iteration + num_iterations,
            out=checkpoint_target,
        )
    return metrics
