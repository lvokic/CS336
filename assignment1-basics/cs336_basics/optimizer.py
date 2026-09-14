"""Optimizer and learning-rate schedule scaffolding for Assignment 1."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Optional

import torch


class SGD(torch.optim.Optimizer):
    """Assignment SGD optimizer with the handout's 1/sqrt(t+1) decay."""

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, defaults={"lr": lr})

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        """Apply one SGD update and return an optional closure loss."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                p.add_(p.grad, alpha=-lr / math.sqrt(t + 1))
                state["t"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        super().__init__(
            params,
            defaults={
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            },
        )

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        """Apply one AdamW update and return an optional closure loss."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            alpha = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = p.grad

                # AdamW applies weight decay independently of the gradient.
                p.mul_(1 - alpha * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1 - beta2,
                )

                adjusted_alpha = alpha * math.sqrt(1 - beta2**step) / (1 - beta1**step)
                denominator = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denominator, value=-adjusted_alpha)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Return linear-warmup/cosine-decay learning rate for iteration ``it``.

    The schedule has three regions: linear warmup before ``warmup_iters``,
    cosine annealing through ``cosine_cycle_iters``, and a constant minimum
    learning rate afterwards.
    """
    if it < 0:
        raise ValueError("it must be non-negative")
    if max_learning_rate < 0 or min_learning_rate < 0:
        raise ValueError("learning rates must be non-negative")
    if min_learning_rate > max_learning_rate:
        raise ValueError("min_learning_rate cannot exceed max_learning_rate")
    if warmup_iters < 0 or cosine_cycle_iters < warmup_iters:
        raise ValueError("invalid warmup and cosine cycle bounds")

    if it < warmup_iters:
        if warmup_iters == 0:
            return max_learning_rate
        return max_learning_rate * it / warmup_iters

    if it <= cosine_cycle_iters:
        cycle_length = cosine_cycle_iters - warmup_iters
        if cycle_length == 0:
            return min_learning_rate
        progress = (it - warmup_iters) / cycle_length
        return min_learning_rate + 0.5 * (
            1.0 + math.cos(math.pi * progress)
        ) * (max_learning_rate - min_learning_rate)

    return min_learning_rate
