"""Train the Assignment 1 TransformerLM on uint16 TinyStories token data.

The defaults are the handout's TinyStories model.  Begin with ``--steps 100``
to smoke-test the entire pipeline; then choose a batch size that fits the GPU
and increase the number of steps toward the desired token budget.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.checkpointing import load_checkpoint
from cs336_basics.optimizer import AdamW
from cs336_basics.training import train
from cs336_basics.transformer import TransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TinyStories Transformer.")
    parser.add_argument("--train-tokens", type=Path, required=True)
    parser.add_argument("--valid-tokens", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/tinystories_baseline"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=1_000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--lr-schedule",choices=("cosine", "wsd"),default="cosine")
    parser.add_argument("--wsd-decay-start-iters",type=int,default=16000)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.context_length <= 0 or args.batch_size <= 0:
        raise ValueError("batch size and context length must be positive")
    # The 1,000-step default is appropriate for the 40k-step baseline, but a
    # smoke test must spend most of its short budget actually learning.
    effective_warmup_iters = min(args.warmup_iters, max(1, args.steps // 10))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    train_tokens = np.memmap(args.train_tokens, dtype="<u2", mode="r")
    valid_tokens = np.memmap(args.valid_tokens, dtype="<u2", mode="r")
    for name, tokens in (("train", train_tokens), ("valid", valid_tokens)):
        if len(tokens) <= args.context_length:
            raise ValueError(f"{name} data is shorter than context_length")
        if int(tokens.max()) >= args.vocab_size:
            raise ValueError(f"{name} data contains an ID outside --vocab-size")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay)

    start_iteration = 0
    if args.resume is not None:
        start_iteration = load_checkpoint(args.resume, model, optimizer)
        print(f"resumed checkpoint at step {start_iteration}")
    if start_iteration >= args.steps:
        raise ValueError("checkpoint is already at or beyond --steps")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {**vars(args), "effective_warmup_iters": effective_warmup_iters},
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint = args.output_dir / "latest.pt"
    start = time.perf_counter()
    metrics = train(
        model,
        optimizer,
        train_tokens,
        lr_schedule=args.lr_schedule,
        wsd_decay_start_iters=args.wsd_decay_start_iters,
        num_iterations=args.steps - start_iteration,
        batch_size=args.batch_size,
        context_length=args.context_length,
        max_learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        warmup_iters=effective_warmup_iters,
        cosine_cycle_iters=args.steps,
        max_grad_norm=args.max_grad_norm,
        valid_dataset=valid_tokens,
        device=device,
        start_iteration=start_iteration,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        log_interval=args.log_interval,
        checkpoint_path=checkpoint,
    )
    elapsed = time.perf_counter() - start
    for metric in metrics:
        if "valid_loss" in metric:
            metric["valid_perplexity"] = math.exp(metric["valid_loss"])
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    trained_tokens = (args.steps - start_iteration) * args.batch_size * args.context_length
    print(f"finished {args.steps - start_iteration:,} steps in {elapsed / 60:.1f} min")
    print(f"throughput: {trained_tokens / elapsed:,.0f} tokens/s")
    print(f"checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
