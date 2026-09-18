"""Capture a PyTorch/Chrome trace and a screenshot-ready training-step summary.

Run this on a CUDA node. The SVG is designed for direct upload to slides; the
JSON trace can be opened in chrome://tracing or Perfetto.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import torch

from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW
from cs336_basics.transformer import TransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one Transformer training step.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/profile"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5, help="Profiled steps; must be at least 3.")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


def train_step(model, optimizer, inputs, targets, max_grad_norm: float) -> None:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(inputs)
        loss = cross_entropy(logits.flatten(0, 1), targets.flatten())
    loss.backward()
    gradient_clipping(model.parameters(), max_grad_norm)
    optimizer.step()


def write_svg(events, output: Path, args: argparse.Namespace) -> None:
    """Write a dependency-free, screenshot-ready operator time chart."""
    rows = []
    for event in events:
        time_us = getattr(event, "self_device_time_total", 0.0)
        if time_us > 0:
            rows.append((event.key, time_us))
    rows.sort(key=lambda row: row[1], reverse=True)
    rows = rows[:10]
    total_us = sum(time_us for _, time_us in rows)
    width, height, left, top, bar_height = 1400, 150 + 48 * len(rows), 390, 110, 30
    max_us = max((time_us for _, time_us in rows), default=1.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0f172a"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#e2e8f0}.title{font-size:28px;font-weight:bold}.meta{font-size:17px;fill:#94a3b8}.label{font-size:16px}.value{font-size:16px;font-weight:bold}</style>',
        '<text class="title" x="50" y="48">Transformer training-step CUDA profile</text>',
        f'<text class="meta" x="50" y="78">B={args.batch_size}, T={args.context_length}, d={args.d_model}, layers={args.num_layers} · self CUDA time by operator</text>',
    ]
    for index, (name, time_us) in enumerate(rows):
        y = top + index * 48
        bar_width = 850 * time_us / max_us
        label = html.escape(name[:42])
        pct = 100 * time_us / total_us if total_us else 0.0
        parts.extend(
            [
                f'<text class="label" x="50" y="{y + 21}">{label}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" rx="5" fill="#38bdf8"/>',
                f'<text class="value" x="{left + bar_width + 14:.1f}" y="{y + 21}">{time_us / 1000:.2f} ms ({pct:.1f}%)</text>',
            ]
        )
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("This script requires a CUDA-enabled PyTorch runtime.")
    if args.steps < 3:
        raise ValueError("--steps must be at least 3")

    device = torch.device(args.device)
    torch.manual_seed(42)
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=device,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    inputs = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    for _ in range(args.warmup_steps):
        train_step(model, optimizer, inputs, targets, args.max_grad_norm)
    torch.cuda.synchronize(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "training_step_trace.json"
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        for _ in range(args.steps):
            train_step(model, optimizer, inputs, targets, args.max_grad_norm)
            profiler.step()
    torch.cuda.synchronize(device)
    profiler.export_chrome_trace(str(trace_path))
    write_svg(profiler.key_averages(), args.output_dir / "training_step_profile.svg", args)
    print(f"screenshot-ready chart: {args.output_dir / 'training_step_profile.svg'}")
    print(f"Chrome/Perfetto trace: {trace_path}")


if __name__ == "__main__":
    main()
