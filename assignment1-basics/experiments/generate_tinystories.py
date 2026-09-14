"""Generate a TinyStories completion from a training checkpoint.

The model-loading path is complete. Implement ``cs336_basics.generation``
before running this script successfully.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.bpe import Tokenizer
from cs336_basics.generation import generate
from cs336_basics.transformer import TransformerLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a TinyStories completion.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=device,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = Tokenizer.load(args.tokenizer)
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    eos_token_id = tokenizer._special_ids.get("<|endoftext|>")
    output_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
