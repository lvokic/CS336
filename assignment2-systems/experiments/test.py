import gc

import numpy as np
import torch
from torch import Tensor, nn

from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmark import MODEL_CONFIGS, benchmark, get_model_config, profile

MODEL_SIZES = tuple(MODEL_CONFIGS)
CONTEXT_LENGTHS = (256,)
MODES = ("forward", "forward-backward", "train-step")
PROFILE_TARGET = ("small", 256, "forward")


def forward_step(
    model: nn.Module,
    inputs: Tensor,
) -> Tensor:
    model.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=inputs.device.type == "cuda",
        ):
            return model(inputs)


def forward_backward(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
) -> Tensor:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=inputs.device.type == "cuda"
    ):
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten()
        )

    loss.backward()
    return loss.detach()


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
) -> Tensor:
    """Run one forward, loss, backward, optional clipping, and optimizer step."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=inputs.device.type == "cuda"
    ):
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten()
        )

    loss.backward()
    optimizer.step()
    return loss.detach()


def main() -> None:
    device = "cuda"
    batch_size = 2
    vocab_size = 10_000
    train_tokens = np.memmap(
        "../assignment1-basics/data/tinystories_train_tokens.bin",
        dtype=np.uint16,
        mode="r",
    )
    max_token_id = int(train_tokens.max())
    if max_token_id >= vocab_size:
        raise ValueError(
            f"token data contains ID {max_token_id}, but vocab_size is {vocab_size}. "
            "Use a tokenizer-compatible dataset or increase vocab_size."
        )

    for model_size in MODEL_SIZES:
        config = get_model_config(model_size)
        for context_length in CONTEXT_LENGTHS:
            model = BasicsTransformerLM(
                vocab_size=vocab_size,
                context_length=context_length,
                d_model=config.d_model,
                d_ff=config.d_ff,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
            ).to(device)
            optimizer = AdamW(model.parameters(), lr=1e-3)

            # Data sampling and host-to-device transfer are intentionally outside
            # the measured closure, so this benchmark measures model computation.
            inputs, targets = get_batch(
                train_tokens,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            step_fns = {
                "forward": lambda: forward_step(model, inputs),
                "forward-backward": lambda: forward_backward(
                    model, optimizer, inputs, targets
                ),
                "train-step": lambda: train_step(model, optimizer, inputs, targets),
            }

            for mode in MODES:
                result = benchmark(
                    step_fns[mode],
                    warmup_steps=5,
                    measure_steps=10,
                    device=device,
                )
                print(
                    f"size={model_size:>6} context={context_length:>4} "
                    f"mode={mode:>16}: {result.mean_ms:.2f} ± {result.std_ms:.2f} ms"
                )

                if (model_size, context_length, mode) == PROFILE_TARGET:
                    report = profile(
                        step_fns[mode],
                        warmup_steps=5,
                        profile_steps=1,
                        device=device,
                        trace_path=f"experiments/profiles/{model_size}_{context_length}_{mode}.json",
                    )
                    print(report.table)
                    print(f"trace: {report.trace_path}")

            # step_fns' lambdas capture model and optimizer. Release those
            # references before allocating the next, potentially larger model.
            del step_fns, inputs, targets, optimizer, model
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
