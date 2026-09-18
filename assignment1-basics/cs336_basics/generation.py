"""Autoregressive decoding scaffolding for a trained Transformer language model.

Complete the three functions in order: temperature scaling, top-p filtering,
then the autoregressive sampling loop.  The model produces logits, not
probabilities, so all sampling must begin from its final-position logits.
"""

from __future__ import annotations
from .nn_utils import softmax

import torch
from torch import Tensor, nn


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Tensor:
    """Sample one token per batch row from final-position logits.

    Args:
        logits: Shape ``(batch_size, vocab_size)``; values are unnormalized.
        temperature: Positive divisor applied before softmax. Smaller values
            make the distribution sharper.
        top_p: Nucleus threshold in ``(0, 1]``. At 1.0, do not filter.

    Returns:
        Integer token IDs with shape ``(batch_size,)``.

    Implementation plan:
        1. Divide logits by temperature and obtain stable probabilities.
        2. Sort probabilities descending per row.
        3. Keep the smallest sorted prefix whose cumulative mass reaches p.
           Always retain the highest-probability token.
        4. Renormalize the retained probabilities, sample with
           ``torch.multinomial``, and map sorted indices back to vocabulary IDs.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch_size, vocab_size)")
    if logits.shape[-1] == 0:
        raise ValueError("vocab_size cannot be zero")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    scaled_logits = logits / temperature
    probs = softmax(scaled_logits, dim=-1)
    sorted_probs, sorted_ids = torch.sort(
        probs,
        dim=-1,
        descending=True,
    )
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative_probs - sorted_probs < top_p
    keep[..., 0] = True
    filtered_probs = sorted_probs * keep
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    sampled_sorted_positions = torch.multinomial(
        filtered_probs,
        num_samples=1,
    )
    next_token_ids = torch.gather(
        sorted_ids,
        dim=-1,
        index=sampled_sorted_positions,
    ).squeeze(-1)
    return next_token_ids


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_token_ids: Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Tensor:
    """Generate a completion from a batch of token-ID prompts.

    ``prompt_token_ids`` has shape ``(batch_size, prompt_length)``. Return the
    prompt plus generated IDs, padded only by continuing generation for rows
    that have not emitted EOS. The initial implementation may require a batch
    size of one; support batched finished-row handling as an extension.

    Per decoding step:
        1. Feed only the trailing ``model.context_length`` IDs to the model.
        2. Select logits at ``[:, -1, :]``.
        3. Call ``sample_next_token`` and append its result.
        4. Stop at EOS (when supplied) or after ``max_new_tokens`` steps.
    """
    if prompt_token_ids.ndim != 2 or prompt_token_ids.shape[1] == 0:
        raise ValueError(
            "prompt_token_ids must have shape (batch_size, nonempty_length)"
        )
    if prompt_token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("prompt_token_ids must contain integer IDs")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if eos_token_id is not None and eos_token_id < 0:
        raise ValueError("eos_token_id must be non-negative")
   
    generated = prompt_token_ids
    for _ in range(max_new_tokens):
        context = generated[:, -model.context_length:]
        logits = model(context)
        next_logits = logits[:, -1, :]
        next_token_ids = sample_next_token(
            next_logits,
            temperature=temperature,
            top_p=top_p,
        )
        generated = torch.cat(
            [generated, next_token_ids.unsqueeze(-1)],
            dim=-1,
        )
        if eos_token_id is not None and torch.all(next_token_ids == eos_token_id):
          break
    
    return generated
    
