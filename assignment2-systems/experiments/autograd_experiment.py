import torch
from torch import nn
from cs336_basics.model import RotaryEmbedding, TransformerBlock


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        return self.weight * x


d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
block = TransformerBlock(
    d_model=d_model,
    d_ff=d_ff,
    num_heads=num_heads,
    positional_encoder=RotaryEmbedding(
        dim=d_model // num_heads, context_length=context_length
    ),
)
block = torch.compile(block, fullgraph=True)
x = torch.randn((4, context_length, d_model), requires_grad=True)

total_size_bytes = 0


def pack_hook(t):
    if isinstance(
        t, torch.nn.Parameter
    ):  # Skip logging parameters to avoid double counting
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


# with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
#     y = block(x)
#     loss = y.float().sum()
#     loss.backward()
# print(
#     f"Total size of saved tensors in single TransformerBlock: {total_size_bytes / (1024**2):.2f} MiB"
# )

from torch.utils.checkpoint import checkpoint


def two_blocks(x):
    x = block(x)
    x = block(x)
    return x


def four_blocks_checkpoint(x):
    x = checkpoint(two_blocks, x, use_reentrant=False, preserve_rng_state=False)
    x = checkpoint(two_blocks, x, use_reentrant=False, preserve_rng_state=False)
    return x


with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks_checkpoint(x)
print(
    f"Total size of saved tensors in four TransformerBlocks with checkpointing: {total_size_bytes / (1024**2):.2f} MiB"
)
