import math

import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    output_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q = tl.load(
        Q_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    )

    m_i = tl.full(
        (Q_TILE_SIZE,),
        -float("inf"),
        dtype=tl.float32,
    )
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k = tl.load(
            K_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        v = tl.load(
            V_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )

        qk = tl.dot(q, tl.trans(k)) * scale
        key_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        key_valid = key_offsets < N_KEYS

        if IS_CAUSAL:
            query_offsets = (
                query_tile_index * Q_TILE_SIZE
                + tl.arange(0, Q_TILE_SIZE)
            )
            valid = key_valid[None, :] & (
                query_offsets[:, None] >= key_offsets[None, :]
            )
            qk = tl.where(valid, qk, -float("inf"))

            # A future key tile can be completely invisible to early query
            # rows. Keep those rows unchanged and avoid exp(-inf - -inf).
            row_has_valid = tl.sum(valid, axis=1) > 0
            m_block = tl.max(qk, axis=1)
            m_new = tl.where(
                row_has_valid,
                tl.maximum(m_i, m_block),
                m_i,
            )
            safe_m_i = tl.where(row_has_valid, m_i, 0.0)
            safe_m_new = tl.where(row_has_valid, m_new, 0.0)
            alpha = tl.where(
                row_has_valid,
                tl.exp(safe_m_i - safe_m_new),
                1.0,
            )
            p = tl.where(
                valid,
                tl.exp(qk - safe_m_new[:, None]),
                0.0,
            )
        else:
            qk = tl.where(key_valid[None, :], qk, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = alpha[:, None] * acc + tl.dot(p, v)
        m_i = m_new

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    acc = acc / l_i[:, None]
    l_i = m_i + tl.log(l_i)

    tl.store(output_block_ptr, acc, boundary_check=(0, 1))
    tl.store(L_block_ptr, l_i, boundary_check=(0,))


@torch.compile(fullgraph=True)
def _flash_backward_torch(Q, K, V, O, dO, L, is_causal):
    """Reference FlashAttention backward using recomputation."""
    n_queries = Q.shape[-2]
    n_keys = K.shape[-2]
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    scores = torch.einsum("bqd,bkd->bqk", Q, K) * scale
    if is_causal:
        q_positions = torch.arange(n_queries, device=Q.device)[:, None]
        k_positions = torch.arange(n_keys, device=Q.device)[None, :]
        scores = scores.masked_fill(q_positions < k_positions, -1e6)

    # Recompute P from the saved log-sum-exp instead of saving P in forward.
    P = torch.exp(scores - L[..., None])

    dV = torch.einsum("bqk,bqd->bkd", P, dO)
    dP = torch.einsum("bqd,bkd->bqk", dO, V)

    # D_i = rowsum(O_i * dO_i), used by the softmax backward formula.
    D = torch.sum(O * dO, dim=-1)
    dS = P * (dP - D[..., None])

    dQ = torch.einsum("bqk,bkd->bqd", dS, K) * scale
    dK = torch.einsum("bqk,bqd->bkd", dS, Q) * scale
    return dQ, dK, dV


class FlashAttentionPytorchFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        d = Q.shape[-1]
        scores = torch.einsum("bqd,bkd->bqk", Q, K) / math.sqrt(d)

        if is_causal:
            q_positions = torch.arange(Q.shape[-2], device=Q.device)[:, None]
            k_positions = torch.arange(K.shape[-2], device=K.device)[None, :]
            scores = scores.masked_fill(q_positions < k_positions, -1e6)

        L = torch.logsumexp(scores, dim=-1)
        P = torch.exp(scores - L[..., None])
        O = torch.einsum("bqk,bkd->bqd", P, V)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = _flash_backward_torch(
            Q, K, V, O, dO, L, ctx.is_causal
        )
        return dQ, dK, dV, None


class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
            raise ValueError("Q, K, and V must have shape [batch, sequence, D]")

        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[1]
        if K.shape != (batch_size, n_keys, d):
            raise ValueError("K must have shape [batch, n_keys, D]")
        if V.shape != (batch_size, n_keys, d):
            raise ValueError("V must have shape [batch, n_keys, D]")
        if not Q.is_cuda or not K.is_cuda or not V.is_cuda:
            raise ValueError("FlashAttention Triton requires CUDA tensors")

        q_tile_size = 64
        k_tile_size = 64
        output = torch.empty_like(Q)
        lse = torch.empty(
            (batch_size, n_queries),
            device=Q.device,
            dtype=torch.float32,
        )

        grid = (triton.cdiv(n_queries, q_tile_size), batch_size)
        flash_fwd_kernel[grid](
            Q, K, V,
            output, lse,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            lse.stride(0), lse.stride(1),
            n_queries, n_keys,
            1.0 / math.sqrt(d),
            D=d,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            IS_CAUSAL=is_causal,
        )

        ctx.save_for_backward(Q, K, V, output, lse)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = _flash_backward_torch(
            Q, K, V, O, dO, L, ctx.is_causal
        )
        return dQ, dK, dV, None
