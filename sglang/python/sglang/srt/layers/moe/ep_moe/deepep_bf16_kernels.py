"""Fused Triton kernels for DeepEP BF16 low-latency MoE decode.

Replaces the naive activation + masking pipeline (5+ CUDA kernels for silu+mul
and arange+comparison+masked_fill+copy) with a single Triton elementwise kernel,
while keeping cuBLAS batched GEMM for the matrix multiplies.

Pipeline: bmm → fused_act_mul_masked (in-place) → bmm(out=hidden)
  (3 ops total: 2 cuBLAS + 1 Triton, vs original 7-8 separate CUDA kernels)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_masked_kernel(
    gate_up_ptr,
    masked_m_ptr,
    M,
    N,
    stride_ge,
    stride_gm,
    stride_gn,
    BLOCK: tl.constexpr,
):
    """Fused SiLU(gate) * up with per-expert masking, written in-place.

    gate_up: [E, M, 2*N] — first N cols are gate, last N cols are up.
    Writes SiLU(gate)*up to gate_up[:,:,:N] in-place.
    Rows m >= masked_m[e] are zeroed.
    """
    expert_id = tl.program_id(1)
    pid = tl.program_id(0)

    expert_valid_m = tl.load(masked_m_ptr + expert_id)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = M * N
    mask = offs < total

    m = offs // N
    n = offs % N

    gate_base = gate_up_ptr + expert_id * stride_ge

    gate_val = tl.load(gate_base + m * stride_gm + n * stride_gn, mask=mask, other=0.0)
    up_val = tl.load(
        gate_base + m * stride_gm + (n + N) * stride_gn, mask=mask, other=0.0
    )

    gate_f32 = gate_val.to(tl.float32)
    result = (gate_f32 * tl.sigmoid(gate_f32)) * up_val.to(tl.float32)

    # Zero invalid rows
    valid = m < expert_valid_m
    result = tl.where(valid, result, 0.0)

    tl.store(
        gate_base + m * stride_gm + n * stride_gn,
        result.to(gate_up_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _gelu_mul_masked_kernel(
    gate_up_ptr,
    masked_m_ptr,
    M,
    N,
    stride_ge,
    stride_gm,
    stride_gn,
    BLOCK: tl.constexpr,
):
    """Fused GELU(gate) * up with per-expert masking, written in-place."""
    expert_id = tl.program_id(1)
    pid = tl.program_id(0)

    expert_valid_m = tl.load(masked_m_ptr + expert_id)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = M * N
    mask = offs < total

    m = offs // N
    n = offs % N

    gate_base = gate_up_ptr + expert_id * stride_ge

    gate_val = tl.load(gate_base + m * stride_gm + n * stride_gn, mask=mask, other=0.0)
    up_val = tl.load(
        gate_base + m * stride_gm + (n + N) * stride_gn, mask=mask, other=0.0
    )

    g = gate_val.to(tl.float32)
    kAlpha = 0.7978845608028654
    gate_act = 0.5 * g * (1.0 + tl.math.tanh(kAlpha * (g + 0.044715 * g * g * g)))
    result = gate_act * up_val.to(tl.float32)

    valid = m < expert_valid_m
    result = tl.where(valid, result, 0.0)

    tl.store(
        gate_base + m * stride_gm + n * stride_gn,
        result.to(gate_up_ptr.dtype.element_ty),
        mask=mask,
    )


def fused_act_mul_masked_inplace(
    gate_up: torch.Tensor,
    intermediate_size: int,
    masked_m: torch.Tensor,
    use_gelu: bool = False,
) -> None:
    """Fused activation + multiply + masking, written in-place to gate_up[:,:,:I].

    After this call, gate_up[:, :, :intermediate_size] contains the masked
    activated intermediate, suitable for the down projection GEMM.

    Args:
        gate_up: [E, M, 2*I] output of bmm(tokens, w13.T), modified in-place
        intermediate_size: I
        masked_m: [E] per-expert valid token count
        use_gelu: use GELU instead of SiLU
    """
    E, M, _ = gate_up.shape
    N = intermediate_size

    total = M * N
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK), E)

    kernel = _gelu_mul_masked_kernel if use_gelu else _silu_mul_masked_kernel
    kernel[grid](
        gate_up,
        masked_m,
        M,
        N,
        gate_up.stride(0),
        gate_up.stride(1),
        gate_up.stride(2),
        BLOCK=BLOCK,
    )
