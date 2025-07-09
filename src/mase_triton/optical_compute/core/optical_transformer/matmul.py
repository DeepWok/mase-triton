import torch
import triton
import triton.language as tl
from torch import Tensor

from ....about import PACKAGE_NAME
from ....dtype import TORCH_DTYPE_TO_TRITON
from .utils import _noisy_quantize


def _get_autotune_configs_ot_qmatmul_kernel():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
    ]


# *: Transformers & Accelerate DDP does not work with this Triton kernel
@triton.jit
def _ot_qmatmul_forward_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    B,
    M,
    N,
    K,
    a_min,
    a_max,
    b_min,
    b_max,
    b_lut_min,
    c_min,
    c_max,
    quant_levels,
    seed,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    ENABLE_LUT_MIN: tl.constexpr,
    SKIP_QUANTIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_batch = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (
        offs_batch * stride_ab
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        offs_batch * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        if not SKIP_QUANTIZE:
            a = _noisy_quantize(
                a,
                quant_levels=quant_levels,
                min_val=a_min,
                max_val=a_max,
                lut_min=0.0,
                seed=seed,
                INPUT_DTYPE=INPUT_DTYPE,
                QUANT_MODE="det",
                ENABLE_LUT_MIN=False,
            )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        if not SKIP_QUANTIZE:
            b = _noisy_quantize(
                b,
                quant_levels=quant_levels,
                min_val=b_min,
                max_val=b_max,
                lut_min=b_lut_min,
                seed=seed,
                INPUT_DTYPE=INPUT_DTYPE,
                QUANT_MODE="det",
                ENABLE_LUT_MIN=ENABLE_LUT_MIN,
            )

        acc = tl.dot(a, b, acc=acc)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = acc.to(INPUT_DTYPE)
    if not SKIP_QUANTIZE:
        c = _noisy_quantize(
            c,
            quant_levels=quant_levels,
            min_val=c_min,
            max_val=c_max,
            lut_min=0.0,
            seed=seed,
            INPUT_DTYPE=INPUT_DTYPE,
            QUANT_MODE="rand",
            ENABLE_LUT_MIN=False,
        )

    c_ptrs = c_ptr + (
        offs_batch * stride_cb
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )
    c_mask = (offs_batch < B) & (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@torch.library.custom_op(
    f"{PACKAGE_NAME}::optical_transformer_quantized_matmul_fn",
    mutates_args={},
)
def ot_qmatmul_fn(
    a: Tensor,
    b: Tensor,
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
    b_lut_min: float,
    o_min: float,
    o_max: float,
    q_levels: int,
    q_seed: int,
    skip_quantize: bool = False,
) -> tuple[Tensor, int]:
    assert a.is_contiguous(), "a must be contiguous"
    assert b.is_contiguous(), "b must be contiguous"
    assert a.ndim >= 2, "a must have at least 2 dimensions"
    assert b.ndim >= 2, "b must have at least 2 dimensions"
    assert a.ndim == b.ndim, "a and b must have the same number of dimensions"

    orig_a_shape = a.size()

    M, K = a.shape[-2:]
    K2, N = b.shape[-2:]
    assert K == K2, "K dimension must match"

    a = a.reshape(-1, M, K)
    b = b.reshape(-1, K, N)
    B = a.shape[0]

    output = torch.empty((B, M, N), dtype=a.dtype, device=a.device)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
            B,
        )

    with torch.cuda.device(a.device.index):
        _ot_qmatmul_forward_kernel[grid](
            a,
            b,
            output,
            B=B,
            M=M,
            N=N,
            K=K,
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
            b_lut_min=b_lut_min,
            c_min=o_min,
            c_max=o_max,
            quant_levels=q_levels,
            seed=q_seed,
            stride_ab=a.stride(0),
            stride_am=a.stride(1),
            stride_ak=a.stride(2),
            stride_bb=b.stride(0),
            stride_bk=b.stride(1),
            stride_bn=b.stride(2),
            stride_cb=output.stride(0),
            stride_cm=output.stride(1),
            stride_cn=output.stride(2),
            INPUT_DTYPE=TORCH_DTYPE_TO_TRITON[a.dtype],
            ENABLE_LUT_MIN=b_lut_min is not None,
            SKIP_QUANTIZE=skip_quantize,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=64,
            GROUP_SIZE_M=8,
        )
    output = output.reshape(orig_a_shape[:-2] + (M, N))
    q_seed += 1 if not skip_quantize else q_seed
    return output, q_seed


@ot_qmatmul_fn.register_fake
def _ot_qmatmul_fn_fake(
    a: Tensor,
    b: Tensor,
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
    b_lut_min: float,
    o_min: float,
    o_max: float,
    q_levels: int,
    q_seed: int,
    skip_quantize: bool = False,
) -> tuple[Tensor, int]:
    output = torch.empty((*a.shape[:-2], b.shape[-1]), device=a.device, dtype=a.dtype)
    return output, q_seed


@ot_qmatmul_fn.register_kernel("cpu")
def _ot_qmatmul_fn_cpu(
    a: Tensor,
    b: Tensor,
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
    b_lut_min: float,
    o_min: float,
    o_max: float,
    q_levels: int,
    q_seed: int,
    skip_quantize: bool = False,
) -> tuple[Tensor, int]:
    raise NotImplementedError("CPU kernel is not implemented")


def _ot_qmatmul_backward(ctx, *grad_outputs):
    a, b = ctx.saved_tensors
    grad_a = grad_b = None
    if ctx.needs_input_grad[0]:
        grad_a = grad_outputs[0] @ b.transpose(-1, -2)
    if ctx.needs_input_grad[1]:
        grad_b = a.transpose(-1, -2) @ grad_outputs[0]
    return grad_a, grad_b, None, None, None, None, None, None, None, None, None, None


def _ot_qmatmul_setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1])


ot_qmatmul_fn.register_autograd(
    _ot_qmatmul_backward,
    setup_context=_ot_qmatmul_setup_context,
)


__all__ = [
    "ot_qmatmul_fn",
    "_ot_qmatmul_forward_kernel",
]
