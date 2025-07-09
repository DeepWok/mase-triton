"""
Optical Transformers: https://arxiv.org/abs/2302.10360
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from ....about import PACKAGE_NAME
from ....dtype import TORCH_DTYPE_TO_TRITON
from .utils import _noisy_quantize


def _get_autotune_configs_ot_qlinear_forward_kernel():
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
def _ot_qlinear_forward_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
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
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_d,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    ENABLE_LUT_MIN: tl.constexpr,
    SKIP_QUANTIZE: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

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
        acc = tl.dot(a, b, acc)

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
    if USE_BIAS:
        d_ptrs = d_ptr + (offs_bn * stride_d)
        bias = tl.load(d_ptrs, mask=offs_bn < N, other=0.0)
        c = c + bias.to(INPUT_DTYPE)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@torch.library.custom_op(
    f"{PACKAGE_NAME}::optical_transformer_quantized_linear_fn",
    mutates_args={},
)
def ot_qlinear_fn(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    x_min: float,
    x_max: float,
    w_min: float,
    w_max: float,
    w_lut_min: float | None,
    o_min: float,
    o_max: float,
    q_levels: int,
    q_seed: int,
    skip_quantize: bool = False,
) -> tuple[Tensor, int]:
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32), (
        f"Unsupported dtype {x.dtype}"
    )
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert weight.ndim == 2, f"Weight tensor must be 2D, got {weight.ndim}D"
    assert weight.dtype in (torch.bfloat16, torch.float16, torch.float32), (
        f"Unsupported dtype {weight.dtype}"
    )

    ori_x_shape = x.size()
    x = x.reshape(-1, ori_x_shape[-1])

    if bias is not None:
        assert bias.shape[0] == weight.shape[0], (
            f"Bias shape {bias.shape} does not match weight shape {weight.shape}"
        )

    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2, f"Input shape {x.shape} does not match weight shape {weight.shape}"

    if bias is None:
        output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    else:
        output = torch.tile(bias, (M, 1)).to(x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )

    with torch.cuda.device(x.device.index):
        _ot_qlinear_forward_kernel[grid](
            x,
            weight.T,
            output,
            bias if bias is not None else x,
            M=M,
            N=N,
            K=K,
            a_min=x_min,
            a_max=x_max,
            b_min=w_min,
            b_max=w_max,
            b_lut_min=w_lut_min or 0.0,
            c_min=o_min,
            c_max=o_max,
            quant_levels=q_levels,
            seed=q_seed,
            stride_am=x.stride(0),
            stride_ak=x.stride(1),
            stride_bk=weight.T.stride(0),
            stride_bn=weight.T.stride(1),
            stride_cm=output.stride(0),
            stride_cn=output.stride(1),
            stride_d=bias.stride(0) if bias is not None else 0,
            ENABLE_LUT_MIN=w_lut_min is not None,
            INPUT_DTYPE=TORCH_DTYPE_TO_TRITON[x.dtype],
            SKIP_QUANTIZE=skip_quantize,
            USE_BIAS=bias is not None,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=64,
            GROUP_SIZE_M=8,
        )

    output = output.reshape(ori_x_shape[:-1] + (N,))
    q_seed = q_seed + 1 if skip_quantize else q_seed
    return output, q_seed


@ot_qlinear_fn.register_fake
def _ot_qlinear_fn_fake(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    x_min: float,
    x_max: float,
    w_min: float,
    w_max: float,
    w_lut_min: float | None,
    o_min: float,
    o_max: float,
    q_levels: int,
    q_seed: int,
    skip_quantize: bool = False,
) -> tuple[Tensor, int]:
    output = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=x.dtype)
    return output, q_seed


@ot_qlinear_fn.register_kernel("cpu")
def _ot_qlinear_fn_cpu(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    x_min: float,
    x_max: float,
    w_min: float,
    w_max: float,
    w_lut_min: float | None,
    o_min: float,
    o_max: float,
    q_levels: int,
    q_seed: int,
    skip_quantize: bool = False,
) -> tuple[Tensor, int]:
    raise NotImplementedError("CPU kernel is not implemented")


def _ot_qlinear_backward(ctx, *grad_outputs):
    x, weight, bias = ctx.saved_tensors
    grad_x = grad_weight = grad_bias = None

    if ctx.needs_input_grad[0]:
        grad_x = grad_outputs[0] @ weight
    if ctx.needs_input_grad[1]:
        grad_weight = grad_outputs[0].transpose(-1, -2) @ x
    if bias is not None and ctx.needs_input_grad[2]:
        grad_bias = grad_outputs[0].sum(0)

    return (
        grad_x,
        grad_weight,
        grad_bias,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _ot_qlinear_setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1], inputs[2])


ot_qlinear_fn.register_autograd(
    _ot_qlinear_backward,
    setup_context=_ot_qlinear_setup_context,
)


__all__ = [
    "ot_qlinear_fn",
    "_ot_qlinear_forward_kernel",
]
