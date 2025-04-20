import os
# os.environ["TRITON_INTERPRET"] = "1"

import torch
from torch import Tensor
import triton
import triton.language as tl
import pdb

from ....dtype import TORCH_DTYPE_TO_TRITON

from .quantize import _input_quantize_fn, _weight_quantize_fn


def _get_autotune_configs_morr_forward_kernel():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 1,
                "BLOCK_SIZE_P": 1,
                "BLOCK_SIZE_Q": 1,
                "BLOCK_SIZE_K1": 4,
                "BLOCK_SIZE_K2": 1,
            },
            num_stages=3,
            num_warps=8,
        ),
    ]


@triton.jit
def _toeplitz(
    x: tl.tensor,                 # Input block, shape (BLOCK_SIZE_K1,)
    BLOCK_SIZE_K1: tl.constexpr,
):
    """
    Creates a Toeplitz matrix block from an input vector block x.
    Assumes x is available for gathering.

    Args:
        x: tl.tensor of shape (BLOCK_SIZE_K1,)
        BLOCK_SIZE_K1: The dimension size.

    Returns:
        tl.tensor: A block of shape (BLOCK_SIZE_K1, BLOCK_SIZE_K1)
                   representing the Toeplitz matrix.
    """

    row_indices = tl.arange(0, BLOCK_SIZE_K1)[:, None]
    col_indices = tl.arange(0, BLOCK_SIZE_K1)[None, :]
    target_indices_in_x = (col_indices - row_indices + BLOCK_SIZE_K1) % BLOCK_SIZE_K1

    result = tl.zeros((BLOCK_SIZE_K1, BLOCK_SIZE_K1), dtype=x.dtype)
    for k in range(BLOCK_SIZE_K1):
         mask = (target_indices_in_x == k)
         val_k_block = tl.sum(x * (tl.arange(0, BLOCK_SIZE_K1) == k)) # Reduces x to scalar, broadcasts back
         result = tl.where(mask, val_k_block, result)

    return result 

@triton.jit
def _mrr_roundtrip_phase_to_tr_func(
    x: tl.tensor,
    a: tl.constexpr = 0.8,
    r: tl.constexpr = 0.9,
    intensity: tl.constexpr = False,
):
    """
    Applies a round-trip phase correction to the input tensor.
    """
    c1 = -2.0 * a * r
    c2 = a * a + r * r
    c3 = 1.0 + r * r * a * a - a * a - r * r

    cos_x = tl.cos(x)
    numerator = cos_x * c1 + c2
    denominator = numerator + c3
    x = numerator / denominator
    if not intensity:
        x = tl.sqrt(x)
    return x


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 1,
                "BLOCK_SIZE_P": 1,
                "BLOCK_SIZE_Q": 1,
                "BLOCK_SIZE_K1": 4,
                "BLOCK_SIZE_K2": 1,
            },
            num_stages=3,
            num_warps=8,
        ),
    ],
    key=["M", "P", "Q", "K"],
)
@triton.jit
def morr_propagate_kernel(
    x_ptr,
    w_ptr,
    o_ptr,
    b_ptr,
    grid_dim_q,
    grid_dim_p,
    miniblock,
    crosstalk_factor,
    use_bias,
    phase_noise_std,
    trainable_morr_bias,
    morr_bias,
    mrr_a,
    mrr_r,
    in_bit,
    w_bit,
    # stride
    stride_wm,
    stride_wp,
    stride_wq,
    stride_wk1,
    stride_wk2,
    stride_xm,
    stride_xp,
    stride_xq,
    stride_xk1,
    stride_xk2,
    stride_om,
    stride_op,
    stride_oq,
    stride_ok1,
    stride_ok2,
    stride_d,
    finegrain_drop_mask,
    ENABLE_PHASE_NOISE: tl.constexpr,
    ENABLE_THERMAL_CROSSTALK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,
    BLOCK_SIZE_K2: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):

    # Program ID for block-based processing
    pid_m = tl.program_id(axis=0)
    pid_p = tl.program_id(axis=1)
    pid_q = tl.program_id(axis=2)
    # each program is assigned a [1, 1, miniblock, 1] block
    num_pid_q = grid_dim_q
    num_pid_p = grid_dim_p
    # 2d coordinates in p, q dimension
    # pid_p = pid // num_pid_q
    # pid_q = pid % num_pid_q
    
    offs_wm = tl.arange(0, BLOCK_SIZE_M)
    offs_wp = pid_p * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
    offs_wq = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_wk1 = tl.arange(0, BLOCK_SIZE_K1)
    offs_wk2 = tl.arange(0, BLOCK_SIZE_K2)

    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_xp = tl.arange(0, BLOCK_SIZE_P)
    offs_xq = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_xk1 = tl.arange(0, BLOCK_SIZE_K1)
    offs_xk2 = tl.arange(0, BLOCK_SIZE_K2)

    # TODO: Create masks for valid indices
    mask_w = None
    mask_x = None

    w_ptrs = w_ptr + (
        offs_wm[:, None, None, None, None] * stride_wm
        + offs_wp[None, :, None, None, None] * stride_wp
        + offs_wq[None, None, :, None, None] * stride_wq
        + offs_wk1[None, None, None, :, None] * stride_wk1
        + offs_wk2[None, None, None, None, :] * stride_wk2
    )
    x_ptrs = x_ptr + (
        offs_xm[:, None, None, None, None] * stride_xm
        + offs_xp[None, :, None, None, None] * stride_xp
        + offs_xq[None, None, :, None, None] * stride_xq
        + offs_xk1[None, None, None, :, None] * stride_xk1
        + offs_xk2[None, None, None, None, :] * stride_xk2
    )
    w = tl.load(w_ptrs)
    x = tl.load(x_ptrs)
    
    # TODO: Test Quantization Function
    # if in_bit < 16:
    #     x = _input_quantize_fn(x)

    ## build_weight()
    # TODO: add morr_output_scale, fix quantization func
    # if w_bit < 16:
    #     w = _weight_quantize_fn(w)
    # else:
    #     w = tl.abs(w)

    w = tl.abs(w).reshape(BLOCK_SIZE_K1, BLOCK_SIZE_K2) # [1, 1, 1, k, 1] -> [k, 1]
    x = x.reshape(BLOCK_SIZE_K1, BLOCK_SIZE_K2) # [1, 1, 1, k, 1] -> [k, 1]
    
    if finegrain_drop_mask is not None:
        w *= tl.cast(finegrain_drop_mask, tl.float32)

    x = x * x  # input_modulator()

    ### propagate_morr()
    # apply thermal crosstalk noise
    if ENABLE_THERMAL_CROSSTALK:
        w = w * crosstalk_factor
    # create toeplitz matrix

    w = _toeplitz(w, BLOCK_SIZE_K1)  # [k, k]
    
    # apply phase noise
    if ENABLE_PHASE_NOISE:
        noise = tl.zeros_like(x) + tl.randn(x.shape) * phase_noise_std
        x = x + noise
    # add trainable bias
    if trainable_morr_bias:
        x = x - morr_bias
    # mrr_roundtrip_phase_to_tr
    x = _mrr_roundtrip_phase_to_tr_func(x, mrr_a, mrr_r, intensity=True)

    # TODO: this is temporary, as tl.dot requires 16*16 matrix at least
    acc = tl.zeros((BLOCK_SIZE_K1, BLOCK_SIZE_K2), dtype=tl.float32)
    x_transposed = tl.trans(x)
    x_broadcasted = tl.broadcast_to(x_transposed, (BLOCK_SIZE_K1, BLOCK_SIZE_K1))
    elementwise_prod = w * x_broadcasted
    acc_sum = tl.sum(elementwise_prod, axis=1)
    acc = tl.reshape(acc_sum, (BLOCK_SIZE_K1, BLOCK_SIZE_K2))
    # acc = tl.dot(w, x, acc)

    out = acc.to(INPUT_DTYPE)
    out = out.reshape(BLOCK_SIZE_M, BLOCK_SIZE_P, BLOCK_SIZE_Q, BLOCK_SIZE_K1) # [k, 1] -> [1, 1, 1, k]

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_op = pid_p * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
    offs_oq = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_ok1 = tl.arange(0, BLOCK_SIZE_K1)
    # offs_ok2 = tl.arange(0, BLOCK_SIZE_K2)
    o_ptrs = o_ptr + (
        stride_om * offs_om[:, None, None, None]
        + stride_op * offs_op[None, :, None, None]
        + stride_oq * offs_oq[None, None, :, None]
        + stride_ok1 * offs_ok1[None, None, None, :]
    )
    o_mask = None
    tl.store(o_ptrs, out, mask=o_mask)


def morr_linear_fn(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    grid_dim_x: int,
    grid_dim_y: int,
    miniblock: int,
    enable_thermal_crosstalk: bool,
    crosstalk_factor: float,
    enable_phase_noise: bool,
    phase_noise_std: float,
    trainable_morr_bias: bool,
    mrr_a: float,
    mrr_r: float,
    finegrain_drop_mask: Tensor | None,
    morr_output_scale: Tensor,
    in_features: int,
    in_features_pad: int,
    out_features: int,
    out_features_pad: int,
    in_bit: int,
    w_bit: int,
) -> Tensor:

    assert x.dtype in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ), f"Unsupported dtype {x.dtype}"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert weight.dtype in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ), f"Unsupported dtype {weight.dtype}"

    # Handle transformer vs non-transformer inputs
    ori_x_shape = x.shape
    is_transformer = len(ori_x_shape) == 3

    if is_transformer:
        B, N_seq, D = x.shape
        M = B * N_seq
        x = x.reshape(M, D)
    else:
        M = x.shape[0]

    # Get dimensions
    M, D = x.shape
    P, Q, K = weight.shape

    assert D == Q * K == in_features, "input and weight dimension mismatch"
    assert Q * K == out_features, "weight and output dimension mismatch"
    
    # get padding dimensions, pad input if needed
    Din_padded = in_features_pad
    Dout_padded = out_features_pad

    if Din_padded > D:
        x_pad = torch.zeros(M, Din_padded - K, device=x.device, dtype=x.dtype)
        x = torch.cat([x, x_pad], dim=1)

    # Reshape x and weight
    x = x.view(-1, grid_dim_x, miniblock)  # [M, q, k]
    x = x.unsqueeze(1).unsqueeze(-1) # [M, 1, q, k, 1]
    weight = weight.unsqueeze(0).unsqueeze(-1) # [p, q, k] -> [1, p, q, k, 1]

    # Allocate output
    output = torch.empty((M, P, Q, K, 1), device=x.device, dtype=x.dtype)

    # Launch the Triton kernel
    grid = lambda meta: (
        M,
        P,
        Q,
    )
    morr_propagate_kernel[grid](
        x_ptr = x,
        w_ptr = weight,
        o_ptr = output,
        b_ptr = bias if bias is not None else x,
        grid_dim_q=grid_dim_x,
        grid_dim_p=grid_dim_y,
        miniblock=miniblock,
        crosstalk_factor=crosstalk_factor,
        use_bias=False,
        phase_noise_std=phase_noise_std,
        trainable_morr_bias=trainable_morr_bias,
        morr_bias=None,
        mrr_a=mrr_a,
        mrr_r=mrr_r,
        in_bit=in_bit,
        w_bit=w_bit,
        finegrain_drop_mask=finegrain_drop_mask,
        stride_wm=weight.stride(0),
        stride_wp=weight.stride(1),
        stride_wq=weight.stride(2),
        stride_wk1=weight.stride(3),
        stride_wk2=weight.stride(4),
        stride_xm=x.stride(0),
        stride_xp=x.stride(1),
        stride_xq=x.stride(2),
        stride_xk1=x.stride(3),
        stride_xk2=x.stride(4),
        stride_om=output.stride(0),
        stride_op=output.stride(1),
        stride_oq=output.stride(2),
        stride_ok1=output.stride(3),
        stride_ok2=output.stride(4),
        stride_d=bias.stride(0) if bias is not None else 0,
        ENABLE_THERMAL_CROSSTALK=enable_thermal_crosstalk,
        ENABLE_PHASE_NOISE=enable_phase_noise and phase_noise_std > 1,
        INPUT_DTYPE=TORCH_DTYPE_TO_TRITON[x.dtype],
    )

    # morr_output_scale in build_weight()
    if w_bit < 16:
        morr_output_scale = _weight_quantize_fn(morr_output_scale)
    else:
        morr_output_scale = morr_output_scale - morr_output_scale.data.mean()
    ## differential balancing factor concatenation
    scale = morr_output_scale[..., :-1, :]
    scale_pad = morr_output_scale[..., -1:, :]
    if grid_dim_x % 2 == 0:
        # even blocks
        scale = torch.cat([scale, -scale], dim=2)  # [1, 1, q, 1]
    else:
        # odd blocks
        if grid_dim_x > 1:
            scale = torch.cat([morr_output_scale, -scale], dim=2)  # [1, 1, q, 1]
        else:
            scale = scale_pad  # [1, 1, q, 1]
    morr_output_scale = scale.squeeze(-1).unsqueeze(0)  # [1 ,1, 1, q]

    # Apply output scale
    output = output.squeeze(-1)  # [m, p, q, k, 1] -> [m, p, q, k]
    output = morr_output_scale.matmul(output)  # [1, 1, 1, q] x [bs, p, q, k] = [bs, p, 1, k]
    output.flatten(1)

    # Trim output if needed
    if out_features < out_features_pad:
        output = output[:, :out_features]

    # Reshape back for transformer
    if is_transformer:
        output = output.view(B, N_seq, out_features)

    return output
