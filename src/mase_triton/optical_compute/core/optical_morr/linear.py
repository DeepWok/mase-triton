import os
# os.environ["TRITON_INTERPRET"] = "1"

import torch
from torch import Tensor
import triton
import triton.language as tl
import pdb

from ....dtype import TORCH_DTYPE_TO_TRITON
from ....about import PACKAGE_NAME
from .optical_original.utils import toeplitz
from .quantize import _input_quantize_fn, _weight_quantize_fn


# def _get_autotune_configs_morr_forward_kernel():
#     return [
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 1,
#                 "BLOCK_SIZE_P": 1,
#                 "BLOCK_SIZE_Q": 1,
#                 "BLOCK_SIZE_K1": 4,
#                 "BLOCK_SIZE_K2": 1,
#             },
#             num_stages=3,
#             num_warps=8,
#         ),
#     ]


# @triton.jit
# def _toeplitz(
#     x: tl.tensor,                 # Input block, shape (BLOCK_SIZE_K1,)
#     BLOCK_SIZE_K1: tl.constexpr,
# ):
#     """
#     Creates a Toeplitz matrix block from an input vector block x.
#     """

#     row_indices = tl.arange(0, BLOCK_SIZE_K1)[:, None]
#     col_indices = tl.arange(0, BLOCK_SIZE_K1)[None, :]
#     target_indices_in_x = (col_indices - row_indices + BLOCK_SIZE_K1) % BLOCK_SIZE_K1

#     result = tl.zeros((BLOCK_SIZE_K1, BLOCK_SIZE_K1), dtype=x.dtype)
#     for k in range(BLOCK_SIZE_K1):
#          mask = (target_indices_in_x == k)
#          val_k_block = tl.sum(x * (tl.arange(0, BLOCK_SIZE_K1) == k)) # Reduces x to scalar, broadcasts back
#          result = tl.where(mask, val_k_block, result)

#     return result 

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
    # each program is assigned GROUP_SIZE_MPQ * [1, 1, miniblock, 1] block
    pid = tl.program_id(axis=0)
    pid_m = pid // (grid_dim_q * grid_dim_p)
    pid_p = (pid // grid_dim_q) % grid_dim_p
    pid_q = pid % grid_dim_q

    # starting element's m, p, q coordinates in the global tensor
    start_m = pid_m * BLOCK_SIZE_M
    start_p = pid_p * BLOCK_SIZE_P
    start_q = pid_q * BLOCK_SIZE_Q
    
    # w [1, p, q, k, 1]
    offs_wm = tl.arange(0, 1)
    offs_wp = pid_p * BLOCK_SIZE_P + tl.arange(0, 1)
    offs_wq = pid_q * BLOCK_SIZE_Q + tl.arange(0, 1)
    offs_wk1 = tl.arange(0, BLOCK_SIZE_K1)
    offs_wk2 = tl.arange(0, BLOCK_SIZE_K1)

    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, 1)
    offs_xp = tl.arange(0, 1)
    offs_xq = pid_q * BLOCK_SIZE_Q + tl.arange(0, 1)
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

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_P, BLOCK_SIZE_Q, BLOCK_SIZE_K1, BLOCK_SIZE_K2), dtype=tl.float32)
    m_indices = tl.arange(0, BLOCK_SIZE_M)[:, None, None, None, None]
    p_indices = tl.arange(0, BLOCK_SIZE_P)[None, :, None, None, None]
    q_indices = tl.arange(0, BLOCK_SIZE_Q)[None, None, :, None, None]

    for m_local in range(BLOCK_SIZE_M):
        m = start_m + m_local
        for p_local in range(BLOCK_SIZE_P):
            p = start_p + p_local
            for q_local in range(BLOCK_SIZE_Q):
                q = start_q + q_local
                
                w = tl.load(w_ptrs)
                x = tl.load(x_ptrs)

                # TODO: Test Quantization Function
                # if in_bit < 16:
                #     x = _input_quantize_fn(x)

                # ----- build_weight() -----
                # TODO: add morr_output_scale, fix quantization func
                # if w_bit < 16:
                #     w = _weight_quantize_fn(w)
                # else:
                #     w = tl.abs(w)

                w = tl.abs(w).reshape(BLOCK_SIZE_K1, BLOCK_SIZE_K1) # [1, 1, 1, k, k] -> [k, k]
                x = x.reshape(BLOCK_SIZE_K1, BLOCK_SIZE_K2) # [1, 1, 1, k, 1] -> [k, 1]
                
                if finegrain_drop_mask is not None:
                    w *= tl.cast(finegrain_drop_mask, tl.float32)
                
                x = x * x  # input_modulator()
                # ----- propagate_morr() -----

                # apply thermal crosstalk noise
                if ENABLE_THERMAL_CROSSTALK:
                    w = w * crosstalk_factor
                
                # MatMals
                # TODO: this is temporary, as tl.dot requires 16*16 matrix at least
                x = tl.trans(x)
                x = tl.broadcast_to(x, (BLOCK_SIZE_K1, BLOCK_SIZE_K1))
                x = w * x
                x = tl.sum(x, axis=1)
                x = tl.reshape(x, (BLOCK_SIZE_K1, BLOCK_SIZE_K2))
                # acc = tl.dot(w, x, acc)

                # apply phase noise
                if ENABLE_PHASE_NOISE:
                    noise = tl.zeros_like(x) + tl.randn(x.shape) * phase_noise_std
                    x = x + noise

                # add trainable bias
                if trainable_morr_bias:
                    x = x - morr_bias
                
                # mrr_roundtrip_phase_to_tr
                x = _mrr_roundtrip_phase_to_tr_func(x, mrr_a, mrr_r, intensity=True)

                # store the value in acc using mask
                res = x
                condition_mask = (m_indices == m_local) & (p_indices == p_local) & (q_indices == q_local)
                res = res[None, None, None, :, :]
                acc = tl.where(condition_mask, res, acc)

                # propagate pointer along Q dimension
                w_ptrs += stride_wq
                x_ptrs += stride_xq
            
            # Q loop end
            # reset pointer along Q dimension
            w_ptrs -= stride_wq * (BLOCK_SIZE_Q + 1)
            x_ptrs -= stride_xq * (BLOCK_SIZE_Q + 1)
            # propagate pointer along P dimension
            w_ptrs += stride_wp
            x_ptrs += stride_xp
        
        # P loop end
        # reset pointer along P dimension
        w_ptrs -= stride_wp * (BLOCK_SIZE_P + 1)
        x_ptrs -= stride_xp * (BLOCK_SIZE_P + 1)
        # propagate pointer along M dimension
        w_ptrs += stride_wp
        x_ptrs += stride_xp


    out = acc.to(INPUT_DTYPE)
    out = out.reshape(BLOCK_SIZE_M, BLOCK_SIZE_P, BLOCK_SIZE_Q, BLOCK_SIZE_K1) # [1, 1, q, k, 1] -> [1, 1, q, k]

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

@torch.library.custom_op(
    f"{PACKAGE_NAME}::optical_morr_linear_linear_fn", mutates_args={},
)
def morr_linear_fn(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    grid_dim_x: int,
    grid_dim_y: int,
    miniblock: int,
    enable_thermal_crosstalk: bool,
    crosstalk_factor: float | None,
    enable_phase_noise: bool,
    phase_noise_std: float | None,
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
) -> tuple[Tensor, Tensor, Tensor, Tensor]:

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
        in_B, in_N, in_D = x.shape
        M = in_B * in_N
        x = x.reshape(M, in_D)
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
    weight = toeplitz(weight).unsqueeze(0) # [p, q, k] -> [1, p, q, k, k]

    x_ctx = x.squeeze(-1).squeeze(1).clone() # [M, q, k]
    w_ctx = weight.clone()
    
    # Allocate output
    output = torch.empty((M, P, Q, K, 1), device=x.device, dtype=x.dtype)

    # Launch the Triton kernel
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(P, meta["BLOCK_SIZE_P"]) * triton.cdiv(Q, meta["BLOCK_SIZE_Q"]),
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

    # ----- build_weight() morr_output_scale part ----- 
    if w_bit < 16:
        morr_output_scale = _weight_quantize_fn(morr_output_scale)
    else:
        morr_output_scale = (morr_output_scale - morr_output_scale.data.mean())
    
    # differential balancing factor concatenation
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
    ctx_morr_output_scale = morr_output_scale.clone()

    # Apply output scale
    output = output.squeeze(-1)  # [m, p, q, k, 1] -> [m, p, q, k]
    output = morr_output_scale.matmul(output)  # [1, 1, 1, q] x [bs, p, q, k] = [bs, p, 1, k]
    output.flatten(1)

    # Trim output if needed
    if out_features < out_features_pad:
        output = output[:, :out_features]

    # Reshape back for transformer
    if is_transformer:
        output = output.view(in_B, in_N, out_features)

    # aux_tensor = (
    #     torch.abs(w_ctx),  # w_morr: weight in propagate_morr matmul
    #     x_ctx,                  # x_modulator: x before x^2
    # )

    return output, torch.abs(w_ctx), x_ctx, ctx_morr_output_scale



def _morr_linear_setup_context(ctx, inputs, output):
    """
    Save for backward only what the backward routine really needs.
    """
    (
        x,                       # 0  Tensor – input
        weight,                  # 1  Tensor – learnable weight
        bias,                    # 2  Tensor | None – bias
        grid_dim_x,              # 3  int
        grid_dim_y,              # 4  int
        miniblock,               # 5  int (== K)
        enable_thermal_crosstalk,# 6  bool
        crosstalk_factor,        # 7  float
        enable_phase_noise,      # 8  bool
        phase_noise_std,         # 9  float
        trainable_morr_bias,     # 10 bool
        mrr_a,                   # 11 float
        mrr_r,                   # 12 float
        finegrain_drop_mask,     # 13 Tensor | None
        _,                       # 14 morr_output_scale
        in_features,             # 15 int
        in_features_pad,         # 16 int
        out_features,            # 17 int
        out_features_pad,        # 18 int
        in_bit,                  # 19 int
        w_bit,                   # 20 int
    ) = inputs

    output, w_morr, x_modulator, morr_output_scale = output
    # (
    #     w_morr, 
    #     x_modulator,
    # ) = aux_tensor

    device, dtype = x.device, x.dtype

    # ----- Tensor meta-data that backward needs -----
    # Shapes
    M = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1]
    P, Q, K = weight.shape
    tensor_shape = (M, P, Q, K)

    # mrr_para: para for mrr_roundtrip_phase_to_tr()
    c1 = -2.0 * mrr_a * mrr_r
    c2 = mrr_a * mrr_a + mrr_r * mrr_r
    c3 = 1.0 + (mrr_r * mrr_r) * (mrr_a * mrr_a) - mrr_a * mrr_a - mrr_r * mrr_r
    c4 = (mrr_a**2.0 - 1.0) * (mrr_r**2.0 - 1.0) * 2.0 * mrr_a * mrr_r                                       
    intensity = True
    mrr_para = (c1, c2, c3, c4, intensity)
    
    # x_morr: x input of matmal in propagate_morr()
    x_morr = x_modulator ** 2 # [m, q, k]
    x_morr = x_morr.unsqueeze(1).unsqueeze(-1) # [m, 1, q, k, 1]

    # x_mrr: x input of mrr_roundtrip_phase_to_tr()
    x_mrr = w_morr.matmul(x_morr).squeeze(-1)
    if enable_phase_noise and phase_noise_std > 1e-5:
        x_mrr = x_mrr + torch.zeros_like(x_mrr).normal_(0, phase_noise_std)
    if trainable_morr_bias:
        x_mrr = x_mrr - morr_bias

    # TODO: complete morr_bias
    morr_bias = torch.zeros(1, device=device, dtype=dtype)


    # 3. stash tensors 
    ctx.save_for_backward(
        x,                        # original input
        weight,                   # original weight
        bias if bias is not None else torch.tensor([], device=device, dtype=dtype),
        morr_output_scale,        # original morr_output_scale
        morr_bias,                # 5
        x_morr,                   # 7
        x_mrr,                    # 8
        w_morr,                   # 10
        x_modulator,              # 11
    )
    ctx.tensor_shape = tensor_shape
    ctx.mrr_para = mrr_para
    ctx.trainable_morr_bias = trainable_morr_bias
    ctx.in_features = in_features 
    ctx.in_features_pad = in_features_pad                     
    ctx.out_features = out_features     
    ctx.out_features_pad = out_features_pad



def _morr_linear_backward(ctx, grad_output, *ignored):
    """
    Backward pass for morr_linear_fn.
    """
    (
        x, weight, bias,
        morr_output_scale,
        morr_bias,
        x_morr,
        x_mrr,
        w_morr,
        x_modulator,
    ) = ctx.saved_tensors

    M, P, Q, K  = ctx.tensor_shape
    c1, c2, c3, c4, intensity = ctx.mrr_para
    in_features = ctx.in_features
    in_features_pad = ctx.in_features_pad
    out_features = ctx.out_features
    out_features_pad = ctx.out_features_pad
    x_input_shape = x.shape
    DEVICE = x.device
    # ----- backward prop -----
    # Reshape
    grad_out = grad_output.view(x.shape[0], weight.shape[1], weight.shape[2], -1)  # [M, P, Q, K]

    # ----- Gradient w.r.t input x -----
    if ctx.needs_input_grad[0]:
        # 0. gradient from output
        
        # 1. reshape
        grad_out = grad_out.view(M, -1) # [m, out_features]

        if ctx.needs_input_grad[4] and bias:
            grad_bias = grad_out.sum(dim=0) # [out_features]
        else:
            grad_bias = None

        out_pad = torch.zeros(grad_out.shape[0], out_features_pad-out_features, device = DEVICE) # [m, out_features_pad - out_features]
        grad_out = torch.cat([grad_out, out_pad], dim=1) # [m * out_features_pad] = [m, p*k]

        # 2. x=x.flatten(1)
        # input: [m, p**k]
        grad_out = grad_out.view(M, P, 1, K) # [m, p, 1, k]

        # 3. x = morr_output_scale.matmul(x)  # [1, 1, 1, q] x [bs, p, q, k] = [bs, p, 1, k]
        # dL/d(morr_output_scale)
        if ctx.needs_input_grad[3]:
            grad_s = grad_out.matmul(x_morr.transpose(-2, -1)) # [bs, p, 1, q]
            grad_s = grad_s.sum(dim=(0, 1)) # [1, 1, 1, q] broadcast-compatible 
        else:
            grad_s = None
        # dL/dx
        grad_x = morr_output_scale.transpose(-2, -1).matmul(grad_out) # [bs, p, q, k]

        # 4. x = mrr_roundtrip_phase_to_tr(x)
        denominator = x_mrr.cos().mul_(c1).add_(c2 + c3)
        if intensity:
            denominator.square_()
            numerator = x_mrr.sin().mul_(c4)
        else:
            numerator = x_mrr.sin().mul_(c4 / 2)
            denominator = (
                denominator.sub(1).pow_(1.5).mul_(denominator.sub(c3).sqrt_())
            )
        grad_x = numerator.div_(denominator).mul_(grad_x) # [bs, p, q, k]
        
        # 5. x += phase_noise and morr_bias
        # these are assumed to have identity gradient

        # 6. x = weight.matmul(x) [1, p, q, k, k] * [bs, 1, q, k, 1] = [bs, p, q, k, 1]
        grad_x = grad_x.unsqueeze(-1) # [bs, p, q, k, 1]
        grad_morr_matmul = grad_x     # stash for weight gradient
        
        # dL/dx
        grad_x = torch.matmul(w_morr.transpose(-1, -2), grad_x) # [1, p, q, k, k] x [bs, p, q, k, 1] = [bs, p, q, k, 1]
        grad_x = grad_x.sum(dim=1, keepdim=True) # [bs, p, q, k, 1] -> [bs, 1, q, k, 1]
        grad_x = grad_x.squeeze(-1).squeeze(1) # [bs, 1, q, k, 1] -> [bs, q, k]

        # 7. input modulator
        grad_x = grad_x * 2 * x_modulator # [bs, q, k]

        # 8. input reshape
        grad_x = grad_x.view(x_input_shape)
        grad_x = grad_x[:, :in_features]



    # ----- Gradient w.r.t weight -----
    if ctx.needs_input_grad[1]:
        
        # 0. gradient after x = weight.matmul(x)
        # grad_morr_matmul # [bs, p, q, k, 1]

        # 1. x = weight.matmul(x)
        grad_w = torch.matmul(grad_morr_matmul, x_morr.transpose(-1,-2)) # [bs,p,q,k,k]
        grad_w = grad_w.sum(dim=0, keepdim=True) # [1,p,q,k,k]

        # 2. weight = toeplitz(weight)
        k = grad_w.size(-1)
        row = torch.arange(k)[:, None]        # (k,1)
        col = torch.arange(k)[None, :]        # (1,k)
        idx = (row - col) & (k - 1) if (k & (k-1)) == 0 else (row - col + k) % k

        idx = idx.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(DEVICE)
        buffer = torch.zeros_like(grad_w, device=DEVICE)
        buffer.scatter_add_(-2, idx, grad_w) # [1, p, q, k, k]
        grad_w = buffer.sum(dim=-1, keepdim=True).squeeze(0).squeeze(-1)

        # 3. build_weight() weight = self.weight.abs()
        grad_w = grad_w * weight.sign()


    return (
        grad_x,               # ∂L/∂x
        grad_w,          # ∂L/∂w
        grad_bias,        # ∂L/∂bias
        None, None, None, None, None, None, None, None, None, None, None,
        grad_s,  # ∂L/∂output_scale
        None, None, None, None, None, None
    )


morr_linear_fn.register_autograd(
    _morr_linear_backward, setup_context=_morr_linear_setup_context,
)