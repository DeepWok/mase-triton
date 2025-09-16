"""
Optical Transformers: https://arxiv.org/abs/2302.10360
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from ....about import PACKAGE_NAME
from ....dtype import TORCH_DTYPE_TO_TRITON


def _get_autotune_configs_ot_quantize_forward_kernel():
    block_sizes = [128, 256, 512, 1024]
    stages = [1, 2, 3, 4]
    configs = []
    for bs in block_sizes:
        for s in stages:
            configs.append(triton.Config({"BLOCK_SIZE": bs}, num_stages=s))
    return configs


@triton.jit
def _ot_quantize_forward_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    quant_levels,
    min_val,
    max_val,
    lut_min,
    seed,
    QUANT_MODE: tl.constexpr,
    ENABLE_LUT_MIN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    load_mask = offsets < n_elements
    x: tl.tensor = tl.load(x_ptr + offsets, mask=load_mask)

    min_val = tl.full((1,), min_val, dtype=INPUT_DTYPE)
    max_val = tl.full((1,), max_val, dtype=INPUT_DTYPE)
    range_val = max_val - min_val
    eps = tl.full((1,), 1e-8, dtype=INPUT_DTYPE)
    quant_levels = tl.full((1,), quant_levels - 1, dtype=INPUT_DTYPE)

    x = tl.clamp(x, min_val, max_val)
    x = x - min_val
    x = x / (range_val + eps)
    x = x * quant_levels

    if QUANT_MODE == "det":
        x = tl.extra.cuda.libdevice.rint(x)
        x = x / quant_levels
        x = x * range_val + min_val
    else:
        # rand
        bias = tl.full((1,), -0.5, dtype=INPUT_DTYPE)
        noise = tl.rand(seed, offsets) + bias
        x = x + noise
        x = tl.extra.cuda.libdevice.rint(x)
        x = x / quant_levels
        x = x * range_val + min_val
    if ENABLE_LUT_MIN:
        lut_min = tl.full((1,), lut_min, dtype=INPUT_DTYPE)
        threshold = lut_min * max_val
        x_mask = x < threshold
        x_mask = x_mask & (x > 0.0)
        x = tl.where(x_mask, threshold, x)
        threshold = lut_min * min_val.abs()
        x_mask = x > -threshold
        x_mask = x_mask & (x < 0.0)
        x = tl.where(x_mask, -threshold, x)

    tl.store(output_ptr + offsets, x, mask=load_mask)


@torch.library.custom_op(
    f"{PACKAGE_NAME}::optical_transformer_quantize_fn",
    mutates_args={},
)
def ot_quantize_fn(
    x: Tensor,
    seed: int,
    quant_levels: int,
    min_val: float,
    max_val: float,
    lut_min: float | None,
    quant_mode: str,
) -> tuple[Tensor, int]:
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32), (
        f"Unsupported dtype {x.dtype}"
    )
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert quant_mode in ["det", "rand"], f"Unsupported quant_mode {quant_mode}"
    assert lut_min is None or lut_min >= 0.0, "lut_min must be non-negative"

    enable_lut_min = lut_min is not None
    lut_min = lut_min or 0.0

    output = torch.empty_like(x)
    n_elements = x.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch.cuda.device(x.device.index):
        _ot_quantize_forward_kernel[grid](
            x,
            output,
            n_elements=n_elements,
            quant_levels=quant_levels,
            min_val=min_val,
            max_val=max_val,
            lut_min=lut_min,
            seed=seed,
            QUANT_MODE=quant_mode,
            ENABLE_LUT_MIN=enable_lut_min,
            INPUT_DTYPE=TORCH_DTYPE_TO_TRITON[x.dtype],
            BLOCK_SIZE=256,
        )

    if quant_mode == "rand":
        seed += 1
    return output, seed


@ot_quantize_fn.register_fake
def _ot_quantize_forward_fn_fake(
    x: Tensor,
    seed: int,
    quant_levels: int,
    min_val: float,
    max_val: float,
    lut_min: float | None,
    quant_mode: str,
) -> tuple[Tensor, int]:
    output = torch.empty_like(x, dtype=x.dtype)
    return output, seed


def _ot_quantize_backward(ctx, *grad_outputs):
    return grad_outputs[0], None, None, None, None, None, None


def _ot_quantize_setup_context(ctx, inputs, output):
    return None


@ot_quantize_fn.register_kernel("cpu")
def _ot_quantize_forward_fn_cpu(
    x: Tensor,
    seed: int,
    quant_levels: int,
    min_val: float,
    max_val: float,
    lut_min: float | None,
    quant_mode: str,
) -> tuple[Tensor, int]:
    """
    CPU implementation of optical transformer quantization using PyTorch APIs.
    Equivalent to the Triton kernel _ot_quantize_forward_kernel.
    """
    # Set random seed for reproducibility if using stochastic quantization
    if quant_mode == "rand":
        torch.manual_seed(seed)

    # Step 1: Clamp input to [min_val, max_val]
    x_clamped = torch.clamp(x, min_val, max_val)

    # Step 2: Normalize to [0, 1] range
    range_val = max_val - min_val
    eps = 1e-8
    x_normalized = (x_clamped - min_val) / (range_val + eps)

    # Step 3: Scale to quantization levels [0, quant_levels-1]
    x_scaled = x_normalized * (quant_levels - 1)

    # Step 4: Quantization (deterministic or stochastic)
    if quant_mode == "det":
        # Deterministic: simple rounding
        x_quantized = torch.round(x_scaled)
    else:  # quant_mode == "rand"
        # Stochastic: add uniform noise in [-0.5, 0.5] then round
        torch.manual_seed(seed)
        noise = torch.rand_like(x_scaled) - 0.5
        x_quantized = torch.round(x_scaled + noise)

    # Step 5: Dequantization - scale back to original range
    x_dequantized = (x_quantized / (quant_levels - 1)) * range_val + min_val

    # Step 6: Apply LUT minimum thresholding if enabled
    if lut_min is not None:
        # For positive values: if x < lut_min * max_val and x > 0, set to lut_min * max_val
        pos_threshold = lut_min * max_val
        pos_mask = (x_dequantized < pos_threshold) & (x_dequantized > 0.0)
        x_dequantized = torch.where(pos_mask, pos_threshold, x_dequantized)

        # For negative values: if x > -lut_min * |min_val| and x < 0, set to -lut_min * |min_val|
        neg_threshold = -lut_min * abs(min_val)
        neg_mask = (x_dequantized > neg_threshold) & (x_dequantized < 0.0)
        x_dequantized = torch.where(neg_mask, neg_threshold, x_dequantized)

    # Update seed for next call if using stochastic quantization
    if quant_mode == "rand":
        seed += 1

    return x_dequantized, seed


ot_quantize_fn.register_autograd(
    _ot_quantize_backward,
    setup_context=_ot_quantize_setup_context,
)
