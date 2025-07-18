import torch
from torch import Tensor

from .meta import MXFPMeta


def extract_mxfp_components(x: Tensor, mxfp_meta: MXFPMeta) -> tuple[Tensor, Tensor]:
    """
    Extracts the scale and element components from a MXFP tensor.

    Args:
        x (Tensor): The input MXFP tensor.
        mxfp_meta (MXFPMeta): The metadata for the MXFP format.
    Returns:
        tuple[Tensor, Tensor]: A tuple containing the scale (shape = [num_blocks, 1]) and element tensors (shape = [num_blocks, block_size]).
    """
    assert x.dtype == torch.bfloat16
    B = mxfp_meta.block_size
    assert x.numel() % B == 0, (
        f"Input tensor size {x.numel()} is not divisible by block size {B}."
    )
    n_blocks = x.numel() // B
    sc_exp_max = (2 << mxfp_meta.scale_exp_bits) - 1
    el_exp_bits = mxfp_meta.element_exp_bits
    el_exp_max = (2 << el_exp_bits) - 1
    el_exp_bias = (1 << (el_exp_bits - 1)) - 1
    el_frac_bits = mxfp_meta.element_frac_bits
    el_frac_max = (2 << el_frac_bits) - 1
    el_implicit_bit = 1 << el_frac_bits
    el_sign_mask = 1 << (el_exp_bits + el_frac_bits)
    x_exp_frac_mask = 0x7FFF

    x = x.flatten()
    x = x.reshape(n_blocks, B)  # [n_blocks, B]
    x_int16 = x.view(torch.int16)
    exp = (x_int16 & 0x7F80) >> 7  # 0-255
    exp_max = exp.max(dim=1, keepdim=True).values  # [n_blocks, 1]
    zero_mask = (x_int16 & x_exp_frac_mask) == 0
    exp = exp - exp_max
    # exp of minifloat
    el_exp = exp + el_exp_bias
    # whether the minifloat is subnormal
    subnormal_mask = (el_exp == 0) & (~zero_mask)
    underflow_mask = (el_exp < 0) | zero_mask
    overflow_mask = el_exp > el_exp_max
    el_exp = torch.where(underflow_mask, 0, el_exp)
    el_exp = torch.where(overflow_mask, el_exp_max, el_exp)

    el_frac = x.view(torch.int16) & 0x007F
    el_frac = el_frac >> (7 - el_frac_bits)
    # add implicit bit for subnormal minifloat
    el_frac = torch.where(subnormal_mask, (el_implicit_bit | el_frac) >> 1, el_frac)
    el_frac = torch.where(underflow_mask, 0, el_frac)
    el_frac = torch.where(overflow_mask, el_frac_max, el_frac)

    sign = x.view(torch.int16) & 0x8000
    sign = sign >> (15 - (el_exp_bits + el_frac_bits))
    sign = sign & el_sign_mask

    el = sign | (el_exp << el_frac_bits) | el_frac
    el = el.view(torch.uint16).to(torch.uint8)

    exp_max = exp_max.clamp(0, sc_exp_max).view(torch.uint16).to(torch.uint8)
    return exp_max, el


def compose_mxfp_tensor(
    scales: Tensor,
    elements: Tensor,
    mxfp_meta: MXFPMeta,
):
    """
    Composes a MXFP tensor from the scale and element components.

    Args:
        shared_scales (Tensor): The shared scales tensor.
        elements (Tensor): The elements tensor.
        mxfp_meta (MXFPMeta): The metadata for the MXFP format.

    Returns:
        Tensor: The composed MXFP tensor.
    """
    assert scales.dtype == torch.uint8
    assert elements.dtype == torch.uint8

    B = mxfp_meta.block_size
    n_blocks = scales.shape[0]
    el_exp_bits = mxfp_meta.element_exp_bits
    el_frac_bits = mxfp_meta.element_frac_bits
    el_frac_mask = (1 << el_frac_bits) - 1
    el_exp_bias = (1 << (el_exp_bits - 1)) - 1
    el_exp_frac_mask = (1 << (el_exp_bits + el_frac_bits)) - 1

    exp_max = scales.to(torch.uint16).view(torch.int16)
    exp_max = exp_max.expand(n_blocks, B)  # [n_blocks, B]

    elements = elements.to(torch.int16)
    zero_mask = (elements & el_exp_frac_mask) == 0
    sign = elements << (15 - (el_exp_bits + el_frac_bits))
    sign = sign & 0x8000

    el_exp = (elements >> el_frac_bits) & ((1 << el_exp_bits) - 1)
    # remove implicit bit
    subnormal_mask = (el_exp == 0) & (~zero_mask)
    fraction = torch.where(subnormal_mask, elements << 1, elements)
    fraction = fraction & el_frac_mask
    fraction = fraction << (7 - el_frac_bits)
    el_exp = el_exp - el_exp_bias
    exp = exp_max + el_exp
    exp = exp << 7

    dequantized = sign | exp | fraction
    dequantized = dequantized.view(torch.bfloat16)
    dequantized = torch.where(zero_mask, 0.0, dequantized)
    dequantized = dequantized.reshape(n_blocks * B)
    return dequantized
