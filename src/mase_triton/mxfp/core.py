import torch
import triton
from torch import Tensor
from triton import language as tl

from .meta import MXFPMeta


def _find_block_max(x: Tensor, block_size: int) -> Tensor:
    B = block_size
    n_blocks = x.numel() // B

    x = x.view(n_blocks, B)
    group_max = x.abs().max(dim=1, keepdim=True).values
    return group_max


@triton.jit
def _extract_mxfp_components_kernel(
    x_ptr,
    block_max_ptr,
    element_ptr,
    scale_ptr,
    n_elements: int,
    n_blocks: int,
    block_size: tl.constexpr,
    sc_exp_bits: tl.constexpr,
    el_exp_bits: tl.constexpr,
    el_man_bits: tl.constexpr,
    BLK: tl.constexpr,
):
    # helper constants
    sc_exp_max = 2**sc_exp_bits - 1
    el_exp_max = 2**el_exp_bits - 1
    el_exp_bias = 2 ** (el_exp_bits - 1) - 1
    el_man_max = 2**el_man_bits - 1
    el_sign_mask = 2 ** (el_exp_bits + el_man_bits)

    pid = tl.program_id(axis=0)
    x_offs = pid * BLK + tl.arange(0, BLK)
    block_max_offs = x_offs // block_size

    x_ptrs = x_ptr + x_offs
    block_max_ptrs = block_max_ptr + block_max_offs
    x = tl.load(x_ptrs, mask=x_offs < n_elements, other=0.0)
    block_max = tl.load(block_max_ptrs, mask=block_max_offs < n_blocks, other=0.0)

    x = x.cast(tl.int16, bitcast=True)
    block_max = block_max.cast(tl.int16, bitcast=True)
    exp_max = (block_max & 0x7F80) >> 7  # 0-255
    el_exp = (x & 0x7F80) >> 7  # 0-255
    el_exp = el_exp - exp_max
    el_exp = el_exp + el_exp_bias
    underflow_mask = el_exp < 0
    overflow_mask = el_exp > el_exp_max
    el_exp = tl.where(underflow_mask, 0, el_exp)
    # FIXME
    tl.static_assert(el_exp.dtype == tl.int16, "el_exp must be int16")
    el_exp = tl.where(underflow_mask, 0, el_exp)
    el_exp = tl.where(overflow_mask, el_exp_max, el_exp)

    el_mantissa = x & 0x007F
    el_mantissa = el_mantissa >> (7 - el_man_bits)
    el_mantissa = tl.where(underflow_mask, 0, el_mantissa)
    el_mantissa = tl.where(overflow_mask, el_man_max, el_mantissa)
    sign = x & 0x8000
    sign = sign >> (15 - (el_exp_bits + el_man_bits))
    sign = sign & el_sign_mask

    el = sign | (el_exp << el_man_bits) | el_mantissa
    el = el.cast(tl.uint8)
    el_ptrs = element_ptr + x_offs
    tl.store(el_ptrs, el, mask=x_offs < n_elements)

    sc = tl.clamp(exp_max, 0, sc_exp_max).cast(tl.uint8)
    sc_ptrs = scale_ptr + block_max_offs
    sc_mask = (block_max_offs < n_blocks) & (x_offs % block_size == 0)
    tl.store(sc_ptrs, sc, mask=sc_mask)


def extract_mxfp_components(
    x: Tensor,
    mxfp_meta: MXFPMeta,
):
    assert x.dtype == torch.bfloat16
    assert x.ndim == 1
    x = x.contiguous()
    n_elements = x.numel()
    B = mxfp_meta.block_size
    assert n_elements % B == 0
    n_groups = n_elements // B
    device = x.device
    scales = torch.empty(n_groups, dtype=torch.uint8, device=device)
    elements = torch.empty(n_elements, dtype=torch.uint8, device=device)

    block_max = _find_block_max(x, B)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLK"]),)

    _extract_mxfp_components_kernel[grid](
        x,
        block_max,
        elements,
        scales,
        n_elements=n_elements,
        n_blocks=n_groups,
        block_size=B,
        sc_exp_bits=mxfp_meta.scale.exponent_bits,
        el_exp_bits=mxfp_meta.element.exponent_bits,
        el_man_bits=mxfp_meta.element.mantissa_bits,
        BLK=128,
    )

    return scales, elements
