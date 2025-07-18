import torch
import triton
from torch import Tensor
from triton import language as tl


@triton.jit
def _minifloat_extract_kernel(
    x_ptr,
    out_ptr,
    n_elems,
    BLK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLK + tl.arange(0, BLK)

    x_val = tl.load(x_ptr + offs, mask=offs < n_elems, other=0.0)
    x_i16 = x_val.cast(tl.int16, bitcast=True)

    sign_raw     = (x_i16 & -32768) >> 15
    exp_raw      = (x_i16 & 0x7F80) >> 7    
    mantissa_raw = x_i16 & 0x007F           

    exp_new = exp_raw - 127 + 7
    under   = exp_new < 0
    over    = exp_new > 15
    exp_new = tl.where(under, 0, exp_new)
    exp_new = tl.where(over,  15, exp_new)

    mant_new = mantissa_raw >> 4

    packed = (sign_raw << 7) | (exp_new << 3) | mant_new
    packed = packed.cast(tl.uint8)
    tl.store(out_ptr + offs, packed, mask=offs < n_elems)


@triton.jit
def _minifloat_compose_kernel(
    in_ptr,
    out_ptr,
    n_elems,
    BLK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLK + tl.arange(0, BLK)

    packed = tl.load(in_ptr + offs, mask=offs < n_elems, other=0).cast(tl.int16)

    sign_bit = (packed & 0x80) >> 7
    exp_val  = (packed & 0x78) >> 3
    man_val  = packed & 0x7

    exp_full  = (exp_val - 7 + 127) << 7
    man_full  = man_val << 4
    sign_full = sign_bit << 15
    output = sign_full | exp_full | man_full

    output = output.cast(tl.bfloat16, bitcast=True)
    tl.store(out_ptr + offs, output, mask=offs < n_elems)


def extract_minifloat_components(x: Tensor) -> Tensor:
    assert x.ndim == 1
    assert x.dtype == torch.bfloat16

    n = x.numel()
    packed = torch.empty(n, dtype=torch.uint8, device=x.device)
    grid = (triton.cdiv(n, 128),)

    _minifloat_extract_kernel[grid](
        x, packed, n, 128,
    )
    return packed


def compose_minifloat_tensor(packed: Tensor) -> Tensor:
    assert packed.ndim == 1
    n = packed.numel()
    out = torch.empty(n, dtype=torch.bfloat16, device=packed.device)

    grid = (triton.cdiv(n, 128),)

    _minifloat_compose_kernel[grid](
        packed, out, n, 128,
    )
    return out
