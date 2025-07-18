import torch
from torch import Tensor

def extract_minifloat_components(x: Tensor) -> Tensor:
    """
    bfloat16 -> uint8(minifloat8) bit-patterns
    """
    assert x.dtype == torch.bfloat16

    x_bits = x.view(torch.int16)

    sign    = (x_bits >> 15) & 0x1
    exp     = (x_bits >> 7)  & 0xFF
    mant    = (x_bits >> 4)  & 0x7

    exp = (exp.to(torch.int32) - 127 + 7).clamp(0, 15).to(torch.int16)

    packed = (sign << 7) | (exp << 3) | mant

    return packed.to(torch.uint8)


def compose_minifloat_tensor(x: Tensor) -> Tensor:
    """
    uint8(minifloat8) bit-patterns -> bfloat16
    """
    assert x.dtype == torch.uint8

    sign = ((x >> 7) & 0x1).to(torch.int16)
    exp = ((x >> 3) & 0xF).to(torch.int16)
    mant = (x & 0x7).to(torch.int16)

    exp = (exp.to(torch.int32) + 127 - 7).clamp(0, 255).to(torch.int16)

    out_bits = (sign << 15) | (exp << 7) | (mant << 4)

    return out_bits.view(torch.bfloat16)
