import torch
from torch import Tensor

from ..utils.bit_repr import get_binary_repr, get_binary_repr_fp32
from .meta import MinifloatMeta


def extract_minifloat_component(x: Tensor, meta: MinifloatMeta) -> Tensor:
    y_exp_bits = meta.exp_bits
    y_frac_bits = meta.frac_bits
    always_finite = meta.is_finite

    round_mode = meta.round_mode
    y_exp_bias = (1 << (y_exp_bits - 1)) - 1  # 2^(y_exp_bits - 1) - 1
    # if always_finite: 2^y_exp_bits - 1 - bias = 2^y_exp_bits - 1 - 2^(y_exp_bits - 1) + 1 = 2^(y_exp_bits - 1)
    y_exp_max = (1 << y_exp_bits) - 1 if always_finite else (1 << y_exp_bits) - 2
    y_exp_max_biased = y_exp_max - y_exp_bias
    y_exp_min = 0
    y_exp_min_biased = y_exp_min - y_exp_bias
    y_frac_max = (1 << y_frac_bits) - 1

    x = x.to(torch.float32)
    y_sign = x < 0
    x_int32 = x.abs().view(torch.int32)
    flush_to_zero = (x_int32 >> 23) & 0xFF == 0
    x_normal = torch.where(flush_to_zero, 0.0, x)
    # (float32, int32)
    x_frac, x_exp = x_normal.abs().frexp()
    x_frac = x_frac * 2
    x_exp = x_exp - 1
    if not always_finite:
        x_is_inf = x.isinf()
        x_is_nan = x.isnan()

    y_exp = x_exp
    underflow = y_exp < y_exp_min_biased
    overflow = y_exp > y_exp_max_biased
    y_exp = y_exp + y_exp_bias

    x_frac_scaled = x_frac * (1 << y_frac_bits)
    if round_mode == "rtz":
        # truncate towards zero
        y_frac = x_frac_scaled.view(torch.int32) >> (23 - y_frac_bits)
        y_frac = (y_frac << (23 - y_frac_bits)).view(torch.float32)
    elif round_mode == "rte":
        y_frac = x_frac_scaled.round()
    elif round_mode == "rtp":
        # round towards positive infinity
        y_frac = x_frac_scaled.ceil()
    elif round_mode == "rtn":
        # round towards negative infinity
        y_frac = x_frac_scaled.floor()
    else:
        raise ValueError(f"Unknown round mode: {round_mode}")
    y_frac = y_frac.view(torch.int32) & 0x7FFFFF
    y_frac = y_frac >> (23 - y_frac_bits)
    y_is_subnormal = (y_exp == y_exp_min) & (y_frac != 0)
    # add implicit leading 1 and shift for subnormals
    y_frac = torch.where(y_is_subnormal, (y_frac | (1 << y_frac_bits)) >> 1, y_frac)
    # underflow -> 0, overflow -> max
    y_frac = torch.where(underflow, 0, y_frac)
    y_exp = torch.where(underflow, 0, y_exp)
    # overflow -> max
    y_frac = torch.where(overflow, y_frac_max, y_frac)
    y_exp = torch.where(overflow, y_exp_max, y_exp)
    if not always_finite:
        y_frac = torch.where(x_is_inf, 0, y_frac)
        y_frac = torch.where(x_is_nan, (1 << y_frac_bits) - 1, y_frac)
        y_exp = torch.where(x_is_inf, y_exp_max, y_exp)
        y_exp = torch.where(x_is_nan, y_exp_max, y_exp)
    y = (y_exp << y_frac_bits) | y_frac
    y = torch.where(y_sign, y + (1 << (y_exp_bits + y_frac_bits)), y)
    y = y.to(torch.uint16)
    return y


def compose_minifloat_component(elements: Tensor, meta: MinifloatMeta) -> Tensor:
    x_exp_bits = meta.exp_bits
    x_frac_bits = meta.frac_bits
    always_finite = meta.is_finite

    fp32_exp_bias = 127

    x_exp_bias = (1 << (x_exp_bits - 1)) - 1

    assert elements.dtype == torch.uint16
    elements = elements.to(torch.int32)
    y_sign = (elements & (1 << (x_exp_bits + x_frac_bits))) != 0

    elements = (elements & 0x7FFF).to(torch.int32)
    x_exp = (elements >> x_frac_bits) & ((1 << x_exp_bits) - 1)
    x_frac = elements & ((1 << x_frac_bits) - 1)
    is_subnormal = (x_exp == 0) & (x_frac != 0)
    is_zero = (x_exp == 0) & (x_frac == 0)

    if not always_finite:
        y_is_not_finite = x_exp == ((1 << x_exp_bits) - 1)
        y_is_inf = y_is_not_finite & (x_frac == 0)
        y_is_nan = y_is_not_finite & (x_frac != 0)

    y_exp = (x_exp - x_exp_bias + fp32_exp_bias) << 23
    y_frac = torch.where(
        is_subnormal, (x_frac & ((1 << (x_frac_bits - 1)) - 1)) << 1, x_frac
    )
    y_frac = y_frac << (23 - x_frac_bits)
    y = y_exp | y_frac
    y = y.view(torch.float32)
    y = torch.where(y_sign, -y, y)
    if not always_finite:
        y = torch.where(y_is_inf, float("inf"), y)
        y = torch.where(y_is_nan, float("nan"), y)
    y = torch.where(is_zero, 0.0, y)
    return y
