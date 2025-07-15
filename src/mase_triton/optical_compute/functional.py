from torch import Tensor

from .core.optical_transformer.linear import ot_qlinear_fn as _ot_qlinear_fn
from .core.optical_transformer.matmul import ot_qmatmul_fn as _ot_qmatmul_fn
from .core.optical_transformer.quantize import ot_quantize_fn as _ot_quantize_fn


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
    """
    Perform quantized linear transformation for optical transformer (https://arxiv.org/abs/2302.10360).

    Args:
        x (Tensor): Input tensor to be transformed.
        weight (Tensor): Weight matrix for the linear transformation.
        bias (Tensor | None): Optional bias vector. If None, no bias is applied.
        x_min (float): Minimum value for input quantization range.
        x_max (float): Maximum value for input quantization range.
        w_min (float): Minimum value for weight quantization range.
        w_max (float): Maximum value for weight quantization range.
        w_lut_min (float | None): Optional minimum value for weight lookup table.
            If None, uses w_min.
        o_min (float): Minimum value for output quantization range.
        o_max (float): Maximum value for output quantization range.
        q_levels (int): Number of quantization levels to use.
        q_seed (int): Random seed for quantization operations.
        skip_quantize (bool, optional): If True, skips quantization and performs
            standard linear operation. Defaults to False.

    Returns:
        tuple[Tensor, int]: A tuple containing:
            - output (Tensor): The quantized linear transformation result.
            - q_seed (int): Updated random seed after quantization operations.
    """
    output, q_seed = _ot_qlinear_fn(
        x=x,
        weight=weight,
        bias=bias,
        x_min=x_min,
        x_max=x_max,
        w_min=w_min,
        w_max=w_max,
        w_lut_min=w_lut_min,
        o_min=o_min,
        o_max=o_max,
        q_levels=q_levels,
        q_seed=q_seed,
        skip_quantize=skip_quantize,
    )
    return output, q_seed


def ot_qmatmul_fn(
    a: Tensor,
    b: Tensor,
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
    b_lut_min: float | None,
    o_min: float,
    o_max: float,
    q_levels: int,
    q_seed: int,
    skip_quantize: bool = False,
) -> tuple[Tensor, int]:
    """
    Perform quantized matrix multiplication for optical transformer (https://arxiv.org/abs/2302.10360).


    Args:
        a (Tensor): First input tensor (left operand of matrix multiplication).
        b (Tensor): Second input tensor (right operand of matrix multiplication).
        a_min (float): Minimum value for tensor 'a' quantization range.
        a_max (float): Maximum value for tensor 'a' quantization range.
        b_min (float): Minimum value for tensor 'b' quantization range.
        b_max (float): Maximum value for tensor 'b' quantization range.
        b_lut_min (float | None): Optional minimum value for tensor 'b' lookup table.
            If None, uses b_min.
        o_min (float): Minimum value for output quantization range.
        o_max (float): Maximum value for output quantization range.
        q_levels (int): Number of quantization levels to use.
        q_seed (int): Random seed for quantization operations.
        skip_quantize (bool, optional): If True, skips quantization and performs
            standard matrix multiplication. Defaults to False.

    Returns:
        tuple[Tensor, int]: A tuple containing:
            - output (Tensor): The quantized matrix multiplication result.
            - q_seed (int): Updated random seed after quantization operations.
    """
    output, q_seed = _ot_qmatmul_fn(
        a=a,
        b=b,
        a_min=a_min,
        a_max=a_max,
        b_min=b_min,
        b_max=b_max,
        b_lut_min=b_lut_min,
        o_min=o_min,
        o_max=o_max,
        q_levels=q_levels,
        q_seed=q_seed,
        skip_quantize=skip_quantize,
    )
    return output, q_seed


def ot_quantize_fn(
    x: Tensor,
    seed: int,
    quant_levels: int,
    min_val: float,
    max_val: float,
    lut_min: float | None,
    quant_mode: str = "det",
) -> tuple[Tensor, int]:
    """
    Quantize a tensor for optical transformer (https://arxiv.org/abs/2302.10360).

    Args:
        x (Tensor): Input tensor to be quantized.
        seed (int): Random seed for stochastic quantization operations.
        quant_levels (int): Number of discrete quantization levels.
        min_val (float): Minimum value of the quantization range.
        max_val (float): Maximum value of the quantization range.
        lut_min (float | None): Optional minimum value for lookup table.
            If None, uses min_val.
        quant_mode (str, optional): Quantization mode. Supported modes:
            - "det": Deterministic quantization (default)
            - Other modes may be supported by the underlying implementation.

    Returns:
        tuple[Tensor, int]: A tuple containing:
            - output (Tensor): The quantized tensor with values mapped to discrete levels.
            - seed_out (int): Updated random seed after quantization operations.
    """
    output, seed_out = _ot_quantize_fn(
        x=x,
        seed=seed,
        quant_levels=quant_levels,
        min_val=min_val,
        max_val=max_val,
        lut_min=lut_min,
        quant_mode=quant_mode,
    )
    return output, seed_out
