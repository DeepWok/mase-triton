from typing import Literal
from torch import Tensor

from .quantize import ot_quantize_fn, _ot_quantize_forward_kernel
from .linear import ot_qlinear_fn, _ot_qlinear_forward_kernel
from .matmul import ot_qmatmul_fn, _ot_qmatmul_forward_kernel


class OpticalTransformerFunctions:
    """
    OpticalTransformerFunctions provides a collection of static methods for simulating
    optical transformer operations accelerated by Triton kernels.

    - Optical Transformers: https://arxiv.org/abs/2302.10360
    """

    kernels = {
        "optical_transformer::quantize_forward_kernel": _ot_quantize_forward_kernel,
        "optical_transformer::quantized_linear_forward_kernel": _ot_qlinear_forward_kernel,
        "optical_transformer::quantized_matmul_forward_kernel": _ot_qmatmul_forward_kernel,
    }

    @staticmethod
    def quantize_fn(
        x: Tensor,
        seed: int,
        quant_levels: int,
        min_val: float,
        max_val: float,
        lut_min: float | None,
        quant_mode: Literal["det", "rand"],
    ) -> tuple[Tensor, int]:
        """
        Applies quantization to the input tensor using the specified parameters.

        Args:
            x (Tensor): The input tensor to be quantized.
            seed (int): The random seed for reproducibility in quantization.
            quant_levels (int): The number of quantization levels to use.
            min_val (float): The minimum value for the quantization range.
            max_val (float): The maximum value for the quantization range.
            lut_min (float | None): The minimum value for the weight lookup table (LUT),
                if applicable. If None, no LUT is used.
            quant_mode (str): The quantization mode to use. Can be either "det" for
                deterministic quantization or "rand" for quantization + random noise.

        Returns:
            tuple[Tensor, int]: A tuple containing the quantized tensor and an
            integer representing the updated seed.
        """
        return ot_quantize_fn(
            x=x,
            seed=seed,
            quant_levels=quant_levels,
            min_val=min_val,
            max_val=max_val,
            lut_min=lut_min,
            quant_mode=quant_mode,
        )

    @staticmethod
    def quantized_linear_fn(
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
        Fused quantized linear operation: Quantize input, quantize weight with w_lut_min (if provided),
        linear matmul, and quantize output + add noise.

        Args:
            x (Tensor): The input tensor to be transformed.
            weight (Tensor): The weight tensor for the linear transformation.
            bias (Tensor | None): The bias tensor for the linear transformation.
            x_min (float): The minimum value for the input quantization range.
            x_max (float): The maximum value for the input quantization range.
            w_min (float): The minimum value for the weight quantization range.
            w_max (float): The maximum value for the weight quantization range.
            w_lut_min (float | None): The minimum value for the weight lookup table (LUT),
                if applicable. If None, no LUT is used.
            o_min (float): The minimum value for the output quantization range.
            o_max (float): The maximum value for the output quantization range.
            q_levels (int): The number of quantization levels to use.
            q_seed (int): The random seed for reproducibility in quantization.
            skip_quantize (bool, optional): Whether to skip quantization. Defaults to False.
                Useful for debugging.

        Returns:
            tuple[Tensor, int]: A tuple containing the transformed tensor and an
            integer representing the updated seed.
        """
        return ot_qlinear_fn(
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

    @staticmethod
    def quantized_matmul_fn(
        a: Tensor,
        b: Tensor,
        a_min: float,
        a_max: float,
        b_min: float,
        b_max: float,
        b_lut_min: float,
        o_min: float,
        o_max: float,
        q_levels: int,
        q_seed: int,
        skip_quantize: bool = False,
    ) -> tuple[Tensor, int]:
        return ot_qmatmul_fn(
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
