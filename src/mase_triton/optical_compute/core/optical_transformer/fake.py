import torch
from torch import Tensor


def _quantize_forward_fn_fake(
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


class _QuantizeFnFake(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        seed: int,
        quant_levels: int,
        min_val: float,
        max_val: float,
        lut_min: float | None,
        quant_mode: str,
    ) -> tuple[Tensor, int]:
        r_val, seed_out = _quantize_forward_fn_fake(
            x, seed, quant_levels, min_val, max_val, lut_min, quant_mode
        )
        ctx.save_for_backward(x)
        ctx.quant_levels = quant_levels
        ctx.min_val = min_val
        ctx.max_val = max_val
        ctx.lut_min = lut_min
        ctx.quant_mode = quant_mode
        return r_val, seed_out

    @staticmethod
    def backward(ctx, grad_output: Tensor, grad_seed: None) -> tuple:
        # straight-through estimator
        (x,) = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_seed = None
        r_val = (grad_x, grad_seed, None, None, None, None, None)
        return r_val


def quantize_fn(
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
    r_val, seed_out = _QuantizeFnFake.apply(
        x, seed, quant_levels, min_val, max_val, lut_min, quant_mode
    )
    return r_val, seed_out


def _qlinear_fn_fake(
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
    """CPU implementation of the optical transformer quantized linear function."""

    # Store original shape
    ori_x_shape = x.size()
    x = x.reshape(-1, ori_x_shape[-1])

    if not skip_quantize:
        x_quantized, q_seed = _quantize_forward_fn_fake(
            x, q_seed, q_levels, x_min, x_max, None, "det"
        )
        w_quantized, q_seed = _quantize_forward_fn_fake(
            weight,
            q_seed,
            q_levels,
            w_min,
            w_max,
            w_lut_min,
            "det" if w_lut_min is None else "rand",
        )
    else:
        x_quantized = x
        w_quantized = weight

    # Perform matrix multiplication: x @ weight.T
    output = torch.matmul(x_quantized, w_quantized.T)

    if not skip_quantize:
        # Quantize output (random)
        output, q_seed = _quantize_forward_fn_fake(
            output, q_seed, q_levels, o_min, o_max, None, "rand"
        )

    # Add bias if provided
    if bias is not None:
        output = output + bias

    # Restore original shape
    output = output.reshape(ori_x_shape[:-1] + (weight.shape[0],))

    # Update seed (only increment if quantization was applied and random mode was used)
    q_seed = q_seed + 1 if not skip_quantize else q_seed

    return output, q_seed


class _QLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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
        r_val, q_seed_out = _qlinear_fn_fake(
            x,
            weight,
            bias,
            x_min,
            x_max,
            w_min,
            w_max,
            w_lut_min,
            o_min,
            o_max,
            q_levels,
            q_seed,
            skip_quantize,
        )
        ctx.save_for_backward(x, weight, bias)
        ctx.x_min = x_min
        ctx.x_max = x_max
        ctx.w_min = w_min
        ctx.w_max = w_max
        ctx.w_lut_min = w_lut_min
        ctx.o_min = o_min
        ctx.o_max = o_max
        ctx.q_levels = q_levels
        ctx.q_seed = q_seed
        ctx.skip_quantize = skip_quantize
        return r_val, q_seed_out

    @staticmethod
    def backward(ctx, *grad_outputs) -> tuple:
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_outputs[0] @ weight
        if ctx.needs_input_grad[1]:
            grad_weight = grad_outputs[0].transpose(-1, -2) @ x
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_outputs[0].sum(0)

        return (
            grad_x,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def qlinear_fn(
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
    r_val, q_seed_out = _QLinearFn.apply(
        x,
        weight,
        bias,
        x_min,
        x_max,
        w_min,
        w_max,
        w_lut_min,
        o_min,
        o_max,
        q_levels,
        q_seed,
        skip_quantize,
    )
    return r_val, q_seed_out


def _qmatmul_fn_fake(
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
    """CPU implementation of the optical transformer quantized matrix multiplication function."""

    # Store original shape
    orig_a_shape = a.size()

    # Extract dimensions
    M, K = a.shape[-2:]
    K2, N = b.shape[-2:]
    assert K == K2, "K dimension must match"

    # Reshape for batch processing
    a_reshaped = a.reshape(-1, M, K)
    b_reshaped = b.reshape(-1, K, N)

    if not skip_quantize:
        # Quantize input matrix a (deterministic mode, no LUT min)
        a_quantized, q_seed = _quantize_forward_fn_fake(
            a_reshaped, q_seed, q_levels, a_min, a_max, None, "det"
        )

        # Quantize input matrix b (deterministic mode, with optional LUT min)
        b_quantized, q_seed = _quantize_forward_fn_fake(
            b_reshaped, q_seed, q_levels, b_min, b_max, b_lut_min, "det"
        )
    else:
        a_quantized = a_reshaped
        b_quantized = b_reshaped

    # Perform batch matrix multiplication: a @ b
    output = torch.matmul(a_quantized, b_quantized)

    if not skip_quantize:
        # Quantize output (random mode, no LUT min)
        output, q_seed = _quantize_forward_fn_fake(
            output, q_seed, q_levels, o_min, o_max, None, "rand"
        )

    # Restore original shape
    output = output.reshape(orig_a_shape[:-2] + (M, N))

    # Update seed (increment if quantization was applied and random mode was used for output)
    q_seed = q_seed + 1 if not skip_quantize else q_seed

    return output, q_seed


class _QMatMulFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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
        r_val, q_seed_out = _qmatmul_fn_fake(
            a,
            b,
            a_min,
            a_max,
            b_min,
            b_max,
            b_lut_min,
            o_min,
            o_max,
            q_levels,
            q_seed,
            skip_quantize,
        )
        ctx.save_for_backward(a, b)
        ctx.a_min = a_min
        ctx.a_max = a_max
        ctx.b_min = b_min
        ctx.b_max = b_max
        ctx.b_lut_min = b_lut_min
        ctx.o_min = o_min
        ctx.o_max = o_max
        ctx.q_levels = q_levels
        ctx.q_seed = q_seed
        ctx.skip_quantize = skip_quantize
        return r_val, q_seed_out

    @staticmethod
    def backward(ctx, grad_output: Tensor, grad_q_seed: None) -> tuple:
        # straight-through estimator
        (a, b) = ctx.saved_tensors
        grad_a = torch.matmul(grad_output, b.transpose(-2, -1))
        grad_b = torch.matmul(a.transpose(-2, -1), grad_output)
        grad_q_seed = None
        r_val = (
            grad_a,
            grad_b,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_q_seed,
        )
        return r_val


def qmatmul_fn(
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
    r_val, q_seed_out = _QMatMulFn.apply(
        a,
        b,
        a_min,
        a_max,
        b_min,
        b_max,
        b_lut_min,
        o_min,
        o_max,
        q_levels,
        q_seed,
        skip_quantize,
    )
    return r_val, q_seed_out
