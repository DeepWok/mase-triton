import triton
import triton.language as tl


@triton.jit
def _noisy_quantize(
    x: tl.tensor,
    quant_levels,
    min_val,
    max_val,
    lut_min,
    seed,
    INPUT_DTYPE: tl.constexpr,
    QUANT_MODE: tl.constexpr,
    ENABLE_LUT_MIN: tl.constexpr,
):
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
        noise = tl.rand(seed, tl.arange(0, 1)) + bias
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
    return x
