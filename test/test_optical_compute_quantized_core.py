import torch

from mase_triton.optical_compute import OpticalTransformerFunctions as OTFunctions
from mase_triton.logging import test_logger, set_logging_verbosity

DEVICE = "cuda"

logger = test_logger.getChild(f"{__name__}")


def test_optical_compute_quantized_forward_fn_simple():
    x = torch.rand(8, device=DEVICE, dtype=torch.float32)
    x = x * 2 - 1
    quant_levels = 256
    min_val = -1.0
    max_val = 1.0
    lut_min = 0.01
    seed = 0

    out, seed_out = OTFunctions.optical_compute_quantize_fn(
        x,
        quant_levels=quant_levels,
        min_val=min_val,
        max_val=max_val,
        lut_min=lut_min,
        quant_mode="det",
        seed=seed,
    )
    assert (out - x).abs().max().item() < 1 / quant_levels

    logger.info("Test passed: output is close to input")


def test_optical_compute_quantized_backward_fn_simple():
    quant_levels = 256
    min_val = -1.0
    max_val = 1.0
    lut_min = 0.01
    seed = 0

    x = torch.rand(256, device=DEVICE, dtype=torch.float32)
    x = x * 2 - 1
    x.requires_grad_()
    out, seed_out = OTFunctions.optical_compute_quantize_fn(
        x,
        quant_levels=quant_levels,
        min_val=min_val,
        max_val=max_val,
        lut_min=lut_min,
        quant_mode="det",
        seed=seed,
    )
    loss = torch.sum(out)
    loss.backward()
    assert torch.all(x.grad == 1.0)
    logger.info(f"Identical gradients test passed")


def test_optical_compute_quantized_linear_forward_fn_skip_quantize():
    x = torch.rand(16, 32, device=DEVICE, dtype=torch.float16)
    w = torch.rand(8, 32, device=DEVICE, dtype=torch.float16)
    bias = torch.rand(8, device=DEVICE, dtype=torch.float16)

    out_ref = torch.matmul(x, w.T) + bias if bias is not None else torch.matmul(x, w.T)
    out, _ = OTFunctions.optical_compute_quantized_linear_fn(
        x,
        w,
        bias,
        x_min=-1.0,
        x_max=1.0,
        w_min=-1.0,
        w_max=1.0,
        w_lut_min=0.01,
        o_min=-1.0,
        o_max=1.0,
        q_levels=256,
        q_seed=0,
        skip_quantize=True,
    )
    assert torch.allclose(out, out_ref, atol=1e-2, rtol=0.0), f"Output mismatch: {out} vs {out_ref}"
    logger.info("Test passed: skip_quantize=True")


def test_optical_compute_quantized_linear_forward_fn():
    x = torch.rand(16, 32, device=DEVICE, dtype=torch.float16) * 2 - 1
    w = torch.rand(8, 32, device=DEVICE, dtype=torch.float16) * 2 - 1
    bias = torch.rand(8, device=DEVICE, dtype=torch.float16)

    out_ref = torch.matmul(x, w.T) + bias if bias is not None else torch.matmul(x, w.T)
    out, _ = OTFunctions.optical_compute_quantized_linear_fn(
        x,
        w,
        bias,
        x_min=-1.0,
        x_max=1.0,
        w_min=-1.0,
        w_max=1.0,
        w_lut_min=0.001,
        o_min=-10.0,
        o_max=10.0,
        q_levels=256,
        q_seed=0,
    )
    err = (out - out_ref).abs().mean().item()
    logger.info(f"Mean abs error: {err}")
    assert torch.allclose(out, out_ref, atol=0.1, rtol=0.0), f"Output mismatch: {out} vs {out_ref}"
    logger.info("Test passed: output is close to reference")


def test_optical_compute_quantized_linear_backward_fn():
    x = torch.rand(16, 32, device=DEVICE, dtype=torch.float16) * 2 - 1
    w = torch.rand(8, 32, device=DEVICE, dtype=torch.float16) * 2 - 1
    bias = torch.rand(8, device=DEVICE, dtype=torch.float16)
    w.requires_grad_()
    x.requires_grad_()
    bias.requires_grad_()

    out, _ = OTFunctions.optical_compute_quantized_linear_fn(
        x,
        w,
        bias,
        x_min=-1.0,
        x_max=1.0,
        w_min=-1.0,
        w_max=1.0,
        w_lut_min=0.001,
        o_min=-10.0,
        o_max=10.0,
        q_levels=256,
        q_seed=0,
    )
    loss = torch.sum(out)
    loss.backward()
    assert torch.allclose(x.grad, torch.ones((16, 8), device=DEVICE, dtype=torch.float16) @ w, atol=1e-2, rtol=0.0)
    logger.info("Test passed: x.grad is correct")


if __name__ == "__main__":
    set_logging_verbosity("info")
    test_optical_compute_quantized_forward_fn_simple()
    test_optical_compute_quantized_backward_fn_simple()
    test_optical_compute_quantized_linear_forward_fn_skip_quantize()
    test_optical_compute_quantized_linear_forward_fn()
    test_optical_compute_quantized_linear_backward_fn()
