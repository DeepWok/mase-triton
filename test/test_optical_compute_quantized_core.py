import logging

import torch

from mase_triton.optical_compute.core import optical_compute_quantize_fn, optical_compute_quantized_matmul_fn
from mase_triton.about import PACKAGE_NAME

DEVICE = "cuda"

logger = logging.getLogger(f"{PACKAGE_NAME}.test.{__name__}")


def test_optical_compute_quantized_forward_fn_simple():
    x = torch.rand(8, device=DEVICE, dtype=torch.float32)
    x = x * 2 - 1
    quant_levels = 256
    min_val = -1.0
    max_val = 1.0
    lut_min = 0.01
    seed = 0

    out, seed_out = optical_compute_quantize_fn(
        x,
        quant_levels=quant_levels,
        min_val=min_val,
        max_val=max_val,
        lut_min=lut_min,
        quant_mode="det",
        seed=seed,
    )
    assert (out - x).abs().max().item() < 1 / quant_levels

    logger.info(f"x:\n{x}")
    logger.info(f"out:\n{out}")


def test_optical_compute_quantized_backward_fn_simple():
    quant_levels = 256
    min_val = -1.0
    max_val = 1.0
    lut_min = 0.01
    seed = 0

    x = torch.rand(256, device=DEVICE, dtype=torch.float32)
    x = x * 2 - 1
    x.requires_grad_()
    out, seed_out = optical_compute_quantize_fn(
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


def test_optical_compute_quantized_matmul_forward_fn_skip_quantize():
    x = torch.rand(16, 32, device=DEVICE, dtype=torch.float16)
    w = torch.rand(32, 8, device=DEVICE, dtype=torch.float16)
    bias = torch.rand(8, device=DEVICE, dtype=torch.float16)

    out_ref = torch.matmul(x, w) + bias if bias is not None else torch.matmul(x, w)
    out, _ = optical_compute_quantized_matmul_fn(
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


def test_optical_compute_quantized_matmul_forward_fn():
    x = torch.rand(16, 32, device=DEVICE, dtype=torch.float16) * 2 - 1
    w = torch.rand(32, 8, device=DEVICE, dtype=torch.float16) * 2 - 1
    bias = torch.rand(8, device=DEVICE, dtype=torch.float16)

    out_ref = torch.matmul(x, w) + bias if bias is not None else torch.matmul(x, w)
    out = optical_compute_quantized_matmul_fn(
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
    logger.info("Test passed")


def test_optical_compute_quantized_matmul_backward_fn():
    x = torch.rand(16, 32, device=DEVICE, dtype=torch.float16) * 2 - 1
    w = torch.rand(32, 8, device=DEVICE, dtype=torch.float16) * 2 - 1
    bias = torch.rand(8, device=DEVICE, dtype=torch.float16)
    w.requires_grad_()
    x.requires_grad_()
    bias.requires_grad_()

    out = optical_compute_quantized_matmul_fn(
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
    assert torch.allclose(x.grad, torch.ones((16, 8), device=DEVICE, dtype=torch.float16) @ w.T, atol=1e-2, rtol=0.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_optical_compute_quantized_forward_fn_simple()
    # test_optical_compute_quantized_backward_fn_simple()
    # test_optical_compute_quantized_matmul_forward_fn_skip_quantize()
    # test_optical_compute_quantized_matmul_forward_fn()
    test_optical_compute_quantized_matmul_backward_fn()
