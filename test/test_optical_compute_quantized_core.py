import logging

import torch

from mase_triton.optical_compute.core import optical_compute_quantized_forward_fn
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

    out, seed_out = optical_compute_quantized_forward_fn(
        x,
        quant_levels=quant_levels,
        min_val=min_val,
        max_val=max_val,
        lut_min=lut_min,
        quant_mode="det",
        seed=seed,
    )
    assert seed_out == (seed + 1)
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
    out, seed_out = optical_compute_quantized_forward_fn(
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_optical_compute_quantized_forward_fn_simple()
    test_optical_compute_quantized_backward_fn_simple()
