import pytest
import torch

from mase_triton.logging import set_logging_verbosity, test_logger
from mase_triton.optical_compute.core.optical_transformer import fake as ot_fake
from mase_triton.utils.deps import all_packages_are_available
from mase_triton.utils.train_utils import set_seed

set_seed(42)

logger = test_logger.getChild(f"{__name__}")


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_optical_compute_quantized_forward_fn_simple(device):
    x = torch.rand(8, device=device, dtype=torch.float32)
    x = x * 2 - 1
    quant_levels = 256
    min_val = -1.0
    max_val = 1.0
    lut_min = 0.01
    seed = 0

    out, seed_out = ot_fake.quantize_fn(
        x,
        quant_levels=quant_levels,
        min_val=min_val,
        max_val=max_val,
        lut_min=lut_min,
        quant_mode="det",
        seed=seed,
    )
    max_err = (out - x).abs().max().item()
    assert max_err < 1 / quant_levels * 2

    logger.info("Test passed: output is close to input")


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_optical_compute_quantized_backward_fn_simple(device):
    quant_levels = 256
    min_val = -1.0
    max_val = 1.0
    lut_min = 0.01
    seed = 0

    x = torch.rand(256, device=device, dtype=torch.float32)
    x = x * 2 - 1
    x.requires_grad_()
    out, seed_out = ot_fake.quantize_fn(
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
    logger.info("Identical gradients test passed")


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_optical_compute_quantized_linear_forward_fn_skip_quantize(device):
    x = torch.rand(16, 32, device=device, dtype=torch.float16)
    w = torch.rand(8, 32, device=device, dtype=torch.float16)
    bias = torch.rand(8, device=device, dtype=torch.float16)

    out_ref = torch.matmul(x, w.T) + bias if bias is not None else torch.matmul(x, w.T)
    out, _ = ot_fake.qlinear_fn(
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
    assert torch.allclose(out, out_ref, atol=1e-2, rtol=0.0), (
        f"Output mismatch: {out} vs {out_ref}"
    )
    logger.info("Test passed: skip_quantize=True")


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_optical_compute_quantized_linear_forward_fn(device):
    x = torch.rand(16, 32, device=device, dtype=torch.float16) * 2 - 1
    w = torch.rand(8, 32, device=device, dtype=torch.float16) * 2 - 1
    bias = torch.rand(8, device=device, dtype=torch.float16)

    out_ref = torch.matmul(x, w.T) + bias if bias is not None else torch.matmul(x, w.T)
    out, _ = ot_fake.qlinear_fn(
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
    assert torch.allclose(out, out_ref, atol=0.1, rtol=0.0), (
        f"Output mismatch: {out} vs {out_ref}"
    )
    logger.info("Test passed: output is close to reference")


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_optical_compute_quantized_linear_backward_fn(device):
    x = torch.rand(16, 32, device=device, dtype=torch.float16) * 2 - 1
    w = torch.rand(8, 32, device=device, dtype=torch.float16) * 2 - 1
    bias = torch.rand(8, device=device, dtype=torch.float16)
    w.requires_grad_()
    x.requires_grad_()
    bias.requires_grad_()

    out, _ = ot_fake.qlinear_fn(
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
    assert torch.allclose(
        x.grad,
        torch.ones((16, 8), device=device, dtype=torch.float16) @ w,
        atol=1e-2,
        rtol=0.0,
    )
    logger.info("Test passed: x.grad is correct")


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_optical_compute_quantized_bmm_forward_fn_skip_quantize(device):
    a = torch.rand(8, 4, 32, 64, device=device, dtype=torch.float16) * 2 - 1
    b = torch.rand(8, 4, 64, 16, device=device, dtype=torch.float16) * 2 - 1

    out, _ = ot_fake.qmatmul_fn(
        a,
        b,
        a_min=-1.0,
        a_max=1.0,
        b_min=-1.0,
        b_max=1.0,
        b_lut_min=0.001,
        o_min=-10.0,
        o_max=10.0,
        q_levels=256,
        q_seed=0,
        skip_quantize=True,
    )
    out_ref = torch.matmul(a, b)

    assert torch.allclose(out, out_ref, atol=1e-2, rtol=0.0), (
        f"Output mismatch: {out} vs {out_ref}"
    )
    logger.info("Test passed: skip_quantize=True")


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_optical_compute_quantized_bmm_forward_fn(device):
    a = torch.rand(8, 4, 32, 64, device=device, dtype=torch.float16) * 2 - 1
    b = torch.rand(8, 4, 64, 16, device=device, dtype=torch.float16) * 2 - 1
    out_ref = torch.matmul(a, b)

    out, _ = ot_fake.qmatmul_fn(
        a,
        b,
        a_min=-1.0,
        a_max=1.0,
        b_min=-1.0,
        b_max=1.0,
        b_lut_min=0.001,
        o_min=-10.0,
        o_max=10.0,
        q_levels=256,
        q_seed=0,
    )

    err = (out - out_ref).abs().mean().item()
    logger.info(f"Mean abs error: {err}")
    close = torch.isclose(out, out_ref, atol=0.15, rtol=0.0)
    close_ratio = close.sum() / close.numel()
    logger.info(f"Close ratio: {close_ratio.item()}")
    assert close_ratio > 0.99, f"Close ratio: {close_ratio.item()}"
    logger.info("Test passed: output is close to reference")


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_optical_compute_quantized_bmm_backward_fn(device):
    a = torch.rand(8, 4, 32, 64, device=device, dtype=torch.float16) * 2 - 1
    b = torch.rand(8, 4, 64, 16, device=device, dtype=torch.float16) * 2 - 1
    a.requires_grad_()
    b.requires_grad_()

    out, _ = ot_fake.qmatmul_fn(
        a,
        b,
        a_min=-1.0,
        a_max=1.0,
        b_min=-1.0,
        b_max=1.0,
        b_lut_min=0.001,
        o_min=-10.0,
        o_max=10.0,
        q_levels=256,
        q_seed=0,
    )
    loss = torch.sum(out)
    loss.backward()

    a_ref = a.clone().detach().requires_grad_()
    b_ref = b.clone().detach().requires_grad_()
    out_ref = torch.matmul(a_ref, b_ref)
    loss_ref = torch.sum(out_ref)
    loss_ref.backward()

    assert torch.allclose(a.grad, a_ref.grad)
    assert torch.allclose(b.grad, b_ref.grad)
    logger.info("Test passed: gradients are correct")
