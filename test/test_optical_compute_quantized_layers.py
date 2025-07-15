import pytest
import torch

from mase_triton.logging import set_logging_verbosity, test_logger
from mase_triton.optical_compute.layers import OpticalTransformerLinear
from mase_triton.utils.deps import all_packages_are_available
from mase_triton.utils.train_utils import set_seed

set_seed(42)
DEVICE = "cuda"

logger = test_logger.getChild(f"{__name__}")


def test_optical_compute_quantized_linear_simple():
    in_features = 32
    out_features = 8
    fc1 = OpticalTransformerLinear(
        in_features=in_features,
        out_features=out_features * 2,
        bias=False,
        device=DEVICE,
        dtype=torch.float32,
    )
    fc2 = OpticalTransformerLinear(
        in_features=out_features * 2,
        out_features=out_features,
        bias=False,
        device=DEVICE,
        dtype=torch.float32,
    )
    fc1.train()
    fc2.train()
    x = torch.rand(2, 8, in_features, device=DEVICE, dtype=torch.float32)
    x = x * 2 - 1
    x.requires_grad_()
    x = fc1(x)
    x = torch.relu(x)
    y = fc2(x)
    assert y.shape == (2, 8, out_features)
    logger.info(f"{fc1}")
    loss = torch.sum(y)
    loss.backward()
    assert torch.all(torch.isfinite(fc1.weight.grad))


def test_optical_compute_quantized_linear_forward_error():
    in_features = 32
    out_features = 8
    fc_baseline = torch.nn.Linear(in_features, out_features, bias=False)
    fc_optical = OpticalTransformerLinear.from_linear(fc_baseline)
    x = torch.rand(8, in_features, device=DEVICE, dtype=torch.float32)
    x = x * 2 - 1
    fc_baseline.to(DEVICE)
    fc_optical.to(DEVICE)
    with torch.no_grad():
        y_baseline = fc_baseline(x)
        y_optical = fc_optical(x)
        abs_error = torch.abs(y_baseline - y_optical)
        error = torch.norm(abs_error) / torch.norm(y_baseline)
        assert error < 0.05
    logger.info(f"ErrorNorm/Norm: {error}")
    logger.info("Test passed: output is close to reference")


if __name__ == "__main__":
    set_logging_verbosity("info")
    test_optical_compute_quantized_linear_simple()
    test_optical_compute_quantized_linear_forward_error()
