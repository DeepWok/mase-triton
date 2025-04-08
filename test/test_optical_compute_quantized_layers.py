import logging

import pytest
import torch
import numpy as np

from mase_triton.optical_compute.layers import OpticalComputeLinear
from mase_triton.about import PACKAGE_NAME
from mase_triton.utils.deps import all_packages_are_available
from mase_triton.logging import set_logging_verbosity

DEVICE = "cuda"

logger = logging.getLogger(f"{PACKAGE_NAME}.test.{__name__}")


def test_optical_compute_quantized_linear_simple():
    in_features = 32
    out_features = 8
    fc1 = OpticalComputeLinear(
        in_features=in_features,
        out_features=out_features * 2,
        bias=False,
        device=DEVICE,
        dtype=torch.float32,
    )
    fc2 = OpticalComputeLinear(
        in_features=out_features * 2,
        out_features=out_features,
        bias=False,
        device=DEVICE,
        dtype=torch.float32,
    )
    x = torch.rand(8, in_features, device=DEVICE, dtype=torch.float32)
    x = x * 2 - 1
    x.requires_grad_()
    x = fc1(x)
    x = torch.relu(x)
    y = fc2(x)
    assert y.shape == (8, out_features)
    logger.info(f"{fc1}")
    loss = torch.sum(y)
    loss.backward()


def test_optical_compute_quantized_linear_forward_error():
    in_features = 32
    out_features = 8
    fc_baseline = torch.nn.Linear(in_features, out_features, bias=False)
    fc_optical = OpticalComputeLinear.from_linear(fc_baseline)
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
    logger.info(f"AbsError: {abs_error}")
    logger.info(f"ErrorNorm/Norm: {error}")

@pytest.mark.skipif(
    not all_packages_are_available(("tqdm",)), reason="Requires datasets and torchvision"
)
def test_optical_compute_quantized_linear_toy_training():
    from tqdm import tqdm
    in_features = 32
    hidden_size = 64
    out_features = 2

    num_epochs = 4
    batch_size = 64

    dtype = torch.float32
    device = DEVICE

    def get_data(num_batches, batch_size):
        # binary classification
        for _ in range(num_batches):
            x = torch.rand(batch_size, in_features, device=device, dtype=dtype) * 2 - 1
            y = (4 * x**3 - 2 * x ).sum(dim=1, keepdim=True)
            y = (y > 0).float()
            yield x, y

    class NetOptical(torch.nn.Module):
        def __init__(self, in_features, hidden_size, out_features):
            super().__init__()
            self.fc1 = OpticalComputeLinear(in_features=in_features, out_features=hidden_size, bias=True)
            self.fc2 = OpticalComputeLinear(in_features=hidden_size, out_features=out_features, bias=True)

        def forward(self, x):
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            return x

    net = NetOptical(in_features, hidden_size, out_features)
    net.to(dtype=dtype, device=device)
    optimizer = torch.optim.AdamW(net.parameters())
    for epoch in tqdm(range(num_epochs), total=num_epochs):
        net.train()
        for i, (x_batch, y_batch) in enumerate(get_data(num_batches=1000, batch_size=batch_size)):
            x_batch = x_batch.to(device).to(dtype)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            out = net(x_batch)
            out = out.sum(dim=1, keepdim=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y_batch)
            loss.backward(retain_graph=True)
            optimizer.step()

        # eval on train set
        if (epoch + 1) % 2 == 0:
            net.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (x_batch, y_batch) in enumerate(get_data(num_batches=1000, batch_size=batch_size)):
                    x_batch = x_batch.to(device).to(dtype)
                    y_batch = y_batch.to(device)
                    out = net(x_batch)
                    pred = torch.sigmoid(out)
                    pred = (pred > 0.5).float()
                    total += y_batch.size(0)
                    correct += (pred == y_batch).sum().item()
                acc = 100 * correct / total
                logger.info(f"Epoch: {epoch}, Train accuracy: {acc:.2f}%")
    logger.info(f"Training completed with accuracy: {acc:.2f}%")

if __name__ == "__main__":
    set_logging_verbosity("info")
    torch.autograd.set_detect_anomaly(True)
    # test_optical_compute_quantized_linear_simple()
    # test_optical_compute_quantized_linear_forward_error()
    test_optical_compute_quantized_linear_toy_training()
    # test_optical_compute_quantized_linear_mnist()
