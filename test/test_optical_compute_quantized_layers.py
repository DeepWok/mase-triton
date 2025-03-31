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
    fc = OpticalComputeLinear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        device=DEVICE,
        dtype=torch.float32,
    )
    x = torch.rand(8, in_features, device=DEVICE, dtype=torch.float32)
    x = x * 2 - 1
    x.requires_grad_()
    y = fc(x)
    assert y.shape == (8, out_features)
    logger.info(f"{fc}")
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
    not all_packages_are_available(("datasets", "torchvision", "tqdm")), reason="Requires datasets and torchvision"
)
def test_optical_compute_quantized_linear_mnist():
    from datasets import load_dataset
    from torchvision.transforms import v2
    from tqdm import tqdm

    mnist = load_dataset("ylecun/mnist", split="train")
    mnist = mnist.shuffle(seed=42)
    in_features = 784
    hidden_size_1 = 200
    hidden_size_2 = 100
    out_features = 10
    num_epochs = 60
    batch_size = 128
    dtype = torch.bfloat16

    class NetOptical(torch.nn.Module):
        def __init__(self, in_features, hidden_size_1, hidden_size_2, out_features):
            super().__init__()
            self.fc1 = OpticalComputeLinear(in_features=in_features, out_features=hidden_size_1, bias=True)
            self.fc2 = OpticalComputeLinear(in_features=hidden_size_1, out_features=hidden_size_2, bias=True)
            self.fc3 = OpticalComputeLinear(in_features=hidden_size_2, out_features=out_features, bias=True)

        def forward(self, x):
            x = x.view(-1, 784)
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            x = torch.relu(x)
            x = self.fc3(x)
            return x

    class DataLoader:
        def __init__(self, mnist, batch_size, device):
            self.mnist = mnist
            self.batch_size = batch_size
            self.device = device
            self.transform = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize((0.1307,), (0.3081,))])

        def __iter__(self):
            for i in range(0, len(self.mnist), self.batch_size):
                img_batch = self.mnist[i : i + self.batch_size]["image"]  # PIL image
                label_batch = self.mnist[i : i + self.batch_size]["label"]  # list of ints
                img_batch = torch.tensor(np.asarray(img_batch).astype(np.float32), device=self.device)
                label_batch = torch.tensor(np.asarray(label_batch).astype(np.int64), device=self.device)
                img_batch = self.transform(img_batch)
                img_batch = img_batch.reshape(self.batch_size, -1)
                yield img_batch, label_batch

        def __len__(self):
            return len(self.mnist) // self.batch_size

    net = NetOptical(in_features, hidden_size_1, hidden_size_2, out_features)
    # net = NetBaseline(in_features, hidden_size_1, hidden_size_2, out_features)
    net.to(dtype=dtype, device=DEVICE)
    dataloader = DataLoader(mnist, batch_size=batch_size, device=DEVICE)

    optimizer = torch.optim.AdamW(net.parameters())
    for epoch in tqdm(range(num_epochs), total=num_epochs):
        net.train()
        for i, (img_batch, label_batch) in enumerate(dataloader):
            img_batch = img_batch.to(DEVICE).to(dtype)
            label_batch = label_batch.to(DEVICE)
            optimizer.zero_grad()
            out = net(img_batch)
            loss = torch.nn.functional.cross_entropy(out, label_batch)
            loss.backward()
            optimizer.step()

        # eval on train set
        if (epoch + 1) % 20 == 0:
            net.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (img_batch, label_batch) in enumerate(dataloader):
                    img_batch = img_batch.to(DEVICE).to(dtype)
                    label_batch = label_batch.to(DEVICE)
                    out = net(img_batch)
                    _, predicted = torch.max(out, 1)
                    total += label_batch.size(0)
                    correct += (predicted == label_batch).sum().item()
                acc = 100 * correct / total
                logger.info(f"Epoch: {epoch}, Train accuracy: {acc:.2f}%")
            if epoch == num_epochs - 1:
                assert acc > 95.0


if __name__ == "__main__":
    set_logging_verbosity("info")
    # test_optical_compute_quantized_linear_simple()
    # test_optical_compute_quantized_linear_forward_error()
    test_optical_compute_quantized_linear_mnist()
