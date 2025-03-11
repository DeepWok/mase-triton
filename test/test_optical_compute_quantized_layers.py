import logging

import pytest
import torch
import numpy as np

from mase_triton.optical_compute.core import optical_compute_quantized_forward_fn
from mase_triton.optical_compute.layers import OpticalComputeQLinear
from mase_triton.about import PACKAGE_NAME
from mase_triton.utils.deps import all_packages_are_available
from mase_triton.logging import set_logging_verbosity

DEVICE = "cuda"

logger = logging.getLogger(f"{PACKAGE_NAME}.test.{__name__}")


def test_optical_compute_quantized_linear_simple():
    in_features = 32
    out_features = 8
    fc = OpticalComputeQLinear(
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


@pytest.mark.skipif(not all_packages_are_available(("datasets", "torchvision", "tqdm")))
def test_optical_compute_quantized_linear_mnist():
    from datasets import load_dataset
    from torchvision.transforms import v2
    from tqdm import tqdm

    mnist = load_dataset("ylecun/mnist", split="train")
    in_features = 784
    intermediate_features = 256
    out_features = 10
    num_epochs = 10
    dtype = torch.bfloat16

    class Net(torch.nn.Module):
        def __init__(self, in_features, intermediate_features, out_features, dtype, device="cpu"):
            super().__init__()
            self.fc1 = OpticalComputeQLinear(
                in_features=in_features,
                out_features=intermediate_features,
                bias=False,
                device=device,
                dtype=dtype,
            )
            self.fc2 = OpticalComputeQLinear(
                in_features=intermediate_features,
                out_features=out_features,
                bias=False,
                device=device,
                dtype=dtype,
            )

        def forward(self, x):
            x = x.view(-1, 784)
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            return x

    class DataLoader:
        def __init__(self, mnist, batch_size):
            self.mnist = mnist
            self.batch_size = batch_size
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32),
                    v2.Normalize((0.1307,), (0.3081,)),
                ]
            )

        def __iter__(self):
            for i in range(0, len(self.mnist), self.batch_size):
                img_batch = self.mnist[i : i + self.batch_size]["image"]  # PIL image
                label_batch = self.mnist[i : i + self.batch_size]["label"]  # list of ints
                img_batch = np.asarray(img_batch).astype(np.float32)
                label_batch = np.asarray(label_batch).astype(np.int64)
                img_batch = self.transform(img_batch)
                img_batch = img_batch.reshape(self.batch_size, -1)
                label_batch = torch.tensor(label_batch)
                yield img_batch, label_batch

        def __len__(self):
            return len(self.mnist) // self.batch_size

    net = Net(in_features, intermediate_features, out_features, dtype, DEVICE)
    dataloader = DataLoader(mnist, batch_size=32)

    optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)
    prog_bar = tqdm(range(num_epochs * len(dataloader)), desc="Training", total=num_epochs * len(dataloader))
    net.to(DEVICE)
    for epoch in range(num_epochs):
        net.train()
        for i, (img_batch, label_batch) in enumerate(dataloader):
            img_batch = img_batch.to(DEVICE).to(dtype)
            label_batch = label_batch.to(DEVICE)
            optimizer.zero_grad()
            out = net(img_batch)
            loss = torch.nn.functional.cross_entropy(out, label_batch)
            loss.backward()
            optimizer.step()
            prog_bar.update(1)

        # eval on train set
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
        logger.info(f"Epoch {epoch}: Train accuracy: {acc:.2f}%")
    prog_bar.close()


if __name__ == "__main__":
    set_logging_verbosity("info")
    # test_optical_compute_quantized_linear_simple()
    test_optical_compute_quantized_linear_mnist()
