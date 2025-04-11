import pytest
import torch

from mase_triton.optical_compute.layers import OpticalTransformerLinear
from mase_triton.utils.deps import all_packages_are_available
from mase_triton.logging import set_logging_verbosity, test_logger

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


@pytest.mark.skipif(not all_packages_are_available(("tqdm", "datasets")), reason="Requires tqdm and datasets")
def test_optical_compute_quantized_linear_toy_training():
    from tqdm import tqdm
    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset
    from mase_triton.utils.train_utils import set_seed

    set_seed(0)

    run_baseline = False
    in_features = 4
    hidden_size = 12
    out_features = 3
    lr = 1e-2

    num_epochs = 20
    batch_size = 32

    dtype = torch.float32
    device = DEVICE

    class IrisDataSet(Dataset):
        def __init__(self):
            self.dataset = load_dataset("scikit-learn/iris", split="train")
            self.feature_entries = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
            self.label_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            x = [self.dataset[idx][entry] for entry in self.feature_entries]
            y = self.label_map[self.dataset[idx]["Species"]]

            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            return x, y

    class NetOptical(torch.nn.Module):
        def __init__(self, in_features, hidden_size, out_features):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(in_features)
            self.fc1 = OpticalTransformerLinear(in_features=in_features, out_features=hidden_size, bias=True)

        def forward(self, x):
            x = self.bn(x)
            x = self.fc1(x)
            x = torch.relu(x)
            return x

    class Net(NetOptical):
        def __init__(self, in_features, hidden_size, out_features):
            super().__init__(in_features, hidden_size, out_features)
            self.fc1 = torch.nn.Linear(in_features, hidden_size, bias=True)

    if run_baseline:
        logger.info("Running baseline")
        net = Net(in_features, hidden_size, out_features)
    else:
        logger.info("Running ONN")
        net = NetOptical(in_features, hidden_size, out_features)

    net.to(dtype=dtype, device=device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    dataloader = DataLoader(dataset=IrisDataSet(), batch_size=batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()

    prog_bar = tqdm(range(num_epochs), total=num_epochs, desc="Training", unit="epoch")
    for epoch in prog_bar:
        prog_bar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # Validate the model
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in dataloader:
                x = x.to(device=device, dtype=dtype)
                y = y.to(device=device, dtype=torch.long)
                out = net(x)
                _, predicted = torch.max(out, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            accuracy = correct / total
            # logger.info(f"Epoch {epoch + 1}, Accuracy: {accuracy:.4f}")
            prog_bar.set_postfix({"accuracy": accuracy})


if __name__ == "__main__":
    set_logging_verbosity("info")
    test_optical_compute_quantized_linear_simple()
    # test_optical_compute_quantized_linear_forward_error()
    # test_optical_compute_quantized_linear_toy_training()
