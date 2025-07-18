import pytest
import torch

from mase_triton.logging import set_logging_verbosity, test_logger
from mase_triton.optical_compute import OpticalTransformerFunctions as OTFunctions
from mase_triton.utils.deps import all_packages_are_available
from mase_triton.utils.train_utils import set_seed

set_seed(42)
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

    out, seed_out = OTFunctions.quantize_fn(
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


def test_optical_compute_quantized_backward_fn_simple():
    quant_levels = 256
    min_val = -1.0
    max_val = 1.0
    lut_min = 0.01
    seed = 0

    x = torch.rand(256, device=DEVICE, dtype=torch.float32)
    x = x * 2 - 1
    x.requires_grad_()
    out, seed_out = OTFunctions.quantize_fn(
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


def test_optical_compute_quantized_linear_forward_fn_skip_quantize():
    x = torch.rand(16, 32, device=DEVICE, dtype=torch.float16)
    w = torch.rand(8, 32, device=DEVICE, dtype=torch.float16)
    bias = torch.rand(8, device=DEVICE, dtype=torch.float16)

    out_ref = torch.matmul(x, w.T) + bias if bias is not None else torch.matmul(x, w.T)
    out, _ = OTFunctions.quantized_linear_fn(
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


def test_optical_compute_quantized_linear_forward_fn():
    x = torch.rand(16, 32, device=DEVICE, dtype=torch.float16) * 2 - 1
    w = torch.rand(8, 32, device=DEVICE, dtype=torch.float16) * 2 - 1
    bias = torch.rand(8, device=DEVICE, dtype=torch.float16)

    out_ref = torch.matmul(x, w.T) + bias if bias is not None else torch.matmul(x, w.T)
    out, _ = OTFunctions.quantized_linear_fn(
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


def test_optical_compute_quantized_linear_backward_fn():
    x = torch.rand(16, 32, device=DEVICE, dtype=torch.float16) * 2 - 1
    w = torch.rand(8, 32, device=DEVICE, dtype=torch.float16) * 2 - 1
    bias = torch.rand(8, device=DEVICE, dtype=torch.float16)
    w.requires_grad_()
    x.requires_grad_()
    bias.requires_grad_()

    out, _ = OTFunctions.quantized_linear_fn(
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
        torch.ones((16, 8), device=DEVICE, dtype=torch.float16) @ w,
        atol=1e-2,
        rtol=0.0,
    )
    logger.info("Test passed: x.grad is correct")


def test_optical_compute_quantized_bmm_forward_fn_skip_quantize():
    a = torch.rand(8, 4, 32, 64, device=DEVICE, dtype=torch.float16) * 2 - 1
    b = torch.rand(8, 4, 64, 16, device=DEVICE, dtype=torch.float16) * 2 - 1

    out, _ = OTFunctions.quantized_matmul_fn(
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


def test_optical_compute_quantized_bmm_forward_fn():
    a = torch.rand(8, 4, 32, 64, device=DEVICE, dtype=torch.float16) * 2 - 1
    b = torch.rand(8, 4, 64, 16, device=DEVICE, dtype=torch.float16) * 2 - 1
    out_ref = torch.matmul(a, b)

    out, _ = OTFunctions.quantized_matmul_fn(
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


def test_optical_compute_quantized_bmm_backward_fn():
    a = torch.rand(8, 4, 32, 64, device=DEVICE, dtype=torch.float16) * 2 - 1
    b = torch.rand(8, 4, 64, 16, device=DEVICE, dtype=torch.float16) * 2 - 1
    a.requires_grad_()
    b.requires_grad_()

    out, _ = OTFunctions.quantized_matmul_fn(
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


@pytest.mark.skipif(
    not all_packages_are_available(("tqdm",)),
    reason="Requires tqdm and datasets",
)
def test_optical_bmm_toy_training():
    from tqdm import tqdm

    batch_size = 32
    seq_len = 8
    in_features = 32
    n_heads = 4
    head_dim = 8
    dtype = torch.float32
    device = DEVICE

    def gen_data(batch_size, seq_len, in_features):
        a = (
            torch.rand((batch_size * seq_len, in_features), device=DEVICE, dtype=dtype)
            * 2
            - 1
        )
        b = (
            torch.rand((batch_size * seq_len, in_features), device=DEVICE, dtype=dtype)
            * 2
            - 1
        )
        for i in range(10):
            yield a, b

    class Net(torch.nn.Module):
        def __init__(self, in_features, n_heads, seq_len, head_dim):
            super().__init__()
            self.in_features = in_features
            self.n_heads = n_heads
            self.seq_len = seq_len
            self.head_dim = head_dim
            self.fc = torch.nn.Linear(
                in_features, n_heads * head_dim, bias=False, dtype=dtype
            )
            self.seed = 0

        def forward(self, x1, x2):
            x1 = self.fc(x1)
            x1 = x1.reshape(-1, self.seq_len, self.n_heads, self.head_dim)
            x1 = x1.permute(0, 2, 1, 3)

            x2 = self.fc(x2)
            x2 = x2.reshape(-1, self.seq_len, self.n_heads, self.head_dim)
            x2 = x2.permute(0, 2, 1, 3)
            x1 = x1.contiguous()
            x2 = x2.contiguous()

            y, self.seed = OTFunctions.quantized_matmul_fn(
                x1,
                x2,
                a_min=-1.0,
                a_max=1.0,
                b_min=-1.0,
                b_max=1.0,
                b_lut_min=0.001,
                o_min=-10.0,
                o_max=10.0,
                q_levels=256,
                q_seed=self.seed,
                skip_quantize=False,
            )
            y = torch.sum(y, dim=(1, 2, 3))

            return y

    net = Net(in_features, n_heads, seq_len, head_dim).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    for x1, x2 in tqdm(gen_data(batch_size, seq_len, in_features), total=10):
        x1 = x1.to(device)
        x2 = x2.to(device)

        optimizer.zero_grad()
        y = net(x1, x2)
        loss = torch.sum(y)
        loss.backward()
        optimizer.step()

    logger.info("Test passed: back propagation completed successfully")


if __name__ == "__main__":
    set_logging_verbosity("info")
    # test_optical_compute_quantized_forward_fn_simple()
    test_optical_compute_quantized_backward_fn_simple()
    # test_optical_compute_quantized_linear_forward_fn_skip_quantize()
    # test_optical_compute_quantized_linear_forward_fn()
    test_optical_compute_quantized_linear_backward_fn()
    # test_optical_compute_quantized_bmm_forward_fn_skip_quantize()
    # test_optical_compute_quantized_bmm_forward_fn()
    test_optical_compute_quantized_bmm_backward_fn()
    # test_optical_bmm_toy_training()
