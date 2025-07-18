import pytest
import torch

from mase_triton.minifloat.functional import (
    compose_minifloat_tensor,
    extract_minifloat_components,
    flatten_for_quantize,
    minifloat_matmul,
    permute_for_dequantize,
    quantize_dequantize,
)
from mase_triton.utils.train_utils import set_seed

set_seed(0)


@pytest.mark.parametrize("block_dim", [0, 1, 2, -1, -2, -3])
def test_minifloat_components(block_dim: int):
    x = torch.arange(24).reshape(2, 3, 4)

    x_flatten = flatten_for_quantize(x, block_dim)
    x_restore = permute_for_dequantize(
        x_flatten,
        block_dim=block_dim,
        ori_shape=x.shape,
    )

    assert x_restore.shape == x.shape
    assert torch.all(x_restore == x)



def test_quantize_dequantize_1d():
    n_elements = 1024
    w = torch.randn(n_elements, dtype=torch.bfloat16, device="cuda")
    elements, tensor_meta = extract_minifloat_components(w, block_dim=0)
    w_dq = compose_minifloat_tensor(elements, tensor_meta=tensor_meta)
    assert w_dq.shape == w.shape, (
        f"Dequantized tensor shape {w_dq.shape} does not match original shape {w.shape}."
    )
    avg_err = (w - w_dq).abs().mean()
    avg_err_ratio = avg_err / w.abs().mean()
    assert avg_err_ratio < 0.7, (
        f"Average error ratio {avg_err_ratio} is too high for {minifloat_format} format."
    )


def test_quantize_dequantize_1d_wrapped():
    n_elements = 1024
    w = torch.randn(n_elements, dtype=torch.bfloat16, device="cuda")
    w_dq = quantize_dequantize(w, block_dim=0)
    assert w_dq.shape == w.shape, (
        f"Dequantized tensor shape {w_dq.shape} does not match original shape {w.shape}."
    )
    avg_err = (w - w_dq).abs().mean()
    avg_err_ratio = avg_err / w.abs().mean()
    assert avg_err_ratio < 0.7, (
        f"Average error ratio {avg_err_ratio} is too high for {minifloat_format} format."
    )


@pytest.mark.parametrize("block_dim", [0, 1, -1])
def test_quantize_dequantize_2d(block_dim: int):
    n_elements = 1024 * 3
    w = torch.randn(n_elements, dtype=torch.bfloat16, device="cuda")

    if block_dim % 2 == 0:
        w = w.reshape(-1, 3)
    else:
        w = w.reshape(3, -1)

    elements, tensor_meta = extract_minifloat_components(w, block_dim=block_dim)
    w_dq = compose_minifloat_tensor(elements, tensor_meta=tensor_meta)
    assert w_dq.shape == w.shape, (
        f"Dequantized tensor shape {w_dq.shape} does not match original shape {w.shape}."
    )
    avg_err = (w - w_dq).abs().mean()
    avg_err_ratio = avg_err / w.abs().mean()
    assert avg_err_ratio < 0.65, (
        f"Average error ratio {avg_err_ratio} is too high for {minifloat_format} format."
    )



# if __name__ == "__main__":


