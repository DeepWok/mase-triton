import pytest
import torch

import mase_triton.minifloat.fake as minifloat_fake
import mase_triton.minifloat.functional as minifloat_functional
import mase_triton.minifloat.kernels as minifloat_kernels
from mase_triton.manager import KernelManager
from mase_triton.minifloat.functional import minifloat_matmul
from mase_triton.minifloat.meta import (
    FP4_E2M1_fn,
    FP6_E2M3_fn,
    FP6_E3M2_fn,
    FP8_E4M3_fn,
    FP8_E5M2_fn,
    MinifloatMeta,
)
from mase_triton.utils.bit_repr import get_binary_repr, get_binary_repr_fp32
from mase_triton.utils.debug import set_ipdb_breakpoint
from mase_triton.utils.train_utils import set_seed

set_ipdb_breakpoint()
set_seed(42)


@pytest.mark.parametrize("n_elements", [1024])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize(
    "meta",
    [
        FP8_E4M3_fn,
        FP8_E5M2_fn,
        FP6_E2M3_fn,
        FP6_E3M2_fn,
        FP4_E2M1_fn,
    ],
)
def test_extract_compose_builtin_meta(
    meta: MinifloatMeta, dtype: str, device: str, n_elements: int
):
    dtype = getattr(torch, dtype)
    x = torch.randn(n_elements, dtype=dtype, device=device)
    x_q, tensor_meta = minifloat_functional.extract_minifloat_component(
        x, minifloat_meta=meta
    )
    x_dq = minifloat_functional.compose_minifloat_component(
        x_q, tensor_meta=tensor_meta, output_dtype=dtype
    )
    err = (x - x_dq).abs().mean()
    err_ratio = (err / x.abs().mean()).item()
    print(f"Average error ratio for {meta}: {err_ratio:.4f}")
    if meta is FP8_E4M3_fn:
        assert err_ratio < 0.1, f"Error ratio {err_ratio:.4f} is too high for {meta}"
    elif meta is FP8_E5M2_fn:
        assert err_ratio < 0.1, f"Error ratio {err_ratio:.4f} is too high for {meta}"
    elif meta is FP6_E3M2_fn:
        assert err_ratio < 0.2, f"Error ratio {err_ratio:.4f} is too high for {meta}"
    elif meta is FP6_E2M3_fn:
        assert err_ratio < 0.3, f"Error ratio {err_ratio:.4f} is too high for {meta}"
    elif meta is FP4_E2M1_fn:
        assert err_ratio < 0.5, f"Error ratio {err_ratio:.4f} is too high for {meta}"
    else:
        raise ValueError(f"Unknown minifloat meta: {meta}")


@pytest.mark.parametrize("n_elements", [1024])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize(
    "meta",
    [
        FP8_E4M3_fn,
        FP8_E5M2_fn,
        FP6_E2M3_fn,
        FP6_E3M2_fn,
        FP4_E2M1_fn,
    ],
)
def test_quantize_dequantize_builtin_meta(
    meta: MinifloatMeta, dtype: str, device: str, n_elements: int
):
    dtype = getattr(torch, dtype)
    x = torch.randn(n_elements, dtype=dtype, device=device)
    x_dq = minifloat_functional.quantize_dequantize(
        x, minifloat_meta=meta, output_dtype=dtype
    )
    err = (x - x_dq).abs().mean()
    err_ratio = (err / x.abs().mean()).item()
    print(f"Average error ratio for {meta}: {err_ratio:.4f}")
    if meta is FP8_E4M3_fn:
        assert err_ratio < 0.1, f"Error ratio {err_ratio:.4f} is too high for {meta}"
    elif meta is FP8_E5M2_fn:
        assert err_ratio < 0.1, f"Error ratio {err_ratio:.4f} is too high for {meta}"
    elif meta is FP6_E3M2_fn:
        assert err_ratio < 0.2, f"Error ratio {err_ratio:.4f} is too high for {meta}"
    elif meta is FP6_E2M3_fn:
        assert err_ratio < 0.3, f"Error ratio {err_ratio:.4f} is too high for {meta}"
    elif meta is FP4_E2M1_fn:
        assert err_ratio < 0.5, f"Error ratio {err_ratio:.4f} is too high for {meta}"
    else:
        raise ValueError(f"Unknown minifloat meta: {meta}")


@pytest.mark.parametrize(
    "x_meta",
    [
        None,
        FP8_E4M3_fn,
        FP8_E5M2_fn,
    ],
)
@pytest.mark.parametrize(
    "y_meta",
    [
        None,
        FP8_E4M3_fn,
        FP8_E5M2_fn,
    ],
)
@pytest.mark.parametrize("device", [torch.device("cuda"), torch.device("cpu")])
@pytest.mark.parametrize("backend", ["separate"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_minifloat_matmul(
    dtype: str, x_meta: MinifloatMeta, y_meta: MinifloatMeta, device, backend: str
):
    dtype = getattr(torch, dtype)
    func_type = ""
    if x_meta is not None:
        func_type += "Xq"
    else:
        func_type += "X"
    if y_meta is not None:
        func_type += "Wq"
    else:
        func_type += "W"

    a = torch.randn((2, 4, 512, 256), dtype=dtype, device=device) * 10
    b = torch.randn((2, 4, 256, 128), dtype=dtype, device=device) * 10

    y_ref = torch.matmul(a, b)
    y = minifloat_matmul(
        a, b, input_meta=x_meta, other_meta=y_meta, func_type=func_type, backend=backend
    )

    assert y.shape == y_ref.shape, (
        f"Output shape {y.shape} does not match reference shape {y_ref.shape}."
    )
    avg_err = (y - y_ref).abs().mean()
    avg_err_ratio = avg_err / y_ref.abs().mean()
    print(
        f"Average error ratio for {func_type} with {x_meta} and {y_meta}: {avg_err_ratio:.4f}"
    )
    if x_meta is None and y_meta is None:
        assert avg_err_ratio == 0
    elif x_meta is FP8_E4M3_fn and y_meta is FP8_E4M3_fn:
        assert avg_err_ratio < 0.2, (
            f"Average error ratio {avg_err_ratio} is too high for {func_type}."
        )
    else:
        assert avg_err_ratio < 0.3, (
            f"Average error ratio {avg_err_ratio} is too high for {func_type}."
        )
