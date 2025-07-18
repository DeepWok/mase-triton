import pytest
import torch

import mase_triton.minifloat.fake as minifloat_fake
import mase_triton.minifloat.functional as minifloat_functional
import mase_triton.minifloat.kernels as minifloat_kernels
from mase_triton.manager import KernelManager
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
