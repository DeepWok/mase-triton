import pytest
import torch

import mase_triton.minifloat.fake as minifloat_fake
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
from mase_triton.utils.debug import set_ipdb_breakpoint
from mase_triton.utils.train_utils import set_seed

set_ipdb_breakpoint()
set_seed(42)


def minifloat_bin_to_float(bin: int, exp_bits: int, frac_bits: int) -> float:
    bin_str = f"{bin:0{exp_bits + frac_bits + 1}b}"
    sign = int(bin_str[0])
    exp = int(bin_str[1 : 1 + exp_bits], 2)
    exp_bias = (1 << (exp_bits - 1)) - 1
    is_subnormal = exp == 0 and int(bin_str[1 + exp_bits :], 2) != 0
    is_zero = exp == 0 and int(bin_str[1 + exp_bits :], 2) == 0

    if is_subnormal:
        exp = 1 - exp_bias
    else:
        exp -= exp_bias

    frac = int(bin_str[1 + exp_bits :], 2) / (1 << frac_bits)
    if not (is_subnormal or is_zero):
        frac += 1.0
    value = frac * (2**exp) if sign == 0 else -frac * (2**exp)
    return value


@pytest.mark.parametrize("n_elements", [1024])
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
def test_extract_component_builtin_meta(meta: MinifloatMeta, n_elements: int):
    device = "cuda"
    x = torch.randn(n_elements, dtype=torch.float32, device=device)
    x_q = minifloat_kernels.extract_minifloat_component(x, minifloat_meta=meta)
    x_q_ref = minifloat_fake.extract_minifloat_component(x, minifloat_meta=meta)
    assert x_q_ref.dtype == x_q.dtype
    assert x_q_ref.shape == x_q.shape
    err = x_q_ref.int() - x_q.int()
    assert (err != 0).sum() / n_elements <= 0.001
    assert err.abs().sum() <= 1


@pytest.mark.parametrize("n_elements", [1024])
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
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_compose_component_builtin_meta(
    dtype: str, meta: MinifloatMeta, n_elements: int
):
    device = "cuda"
    dtype = getattr(torch, dtype)
    x = torch.randn(n_elements, dtype=dtype, device=device)
    x_q = minifloat_kernels.extract_minifloat_component(x, minifloat_meta=meta)
    x_dq = minifloat_kernels.compose_minifloat_component(
        x_q, minifloat_meta=meta, output_dtype=dtype
    )
    x_dq_ref = minifloat_fake.compose_minifloat_component(
        x_q, minifloat_meta=meta, output_dtype=dtype
    )
    assert x_dq_ref.dtype == x_dq.dtype
    assert x_dq_ref.shape == x_dq.shape
    err = x_dq_ref.float() - x_dq.float()
    assert (err == 0.0).all()


@pytest.mark.parametrize(
    "meta", [FP4_E2M1_fn, FP6_E2M3_fn, FP6_E3M2_fn, FP8_E4M3_fn, FP8_E5M2_fn]
)
@pytest.mark.parametrize("n_elements", [8])
def test_extract_compose_builtin_meta_saturate(meta: MinifloatMeta, n_elements: int):
    device = "cuda"
    x = torch.ones(n_elements, dtype=torch.float32, device=device) * meta.max_normal * 2
    x_q = minifloat_kernels.extract_minifloat_component(x, minifloat_meta=meta)
    x_dq = minifloat_kernels.compose_minifloat_component(
        x_q, minifloat_meta=meta, output_dtype=torch.float32
    )
    assert (x_dq == meta.max_normal).all()


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_fp4_compose(dtype: str):
    dtype = getattr(torch, dtype)
    device = "cuda"
    # fmt: off
    x_raw = [0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0111,
             0b1000, 0b1001, 0b1010, 0b1011, 0b1100, 0b1101, 0b1110, 0b1111]
    # fmt: on
    x_fp4_ref_ref = [minifloat_bin_to_float(x, exp_bits=2, frac_bits=1) for x in x_raw]
    x_fp4_ref_ref = torch.tensor(x_fp4_ref_ref, dtype=dtype, device=device)
    x = torch.tensor(x_raw, dtype=torch.uint16, device=device)
    x_fp4_ref = minifloat_fake.compose_minifloat_component(
        x, minifloat_meta=FP4_E2M1_fn, output_dtype=dtype
    )
    x_fp4 = minifloat_kernels.compose_minifloat_component(
        x, minifloat_meta=FP4_E2M1_fn, output_dtype=dtype
    )
    assert (x_fp4 == x_fp4_ref).all(), f"Expected {x_fp4_ref}, \ngot {x_fp4}"
    assert (x_fp4 == x_fp4_ref_ref).all(), f"Expected {x_fp4_ref_ref}, \ngot {x_fp4}"


@pytest.mark.parametrize("n_elements", [1024])
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
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_extract_compose_builtin_meta(dtype: str, meta: MinifloatMeta, n_elements: int):
    device = "cuda"
    dtype = getattr(torch, dtype)
    x = torch.randn(n_elements, dtype=dtype, device=device)
    x_q = minifloat_kernels.extract_minifloat_component(x, minifloat_meta=meta)
    x_dq = minifloat_kernels.compose_minifloat_component(
        x_q, minifloat_meta=meta, output_dtype=dtype
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


@pytest.mark.slow
@pytest.mark.parametrize(
    "meta", [FP4_E2M1_fn, FP6_E2M3_fn, FP6_E3M2_fn, FP8_E4M3_fn, FP8_E5M2_fn]
)
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("n_elements", [1024 * 1024])
@pytest.mark.parametrize("enable", [True, False])
def test_autotune_cast(enable: bool, n_elements: int, dtype: str, meta: MinifloatMeta):
    dtype = getattr(torch, dtype)
    x = torch.randn(n_elements, dtype=dtype, device="cuda")
    if enable:
        KernelManager.enable_autotune()
    else:
        KernelManager.disable_autotune()
    x_q = minifloat_kernels.extract_minifloat_component(x, minifloat_meta=meta)
    _ = minifloat_kernels.compose_minifloat_component(
        x_q, minifloat_meta=meta, output_dtype=torch.float32
    )


if __name__ == "__main__":
    # test_extract_component_builtin_meta(FP6_E2M3_fn, 1024)
    test_compose_component_builtin_meta("float16", FP6_E2M3_fn, 1024)
    # test_fp4_compose()
    # test_autotune_cast(True, FP6_E2M3_fn)
