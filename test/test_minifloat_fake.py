import pytest
import torch

from mase_triton.minifloat.fake import (
    compose_minifloat_component,
    extract_minifloat_component,
)
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


@pytest.mark.parametrize("meta", [FP4_E2M1_fn, FP6_E2M3_fn, FP6_E3M2_fn, FP8_E4M3_fn])
def test_builtin_meta(meta: MinifloatMeta):
    meta_finfo = {
        FP4_E2M1_fn: (6.0, 1.0, 0.5, 0.5),
        FP6_E2M3_fn: (7.5, 1.0, 0.875, 0.125),
        FP6_E3M2_fn: (28.0, 0.25, 0.1875, 0.0625),
        FP8_E4M3_fn: (480.0, 2 ** (-6), 2 ** (-6) * 0.875, 2 ** (-9)),
    }
    max_normal, min_normal, max_subnormal, min_subnormal = meta_finfo[meta]
    assert meta.max_normal == max_normal
    assert meta.min_normal == min_normal
    assert meta.max_subnormal == max_subnormal
    assert meta.min_subnormal == min_subnormal


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_fp4_compose(device: str):
    # fmt: off
    x_raw = [0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0111,
             0b1000, 0b1001, 0b1010, 0b1011, 0b1100, 0b1101, 0b1110, 0b1111]
    # fmt: on
    x_fp4_ref = [minifloat_bin_to_float(x, exp_bits=2, frac_bits=1) for x in x_raw]
    x = torch.tensor(x_raw, dtype=torch.uint16, device=device)
    x_fp4 = compose_minifloat_component(
        x, minifloat_meta=FP4_E2M1_fn, output_dtype=torch.float32
    )
    x_fp4_ref = torch.tensor(x_fp4_ref, dtype=torch.float32, device=device)
    assert (x_fp4 == x_fp4_ref).all(), f"Expected {x_fp4_ref}, \ngot {x_fp4}"


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
    x_q = extract_minifloat_component(x, minifloat_meta=meta)
    x_dq = compose_minifloat_component(x_q, minifloat_meta=meta, output_dtype=dtype)
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
    "meta", [FP4_E2M1_fn, FP6_E2M3_fn, FP6_E3M2_fn, FP8_E4M3_fn, FP8_E5M2_fn]
)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("n_elements", [8])
def test_extract_compose_builtin_meta_saturate(
    meta: MinifloatMeta, device: str, n_elements: int
):
    x = torch.ones(n_elements, dtype=torch.float32, device=device) * meta.max_normal * 2
    x_q = extract_minifloat_component(x, minifloat_meta=meta)
    x_dq = compose_minifloat_component(
        x_q, minifloat_meta=meta, output_dtype=torch.float32
    )
    assert (x_dq == meta.max_normal).all()


@pytest.mark.parametrize("n_elements", [1024])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("is_finite", [True, False])
@pytest.mark.parametrize("exp_frac_bits", [(2, 1), (2, 3), (3, 2), (4, 3), (5, 2)])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_extract_compose_random_meta(
    dtype: str,
    exp_frac_bits: tuple[int, int],
    is_finite: bool,
    device: str,
    n_elements: int,
):
    dtype = getattr(torch, dtype)
    if exp_frac_bits in [(2, 1), (2, 3)] and not is_finite:
        pytest.skip("FP4_E2M1 is always finite, skipping test for non-finite case")
    meta = MinifloatMeta(
        exp_bits=exp_frac_bits[0],
        frac_bits=exp_frac_bits[1],
        is_finite=is_finite,
        round_mode="rn",
    )
    x = torch.randn(n_elements, dtype=dtype, device=device) * 2.5
    x_q = extract_minifloat_component(x, minifloat_meta=meta)
    x_dq = compose_minifloat_component(x_q, minifloat_meta=meta, output_dtype=dtype)
    error = (x - x_dq).abs().mean()
    error_ratio = (error / x.abs().mean()).item()
    print(f"Average error ratio for {meta}: {error_ratio:.4f}")
    if meta.n_bits == 4:
        assert error_ratio < 0.5
    elif meta.n_bits == 6:
        assert error_ratio < 0.3
    elif meta.n_bits == 8 and meta.is_finite:
        assert error_ratio < 0.1
    elif meta.n_bits == 8 and not meta.is_finite:
        assert error_ratio < 0.2
    else:
        raise ValueError(f"Unknown minifloat meta: {meta}")


if __name__ == "__main__":
    # test_fp4_compose("cpu")
    # test_extract_compose_builtin_meta_saturate(FP4_E2M1_fn, "cpu", 8)
    test_extract_compose_builtin_meta(
        meta=FP6_E2M3_fn, dtype="float32", device="cuda", n_elements=1024
    )
