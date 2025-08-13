import pytest
import torch

from mase_triton.mxfp.fake import compose_mxfp_tensor, extract_mxfp_components
from mase_triton.mxfp.meta import (
    OCP_MXFP4_E2M1,
    OCP_MXFP6_E2M3,
    OCP_MXFP6_E3M2,
    MXFP8_E4M3_fn,
    MXFP8_E5M2_fn,
    MXFPMeta,
)
from mase_triton.utils.debug import set_ipdb_breakpoint
from mase_triton.utils.train_utils import set_seed

set_ipdb_breakpoint()
set_seed(0)


@pytest.mark.parametrize("seed_iter", [0, 1, 2])
@pytest.mark.parametrize("n_groups", [16])
@pytest.mark.parametrize(
    "mxfp_format",
    [MXFP8_E4M3_fn, MXFP8_E5M2_fn, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_simulated_extract_and_compose_cpu_cuda(
    dtype: str, mxfp_format: MXFPMeta, n_groups: int, seed_iter: int
):
    dtype = getattr(torch, dtype)
    # Check the consistency of cpu and cuda implementations
    n_elements = mxfp_format.block_size * n_groups
    w = torch.randn(n_elements, dtype=dtype, device="cuda") * 100.0
    w_cpu = w.cpu()
    scales_cuda, elements_cuda = extract_mxfp_components(w, mxfp_meta=mxfp_format)
    w_dq_cuda = compose_mxfp_tensor(
        scales=scales_cuda,
        elements=elements_cuda,
        mxfp_meta=mxfp_format,
        output_dtype=dtype,
    )
    scales_cpu, elements_cpu = extract_mxfp_components(w_cpu, mxfp_meta=mxfp_format)
    w_dq_cpu = compose_mxfp_tensor(
        scales=scales_cpu,
        elements=elements_cpu,
        mxfp_meta=mxfp_format,
        output_dtype=dtype,
    )
    # check that the results are the same on CPU and GPU
    assert (scales_cuda == scales_cpu.cuda()).all()
    assert (elements_cuda == elements_cpu.cuda()).all()
    assert (w_dq_cuda.cpu() == w_dq_cpu).all()


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("n_groups", [16])
@pytest.mark.parametrize(
    "mxfp_format",
    [MXFP8_E4M3_fn, MXFP8_E5M2_fn, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_simulated_extract_and_compose_normal(
    dtype: str, mxfp_format: MXFPMeta, n_groups: int, device: str
):
    dtype = getattr(torch, dtype)
    n_elements = mxfp_format.block_size * n_groups
    w = torch.randn(n_elements, dtype=dtype, device=device) * 100.0
    scales, elements = extract_mxfp_components(w, mxfp_meta=mxfp_format)
    w_dq = compose_mxfp_tensor(
        scales=scales, elements=elements, mxfp_meta=mxfp_format, output_dtype=dtype
    )
    avg_err = (w - w_dq).abs().mean()
    avg_err_ratio = avg_err / w.abs().mean()
    print(f"Average error ratio for {mxfp_format} format: {avg_err_ratio:.4f}")
    if mxfp_format is MXFP8_E4M3_fn:
        assert avg_err_ratio < 0.05, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    elif mxfp_format is MXFP8_E5M2_fn:
        assert avg_err_ratio < 0.1, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    elif mxfp_format is OCP_MXFP6_E3M2:
        assert avg_err_ratio < 0.1, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    elif mxfp_format is OCP_MXFP6_E2M3:
        assert avg_err_ratio < 0.1, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    else:
        assert avg_err_ratio < 0.25, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("n_groups", [16])
@pytest.mark.parametrize(
    "mxfp_format",
    [MXFP8_E4M3_fn, MXFP8_E5M2_fn, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_simulated_extract_and_compose_outliers(
    dtype: str, mxfp_format: MXFPMeta, n_groups: int, device: str
):
    dtype = getattr(torch, dtype)
    n_elements = mxfp_format.block_size * n_groups

    w = torch.randn(n_elements, dtype=dtype, device=device) * 100.0
    for i in range(n_groups):
        w[i * mxfp_format.block_size] *= 2**32
    scales, elements = extract_mxfp_components(w, mxfp_meta=mxfp_format)
    w_dq = compose_mxfp_tensor(
        scales=scales,
        elements=elements,
        mxfp_meta=mxfp_format,
        output_dtype=dtype,
    )
    avg_err = (w.float() - w_dq.float()).abs().mean()
    avg_err_ratio = avg_err / w.abs().mean()
    print(f"Average error ratio for {mxfp_format} format: {avg_err_ratio:.4f}")
    if mxfp_format is OCP_MXFP4_E2M1 or mxfp_format is OCP_MXFP6_E2M3:
        assert avg_err_ratio < 0.25, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    else:
        assert avg_err_ratio < 0.15, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("n_groups", [16])
@pytest.mark.parametrize(
    "mxfp_format",
    [MXFP8_E4M3_fn, MXFP8_E5M2_fn, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
def test_simulated_extract_and_compose_subnormal_input(
    mxfp_format: MXFPMeta, n_groups: int, device: str
):
    n_elements = mxfp_format.block_size * n_groups
    largest_subnormal = 0b0000_0000_0111_1111
    w = torch.randint(
        0, largest_subnormal, (n_elements,), dtype=torch.int16, device=device
    ).view(torch.float32)
    scales, elements = extract_mxfp_components(w, mxfp_meta=mxfp_format)
    w_dq = compose_mxfp_tensor(
        scales, elements, mxfp_meta=mxfp_format, output_dtype=torch.float32
    )
    assert w_dq.dtype == torch.float32
    assert (w_dq == 0.0).all(), (
        "Dequantized tensor should be all zeros for subnormal values"
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_mxfp4_all_normal_minifloat(dtype: str, device: str):
    dtype = getattr(torch, dtype)
    exp_bias = -120
    x = torch.tensor(
        [
            [6.0, 6.0, 6.0, 6.0, -6.0, -6.0, -6.0, -6.0],
            [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
            [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    ).flatten()
    x = x * (2**exp_bias)
    mxfp_meta = MXFPMeta(
        block_size=8,
        scale_exp_bits=8,
        element_exp_bits=2,
        element_frac_bits=1,
        element_is_finite=True,
        round_mode="rn",
    )
    scales, elements = extract_mxfp_components(x, mxfp_meta=mxfp_meta)
    x_dq = compose_mxfp_tensor(
        scales=scales, elements=elements, mxfp_meta=mxfp_meta, output_dtype=dtype
    )
    assert (x == x_dq).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_mxfp4_mixture_of_normal_subnormal_minifloat(dtype: str, device: str):
    exp_bias = -120
    x = torch.tensor(
        [
            [1.0, 0.0, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    ).flatten()
    x = x * (2**exp_bias)
    mxfp_meta = MXFPMeta(
        block_size=8,
        scale_exp_bits=8,
        element_exp_bits=2,
        element_frac_bits=1,
        element_is_finite=True,
        round_mode="rn",
    )
    scales, elements = extract_mxfp_components(x, mxfp_meta=mxfp_meta)
    x_dq = compose_mxfp_tensor(
        scales=scales,
        elements=elements,
        mxfp_meta=mxfp_meta,
        output_dtype=torch.float32,
    )
    assert (x == x_dq).all()


if __name__ == "__main__":
    device = "cuda"
    # test_simulated_extract_and_compose_cpu_cuda(OCP_MXFP4_E2M1, n_groups=2, seed_iter=0)
    # test_simulated_extract_and_compose_normal(OCP_MXFP4_E2M1, n_groups=2, device=device)
    test_simulated_extract_and_compose_subnormal_input(
        MXFP8_E4M3_fn, n_groups=2, device=device
    )
