import pytest
import torch

from mase_triton._mxfp_simple.fake import compose_mxfp_tensor, extract_mxfp_components
from mase_triton._mxfp_simple.meta import (
    OCP_MXFP4_E2M1,
    OCP_MXFP6_E2M3,
    OCP_MXFP6_E3M2,
    OCP_MXFP8_E4M3,
    OCP_MXFP8_E5M2,
    MXFPMeta,
)
from mase_triton.logging import set_logging_verbosity, test_logger
from mase_triton.utils.debug import set_ipdb_breakpoint
from mase_triton.utils.train_utils import set_seed

set_logging_verbosity("INFO")
logger = test_logger.getChild(__name__)

set_seed(0)

_DEBUG_MXFP8_E4M3 = MXFPMeta(
    block_size=4,
    scale_exp_bits=8,
    element_exp_bits=4,
    element_frac_bits=3,
)


@pytest.mark.parametrize("seed_iter", [0, 1, 2])
@pytest.mark.parametrize("n_groups", [16])
@pytest.mark.parametrize(
    "mxfp_format",
    [OCP_MXFP8_E4M3, OCP_MXFP8_E5M2, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
def test_simulated_extract_and_compose_cpu_cuda(
    mxfp_format: MXFPMeta, n_groups: int, seed_iter: int
):
    # Check the consistency of cpu and cuda implementations
    n_elements = mxfp_format.block_size * n_groups
    w = torch.randn(n_elements, dtype=torch.bfloat16, device="cuda") * 100.0
    w_cpu = w.cpu()
    scales_cuda, elements_cuda = extract_mxfp_components(w, mxfp_meta=mxfp_format)
    w_dq_cuda = compose_mxfp_tensor(
        scales=scales_cuda, elements=elements_cuda, mxfp_meta=mxfp_format
    )
    scales_cpu, elements_cpu = extract_mxfp_components(w_cpu, mxfp_meta=mxfp_format)
    w_dq_cpu = compose_mxfp_tensor(
        scales=scales_cpu, elements=elements_cpu, mxfp_meta=mxfp_format
    )
    # check that the results are the same on CPU and GPU
    assert (scales_cuda == scales_cpu.cuda()).all()
    assert (elements_cuda == elements_cpu.cuda()).all()
    assert (w_dq_cuda.cpu() == w_dq_cpu).all()


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("n_groups", [16])
@pytest.mark.parametrize(
    "mxfp_format",
    [OCP_MXFP8_E4M3, OCP_MXFP8_E5M2, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
def test_simulated_extract_and_compose_normal(
    mxfp_format: MXFPMeta, n_groups: int, device: str
):
    n_elements = mxfp_format.block_size * n_groups
    w = torch.randn(n_elements, dtype=torch.bfloat16, device=device) * 100.0
    scales, elements = extract_mxfp_components(w, mxfp_meta=mxfp_format)
    w_dq = compose_mxfp_tensor(scales=scales, elements=elements, mxfp_meta=mxfp_format)
    avg_err = (w - w_dq).abs().mean()
    avg_err_ratio = avg_err / w.abs().mean()
    logger.info(f"Average error ratio for {mxfp_format} format: {avg_err_ratio:.4f}")
    if mxfp_format is OCP_MXFP8_E4M3:
        assert avg_err_ratio < 0.05, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    elif mxfp_format is OCP_MXFP8_E5M2:
        assert avg_err_ratio < 0.1, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    elif mxfp_format is OCP_MXFP6_E3M2:
        assert avg_err_ratio < 0.2, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    elif mxfp_format is OCP_MXFP6_E2M3:
        assert avg_err_ratio < 0.45, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    else:
        assert avg_err_ratio < 0.5, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("n_groups", [16])
@pytest.mark.parametrize(
    "mxfp_format",
    [OCP_MXFP8_E4M3, OCP_MXFP8_E5M2, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
def test_simulated_extract_and_compose_outliers(
    mxfp_format: MXFPMeta, n_groups: int, device: str
):
    n_elements = mxfp_format.block_size * n_groups

    w = torch.randn(n_elements, dtype=torch.bfloat16, device=device) * 100.0
    for i in range(n_groups):
        w[i * mxfp_format.block_size] *= 2**32
    scales, elements = extract_mxfp_components(w, mxfp_meta=mxfp_format)
    w_dq = compose_mxfp_tensor(
        scales=scales,
        elements=elements,
        mxfp_meta=mxfp_format,
    )
    avg_err = (w.float() - w_dq.float()).abs().mean()
    avg_err_ratio = avg_err / w.abs().mean()
    logger.info(f"Average error ratio for {mxfp_format} format: {avg_err_ratio:.4f}")
    if mxfp_format is OCP_MXFP4_E2M1:
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
    [OCP_MXFP8_E4M3, OCP_MXFP8_E5M2, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
def test_simulated_extract_and_compose_subnormal(
    mxfp_format: MXFPMeta, n_groups: int, device: str
):
    n_elements = mxfp_format.block_size * n_groups
    largest_subnormal = 0b0000_0000_0111_1111
    w = torch.randint(
        0, largest_subnormal, (n_elements,), dtype=torch.int16, device=device
    ).view(torch.bfloat16)
    scales, elements = extract_mxfp_components(w, mxfp_meta=mxfp_format)
    w_dq = compose_mxfp_tensor(scales, elements, mxfp_meta=mxfp_format)
    assert w_dq.dtype == torch.bfloat16
    avg_err = (w - w_dq).abs().mean()
    avg_err_ratio = avg_err / w.abs().mean()
    logger.info(f"Average error ratio for {mxfp_format} format: {avg_err_ratio:.4f}")
    if mxfp_format in [OCP_MXFP8_E4M3, OCP_MXFP6_E2M3]:
        assert avg_err_ratio < 0.2, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    elif mxfp_format in [OCP_MXFP8_E5M2, OCP_MXFP6_E3M2]:
        assert avg_err_ratio < 0.3, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    else:
        assert avg_err_ratio < 0.6, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mxfp4_all_normal(device):
    exp_bias = -120
    x = torch.tensor(
        [
            [6.0, 6.0, 6.0, 6.0, -6.0, -6.0, -6.0, -6.0],
            [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
            [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
        device=device,
    ).flatten()
    x = x * (2**exp_bias)
    mxfp_meta = MXFPMeta(
        block_size=8,
        scale_exp_bits=8,
        element_exp_bits=2,
        element_frac_bits=1,
    )
    scales, elements = extract_mxfp_components(x, mxfp_meta=mxfp_meta)
    x_dq = compose_mxfp_tensor(scales=scales, elements=elements, mxfp_meta=mxfp_meta)
    assert (x == x_dq).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mxfp4_all_subnormal(device):
    exp_bias = -126
    x = torch.tensor(
        [
            [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
        device=device,
    ).flatten()
    x = x * (2**exp_bias)
    mxfp_meta = MXFPMeta(
        block_size=8,
        scale_exp_bits=8,
        element_exp_bits=2,
        element_frac_bits=1,
    )
    scales, elements = extract_mxfp_components(x, mxfp_meta=mxfp_meta)
    x_dq = compose_mxfp_tensor(scales=scales, elements=elements, mxfp_meta=mxfp_meta)
    assert (x == x_dq).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mxfp4_mixture_of_normal_subnormal(device):
    exp_bias = -120
    x = torch.tensor(
        [
            [1.0, 0.0, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
        device=device,
    ).flatten()
    x = x * (2**exp_bias)
    mxfp_meta = MXFPMeta(
        block_size=8,
        scale_exp_bits=8,
        element_exp_bits=2,
        element_frac_bits=1,
    )
    scales, elements = extract_mxfp_components(x, mxfp_meta=mxfp_meta)
    x_dq = compose_mxfp_tensor(scales=scales, elements=elements, mxfp_meta=mxfp_meta)
    assert (x == x_dq).all()


if __name__ == "__main__":
    set_ipdb_breakpoint()
    device = "cuda"
    test_simulated_extract_and_compose_cpu_cuda(
        OCP_MXFP4_E2M1, n_groups=16, seed_iter=0
    )
    test_simulated_extract_and_compose_normal(OCP_MXFP8_E4M3, n_groups=2, device=device)
    test_simulated_extract_and_compose_outliers(
        OCP_MXFP8_E4M3, n_groups=2, device=device
    )
    test_simulated_extract_and_compose_subnormal(
        _DEBUG_MXFP8_E4M3, n_groups=2, device=device
    )
