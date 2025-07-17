import pytest
import torch

from mase_triton._mxfp_simple import fake as mxfp_fake
from mase_triton._mxfp_simple import kernels as mxfp_kernels
from mase_triton._mxfp_simple.meta import (
    OCP_MXFP4_E2M1,
    OCP_MXFP6_E2M3,
    OCP_MXFP6_E3M2,
    OCP_MXFP8_E4M3,
    OCP_MXFP8_E5M2,
    MXFPMeta,
)
from mase_triton.utils.train_utils import set_seed

set_seed(42)

_DEBUG_MXFP8_E4M3 = MXFPMeta(
    block_size=4,
    scale_exp_bits=8,
    element_exp_bits=4,
    element_frac_bits=3,
)


@pytest.mark.parametrize("n_groups", [16])
@pytest.mark.parametrize(
    "mxfp_format",
    [OCP_MXFP8_E4M3, OCP_MXFP8_E5M2, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
def test_extract_mxfp_components_normal(mxfp_format: MXFPMeta, n_groups: int):
    n_elements = mxfp_format.block_size * n_groups
    w = torch.randn(n_elements, dtype=torch.bfloat16, device="cuda") * 100.0
    scales, elements = mxfp_kernels.extract_mxfp_components(w, mxfp_meta=mxfp_format)
    scales_ref, elements_ref = mxfp_fake.extract_mxfp_components(
        w, mxfp_meta=mxfp_format
    )

    assert scales.dtype == torch.uint8
    assert elements.dtype == torch.uint8
    assert scales.shape == (n_groups, 1)
    assert elements.shape == (n_groups, mxfp_format.block_size)

    assert torch.all(scales == scales_ref)
    assert torch.all(elements == elements_ref)


@pytest.mark.parametrize("n_groups", [16])
@pytest.mark.parametrize(
    "mxfp_format",
    [OCP_MXFP8_E4M3, OCP_MXFP8_E5M2, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
def test_extract_mxfp_components_outliers(mxfp_format: MXFPMeta, n_groups: int):
    n_elements = mxfp_format.block_size * n_groups
    w = torch.randn(n_elements, dtype=torch.bfloat16, device="cuda") * 100.0
    for i in range(n_groups):
        w[i * mxfp_format.block_size] *= 2**32
    scales, elements = mxfp_kernels.extract_mxfp_components(w, mxfp_meta=mxfp_format)
    scales_ref, elements_ref = mxfp_fake.extract_mxfp_components(
        w, mxfp_meta=mxfp_format
    )
    assert scales.dtype == torch.uint8
    assert elements.dtype == torch.uint8
    assert scales.shape == (n_groups, 1)
    assert elements.shape == (n_groups, mxfp_format.block_size)
    assert torch.all(scales == scales_ref)
    assert torch.all(elements == elements_ref)


@pytest.mark.parametrize("n_groups", [16])
@pytest.mark.parametrize(
    "mxfp_format",
    [OCP_MXFP8_E4M3, OCP_MXFP8_E5M2, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
def test_extract_mxfp_components_subnormal(mxfp_format: MXFPMeta, n_groups: int):
    n_elements = mxfp_format.block_size * n_groups
    w = (
        torch.randint(-4, 4, (n_elements,), dtype=torch.bfloat16, device="cuda")
        * 2
        * torch.tensor(0b0000_0000_0000_0001, dtype=torch.uint16, device="cuda").view(
            torch.bfloat16
        )
    )
    scales, elements = mxfp_kernels.extract_mxfp_components(w, mxfp_meta=mxfp_format)
    scales_ref, elements_ref = mxfp_fake.extract_mxfp_components(
        w, mxfp_meta=mxfp_format
    )
    assert scales.dtype == torch.uint8
    assert elements.dtype == torch.uint8
    assert scales.shape == (n_groups, 1)
    assert elements.shape == (n_groups, mxfp_format.block_size)
    assert torch.all(scales == scales_ref)
    assert torch.all(elements == elements_ref)


@pytest.mark.parametrize("n_groups", [16])
@pytest.mark.parametrize(
    "mxfp_format",
    [OCP_MXFP8_E4M3, OCP_MXFP8_E5M2, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
def test_compose_mxfp_tensor(mxfp_format: MXFPMeta, n_groups: int):
    n_elements = mxfp_format.block_size * n_groups
    w = torch.randn(n_elements, dtype=torch.bfloat16, device="cuda") * 100.0
    scales, elements = mxfp_kernels.extract_mxfp_components(w, mxfp_meta=mxfp_format)

    w_dq = mxfp_kernels.compose_mxfp_tensor(
        shared_scales=scales,
        elements=elements,
        mxfp_meta=mxfp_format,
    )
    w_dq_ref = mxfp_fake.compose_mxfp_tensor(
        scales=scales,
        elements=elements,
        mxfp_meta=mxfp_format,
    )

    assert w_dq.dtype == torch.bfloat16
    assert w_dq.shape == (n_elements,)
    assert torch.all(w_dq == w_dq_ref)


if __name__ == "__main__":
    # test_extract_mxfp_components_normal(_DEBUG_MXFP8_E4M3, 2)
    test_extract_mxfp_components_subnormal(_DEBUG_MXFP8_E4M3, 2)
    # test_compose_mxfp_tensor(_DEBUG_MXFP8_E4M3, 2)
