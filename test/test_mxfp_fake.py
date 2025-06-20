import pytest
import torch

from mase_triton.mxfp.fake import compose_mxfp_tensor, extract_mxfp_components
from mase_triton.mxfp.meta import (
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


@pytest.mark.parametrize(
    "mxfp_format",
    [OCP_MXFP8_E4M3, OCP_MXFP8_E5M2, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
@pytest.mark.parametrize("n_groups", [16])
def test_simulated_extract_and_compose_normal(mxfp_format: MXFPMeta, n_groups: int):
    n_elements = mxfp_format.block_size * n_groups
    w = torch.randn(n_elements, dtype=torch.bfloat16, device="cuda") * 100.0
    scales, elements = extract_mxfp_components(w, mxfp_meta=mxfp_format)
    w_dq = compose_mxfp_tensor(
        shared_scales=scales,
        elements=elements,
        mxfp_meta=mxfp_format,
    )

    avg_err = (w - w_dq).abs().mean()
    avg_err_ratio = avg_err / w.abs().mean()
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
        assert avg_err_ratio < 0.4, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    else:
        assert avg_err_ratio < 0.6, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )


@pytest.mark.parametrize(
    "mxfp_format",
    [OCP_MXFP8_E4M3, OCP_MXFP8_E5M2, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
@pytest.mark.parametrize("n_groups", [16])
def test_simulated_extract_and_compose_outliers(mxfp_format: MXFPMeta, n_groups: int):
    n_elements = mxfp_format.block_size * n_groups

    w = torch.randn(n_elements, dtype=torch.bfloat16, device="cuda") * 100.0
    for i in range(n_groups):
        w[i * mxfp_format.block_size] *= 2**32
    scales, elements = extract_mxfp_components(w, mxfp_meta=mxfp_format)
    w_dq = compose_mxfp_tensor(
        shared_scales=scales,
        elements=elements,
        mxfp_meta=mxfp_format,
    )
    avg_err = (w - w_dq).abs().mean()
    avg_err_ratio = avg_err / w.abs().mean()
    if mxfp_format is OCP_MXFP4_E2M1:
        assert avg_err_ratio < 0.2, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    else:
        assert avg_err_ratio < 0.1, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )


@pytest.mark.parametrize(
    "mxfp_format",
    [OCP_MXFP8_E4M3, OCP_MXFP8_E5M2, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
@pytest.mark.parametrize("n_groups", [16])
def test_simulated_extract_and_compose_subnormal(mxfp_format: MXFPMeta, n_groups: int):
    n_elements = mxfp_format.block_size * n_groups
    largest_subnormal = 0b0000_0000_0111_1111
    w = torch.randint(
        0,
        largest_subnormal,
        (n_elements,),
        dtype=torch.int16,
        device="cuda",
    ).view(torch.bfloat16)
    scales, elements = extract_mxfp_components(w, mxfp_meta=mxfp_format)
    w_dq = compose_mxfp_tensor(scales, elements, mxfp_meta=mxfp_format)
    assert (w_dq == 0.0).all()


if __name__ == "__main__":
    # test_simulated_extract_and_compose_normal(mxfp_format=OCP_MXFP8_E4M3, n_groups=16)
    # test_simulated_extract_and_compose_outliers(mxfp_format=OCP_MXFP8_E4M3, n_groups=2)
    test_simulated_extract_and_compose_subnormal(
        mxfp_format=_DEBUG_MXFP8_E4M3, n_groups=2
    )
