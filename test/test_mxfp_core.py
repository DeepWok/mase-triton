import pytest
import torch

from mase_triton.mxfp.dtypes import (
    OCP_MXFP4_E2M1,
    OCP_MXFP6_E2M3,
    OCP_MXFP6_E3M2,
    OCP_MXFP8_E4M3,
    OCP_MXFP8_E5M2,
    MXFPMeta,
)
from mase_triton.mxfp.fake import compose_mxfp_tensor, extract_mxfp_components
from mase_triton.utils.train_utils import set_seed

set_seed(42)


@pytest.mark.parametrize(
    "mxfp_format",
    [OCP_MXFP8_E4M3, OCP_MXFP8_E5M2, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
@pytest.mark.parametrize("n_groups", [16])
def test_simulated_extract_and_compose(mxfp_format: MXFPMeta, n_groups: int):
    mxfp_format = OCP_MXFP8_E4M3
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
    if mxfp_format.element.n_bits >= 8:
        assert avg_err_ratio < 0.05
    elif mxfp_format.element.n_bits == 6:
        assert avg_err_ratio < 0.1
    elif mxfp_format.element.n_bits == 4:
        assert avg_err_ratio < 0.2
    else:
        assert avg_err_ratio < 0.5


if __name__ == "__main__":
    test_simulated_extract_and_compose(mxfp_format=OCP_MXFP8_E4M3, n_groups=16)
