import pytest
import torch

from mase_triton.mxfp import core as mxfp_core
from mase_triton.mxfp import fake as mxfp_fake
from mase_triton.mxfp.meta import (
    OCP_MXFP4_E2M1,
    OCP_MXFP6_E2M3,
    OCP_MXFP6_E3M2,
    OCP_MXFP8_E4M3,
    OCP_MXFP8_E5M2,
    MXFPMeta,
)


@pytest.mark.parametrize("mxfp_format", [OCP_MXFP8_E4M3])
@pytest.mark.parametrize("n_groups", [2])
def test_extract_mxfp_components(mxfp_format: MXFPMeta, n_groups: int): ...
