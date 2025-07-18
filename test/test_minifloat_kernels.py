import pytest
import torch

import mase_triton.minifloat.fake as minifloat_fake
import mase_triton.minifloat.kernels as minifloat_kernels
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
def test_extract_compose_builtin_meta(meta: MinifloatMeta, n_elements: int):
    device = "cuda"
    x = torch.randn(n_elements, dtype=torch.float32, device=device)
    x_q = minifloat_kernels.extract_minifloat_component(x, minifloat_meta=meta)
    x_q_ref = minifloat_fake.extract_minifloat_component(x, minifloat_meta=meta)
    assert x_q_ref.dtype == x_q.dtype
    assert x_q_ref.shape == x_q.shape
    err = (x_q_ref.float() - x_q.float()).abs().max().item()
    assert err <= 1


if __name__ == "__main__":
    test_extract_compose_builtin_meta(FP6_E2M3_fn, 1024)
