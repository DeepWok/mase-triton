import pytest
import torch

from mase_triton.utils.train_utils import set_seed
from mase_triton.scaled_int.meta import ScaledIntMeta
from mase_triton.scaled_int.kernels.cast import extract_int_components

set_seed(42)

_DEBUG_SCALEDINT8 = ScaledIntMeta(
    block_size=4,
    bits=8,
)


@pytest.mark.parametrize("n_groups", [16])
def test_extract_int_components(
        scaled_int_meta: ScaledIntMeta, n_groups: int):
    n_elements = scaled_int_meta.block_size * n_groups
    w = torch.randn(n_elements, dtype=torch.bfloat16, device="cuda") * 100.0
    scales, elements = extract_int_components(
        w, scaled_int_meta=scaled_int_meta)


if __name__ == "__main__":
    test_extract_int_components(
        scaled_int_meta=_DEBUG_SCALEDINT8, n_groups=2)
