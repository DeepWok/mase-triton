import pytest
import torch

from mase_triton.minifloat import fake as minifloat_fake
from mase_triton.minifloat import kernels as minifloat_kernels
from mase_triton.utils.train_utils import set_seed

set_seed(42)

def test_extract_minifloat_components_normal():
    n_elements = 1024
    w = torch.randn(n_elements, dtype=torch.bfloat16, device="cuda")
    elements = minifloat_kernels.extract_minifloat_components(w)
    elements_ref = minifloat_fake.extract_minifloat_components(w)

    assert elements.dtype == torch.uint8
    assert torch.all(elements == elements_ref)


def test_compose_minifloat_tensor():
    n_elements = 1024
    w = torch.randn(n_elements, dtype=torch.bfloat16, device="cuda")
    elements = minifloat_kernels.extract_minifloat_components(w)

    w_dq = minifloat_kernels.compose_minifloat_tensor(elements)
    w_dq_ref = minifloat_fake.compose_minifloat_tensor(elements)

    assert w_dq.dtype == torch.bfloat16
    assert torch.all(w_dq == w_dq_ref)


# if __name__ == "__main__":

