import pytest
import torch

from mase_triton.minifloat.fake import compose_minifloat_tensor, extract_minifloat_components
from mase_triton.utils.train_utils import set_seed

set_seed(42)


def test_simulated_extract_and_compose_normal():
    n_elements = 1024

    w = torch.randn(n_elements, dtype=torch.bfloat16, device="cuda")
    elements = extract_minifloat_components(w)
    w_dq = compose_minifloat_tensor(elements)

    avg_err = (w - w_dq).abs().mean()
    avg_err_ratio = avg_err / w.abs().mean()
    assert avg_err_ratio < 0.05, (
        f"Average error ratio {avg_err_ratio} is too high."
    )



if __name__ == "__main__":
    test_simulated_extract_and_compose_normal()
