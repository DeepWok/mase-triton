import pytest
import torch

from mase_triton.manager import KernelManager
from mase_triton.mxfp import fake as mxfp_fake
from mase_triton.mxfp import kernels as mxfp_kernels
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
def test_extract_components(mxfp_format: MXFPMeta, n_groups: int, seed_iter: int):
    n_elements = mxfp_format.block_size * n_groups
    w = torch.randn(n_elements, dtype=torch.float32, device="cuda") * 100.0
    scales_ref, elements_ref = mxfp_fake.extract_mxfp_components(
        w, mxfp_meta=mxfp_format
    )
    scales, elements = mxfp_kernels.extract_mxfp_components(w, mxfp_meta=mxfp_format)
    assert scales_ref.dtype == scales.dtype
    assert elements_ref.dtype == elements.dtype
    assert scales_ref.shape == scales.shape
    assert elements_ref.shape == elements.shape
    # err_sc = scales_ref.int() - scales.int()
    # err_el = elements_ref.int() - elements.int()
    assert (scales_ref.int() == scales.int()).all()
    assert (elements_ref.int() == elements.int()).all()


@pytest.mark.parametrize("n_groups", [16])
@pytest.mark.parametrize(
    "mxfp_format",
    [MXFP8_E4M3_fn, MXFP8_E5M2_fn, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_compose_normal(dtype: str, mxfp_format: MXFPMeta, n_groups: int):
    device = "cuda"
    dtype = getattr(torch, dtype)
    n_elements = mxfp_format.block_size * n_groups
    w = torch.randn(n_elements, dtype=dtype, device=device) * 100.0
    scales, elements = mxfp_fake.extract_mxfp_components(w, mxfp_meta=mxfp_format)
    w_dq_ref = mxfp_fake.compose_mxfp_tensor(
        scales=scales, elements=elements, mxfp_meta=mxfp_format, output_dtype=dtype
    )
    w_dq = mxfp_kernels.compose_mxfp_tensor(
        scales=scales, elements=elements, mxfp_meta=mxfp_format, output_dtype=dtype
    )
    assert w_dq_ref.shape == w_dq.shape
    assert w_dq_ref.dtype == w_dq.dtype
    assert w_dq.dtype == dtype
    assert (w_dq_ref == w_dq).all(), (
        f"Dequantized tensor does not match original tensor. "
        f"Expected {w_dq_ref}, got {w_dq}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "mxfp_format",
    [MXFP8_E4M3_fn, MXFP8_E5M2_fn, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("n_elements", [1024 * 1024])
@pytest.mark.parametrize("enable", [True, False])
def test_autotune_cast(
    enable: bool, n_elements: int, dtype: str, mxfp_format: MXFPMeta
):
    dtype = getattr(torch, dtype)
    if enable:
        KernelManager.enable_autotune()
    else:
        KernelManager.disable_autotune()
    w = torch.randn(n_elements, dtype=dtype, device="cuda") * 10
    scales_ref, elements_ref = mxfp_kernels.extract_mxfp_components(
        w, mxfp_meta=mxfp_format
    )
    _ = mxfp_kernels.compose_mxfp_tensor(
        scales=scales_ref,
        elements=elements_ref,
        mxfp_meta=mxfp_format,
        output_dtype=dtype,
    )


if __name__ == "__main__":
    # test_extract_components(mxfp_format=OCP_MXFP6_E2M3, n_groups=2, seed_iter=0)
    # test_compose_normal(dtype="float32", mxfp_format=OCP_MXFP6_E3M2, n_groups=2)
    test_autotune_cast(
        enable=True, n_elements=1024 * 1024, dtype="float32", mxfp_format=OCP_MXFP6_E2M3
    )
