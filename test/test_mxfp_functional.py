import pytest
import torch

from mase_triton.mxfp.functional import (
    compose_mxfp_tensor,
    extract_mxfp_components,
    flatten_for_quantize,
    mxfp_matmul,
    permute_for_dequantize,
    quantize_dequantize,
)
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


@pytest.mark.parametrize("block_dim", [0, 1, 2, -1, -2, -3])
def test_mxfp_components(block_dim: int):
    x = torch.arange(24).reshape(2, 3, 4)

    x_flatten = flatten_for_quantize(x, block_dim)
    x_restore = permute_for_dequantize(
        x_flatten,
        block_dim=block_dim,
        ori_shape=x.shape,
    )

    assert x_restore.shape == x.shape
    assert torch.all(x_restore == x)


@pytest.mark.parametrize("n_groups", [16])
@pytest.mark.parametrize(
    "mxfp_format",
    [MXFP8_E4M3_fn, MXFP8_E5M2_fn, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_quantize_dequantize_1d(dtype: str, mxfp_format: MXFPMeta, n_groups: int):
    dtype = getattr(torch, dtype)
    n_elements = mxfp_format.block_size * n_groups
    w = torch.randn(n_elements, dtype=dtype, device="cuda") * 100.0
    scales, elements, tensor_meta = extract_mxfp_components(
        w, block_dim=0, mxfp_meta=mxfp_format
    )
    w_dq = compose_mxfp_tensor(
        scales=scales, elements=elements, tensor_meta=tensor_meta
    )
    assert w_dq.shape == w.shape, (
        f"Dequantized tensor shape {w_dq.shape} does not match original shape {w.shape}."
    )
    avg_err = (w - w_dq).abs().mean()
    avg_err_ratio = avg_err / w.abs().mean()
    print(f"Average error ratio for {mxfp_format} format: {avg_err_ratio:.4f}")
    if mxfp_format is MXFP8_E4M3_fn:
        assert avg_err_ratio < 0.05, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    elif mxfp_format is MXFP8_E5M2_fn:
        assert avg_err_ratio < 0.1, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    elif mxfp_format is OCP_MXFP6_E3M2:
        assert avg_err_ratio < 0.1, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    elif mxfp_format is OCP_MXFP6_E2M3:
        assert avg_err_ratio < 0.1, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    else:
        assert avg_err_ratio < 0.3, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )


@pytest.mark.parametrize("n_groups", [16])
@pytest.mark.parametrize(
    "mxfp_format",
    [MXFP8_E4M3_fn, MXFP8_E5M2_fn, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_quantize_dequantize_1d_wrapped(
    dtype: str, mxfp_format: MXFPMeta, n_groups: int
):
    dtype = getattr(torch, dtype)
    n_elements = mxfp_format.block_size * n_groups
    w = torch.randn(n_elements, dtype=dtype, device="cuda") * 100.0
    w_dq = quantize_dequantize(w, block_dim=0, mxfp_meta=mxfp_format)
    assert w_dq.shape == w.shape, (
        f"Dequantized tensor shape {w_dq.shape} does not match original shape {w.shape}."
    )
    avg_err = (w - w_dq).abs().mean()
    avg_err_ratio = avg_err / w.abs().mean()
    print(f"Average error ratio for {mxfp_format} format: {avg_err_ratio:.4f}")
    if mxfp_format is MXFP8_E4M3_fn:
        assert avg_err_ratio < 0.05, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    elif mxfp_format is MXFP8_E5M2_fn:
        assert avg_err_ratio < 0.1, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    elif mxfp_format is OCP_MXFP6_E3M2:
        assert avg_err_ratio < 0.1, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    elif mxfp_format is OCP_MXFP6_E2M3:
        assert avg_err_ratio < 0.1, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    else:
        assert avg_err_ratio < 0.3, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )


@pytest.mark.parametrize("block_dim", [0, 1, -1])
@pytest.mark.parametrize("n_groups", [16])
@pytest.mark.parametrize(
    "mxfp_format",
    [MXFP8_E4M3_fn, MXFP8_E5M2_fn, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_quantize_dequantize_2d(
    dtype: str, mxfp_format: MXFPMeta, n_groups: int, block_dim: int
):
    dtype = getattr(torch, dtype)
    n_elements = mxfp_format.block_size * n_groups * 3
    w = torch.randn(n_elements, dtype=dtype, device="cuda") * 50.0

    if block_dim % 2 == 0:
        w = w.reshape(-1, 3)
    else:
        w = w.reshape(3, -1)

    scales, elements, tensor_meta = extract_mxfp_components(
        w, block_dim=block_dim, mxfp_meta=mxfp_format
    )
    w_dq = compose_mxfp_tensor(
        scales=scales,
        elements=elements,
        tensor_meta=tensor_meta,
        output_dtype=torch.float32,
    )
    assert w_dq.shape == w.shape, (
        f"Dequantized tensor shape {w_dq.shape} does not match original shape {w.shape}."
    )
    avg_err = (w - w_dq).abs().mean()
    avg_err_ratio = avg_err / w.abs().mean()
    print(f"Average error ratio for {mxfp_format} format: {avg_err_ratio:.4f}")
    if mxfp_format is MXFP8_E4M3_fn:
        assert avg_err_ratio < 0.05, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    elif mxfp_format is MXFP8_E5M2_fn:
        assert avg_err_ratio < 0.1, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    elif mxfp_format is OCP_MXFP6_E3M2:
        assert avg_err_ratio < 0.1, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    elif mxfp_format is OCP_MXFP6_E2M3:
        assert avg_err_ratio < 0.1, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )
    else:
        assert avg_err_ratio < 0.3, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )


@pytest.mark.parametrize(
    "x_meta",
    [
        None,
        MXFP8_E4M3_fn,
        MXFP8_E5M2_fn,
        OCP_MXFP6_E2M3,
        OCP_MXFP6_E3M2,
        OCP_MXFP4_E2M1,
    ],
)
@pytest.mark.parametrize(
    "y_meta",
    [
        None,
        MXFP8_E4M3_fn,
        MXFP8_E5M2_fn,
        OCP_MXFP6_E2M3,
        OCP_MXFP6_E3M2,
        OCP_MXFP4_E2M1,
    ],
)
@pytest.mark.parametrize("device", [torch.device("cuda"), torch.device("cpu")])
@pytest.mark.parametrize("backend", ["separate"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("problem_size", [(128, 512, 1024)])
def test_mxfp_matmul(
    dtype: str,
    x_meta: MXFPMeta,
    y_meta: MXFPMeta,
    device,
    backend: str,
    problem_size: tuple[int],
):
    dtype = getattr(torch, dtype)
    func_type = ""
    if x_meta is not None:
        func_type += "Xq"
    else:
        func_type += "X"
    if y_meta is not None:
        func_type += "Wq"
    else:
        func_type += "W"

    m, n, k = problem_size
    a = torch.randn((2, 4, m, k), dtype=dtype, device=device) * 5
    b = torch.randn((2, 4, k, n), dtype=dtype, device=device) * 3

    y_ref = torch.matmul(a, b)
    y = mxfp_matmul(
        a, b, input_meta=x_meta, other_meta=y_meta, func_type=func_type, backend=backend
    )

    assert y.shape == y_ref.shape, (
        f"Output shape {y.shape} does not match reference shape {y_ref.shape}."
    )
    avg_err = (y - y_ref).abs().mean()
    avg_err_ratio = avg_err / y_ref.abs().mean()
    x_meta_tag = x_meta.tag if x_meta else "None"
    y_meta_tag = y_meta.tag if y_meta else "None"
    problem_size_str = ("(problem_size: " + str(problem_size) + ")").ljust(40)
    print(
        f"Average error ratio for {func_type.ljust(6)} with {x_meta_tag.ljust(16)} and {y_meta_tag.ljust(16)} {problem_size_str}: {avg_err_ratio:.4f}"
    )
    if x_meta is None and y_meta is None:
        assert avg_err_ratio == 0
    elif x_meta is MXFP8_E4M3_fn and y_meta is MXFP8_E4M3_fn:
        assert avg_err_ratio < 0.2, (
            f"Average error ratio {avg_err_ratio} is too high for {func_type}."
        )
    elif x_meta is OCP_MXFP4_E2M1 or y_meta is OCP_MXFP4_E2M1:
        assert avg_err_ratio < 0.5, (
            f"Average error ratio {avg_err_ratio} is too high for {func_type}."
        )
    elif (x_meta is OCP_MXFP6_E2M3) or (y_meta is OCP_MXFP6_E2M3):
        assert avg_err_ratio < 0.45, (
            f"Average error ratio {avg_err_ratio} is too high for {func_type}."
        )
    else:
        assert avg_err_ratio < 0.25, (
            f"Average error ratio {avg_err_ratio} is too high for {func_type}."
        )


if __name__ == "__main__":
    test_mxfp_matmul(
        x_meta=None,
        y_meta=OCP_MXFP4_E2M1,
        device=torch.device("cuda"),
        backend="separate",
    )
