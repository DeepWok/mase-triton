import pytest
import torch

from mase_triton._mxfp_simple.functional import (
    compose_mxfp_tensor,
    extract_mxfp_components,
    flatten_for_quantize,
    mxfp_matmul,
    permute_for_dequantize,
    quantize_dequantize,
)
from mase_triton._mxfp_simple.meta import (
    OCP_MXFP4_E2M1,
    OCP_MXFP6_E2M3,
    OCP_MXFP6_E3M2,
    OCP_MXFP8_E4M3,
    OCP_MXFP8_E5M2,
    MXFPMeta,
)
from mase_triton.logging import set_logging_verbosity, test_logger
from mase_triton.utils.debug import set_ipdb_breakpoint
from mase_triton.utils.train_utils import set_seed

set_logging_verbosity("INFO")
logger = test_logger.getChild(__name__)

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
    [OCP_MXFP8_E4M3, OCP_MXFP8_E5M2, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
def test_quantize_dequantize_1d(mxfp_format: MXFPMeta, n_groups: int):
    n_elements = mxfp_format.block_size * n_groups
    w = torch.randn(n_elements, dtype=torch.bfloat16, device="cuda") * 100.0
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
    logger.info(f"Average error ratio for {mxfp_format} format: {avg_err_ratio:.4f}")
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
        assert avg_err_ratio < 0.5, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )


@pytest.mark.parametrize("n_groups", [16])
@pytest.mark.parametrize(
    "mxfp_format",
    [OCP_MXFP8_E4M3, OCP_MXFP8_E5M2, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
def test_quantize_dequantize_1d_wrapped(mxfp_format: MXFPMeta, n_groups: int):
    n_elements = mxfp_format.block_size * n_groups
    w = torch.randn(n_elements, dtype=torch.bfloat16, device="cuda") * 100.0
    w_dq = quantize_dequantize(w, block_dim=0, mxfp_meta=mxfp_format)
    assert w_dq.shape == w.shape, (
        f"Dequantized tensor shape {w_dq.shape} does not match original shape {w.shape}."
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
        assert avg_err_ratio < 0.5, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )


@pytest.mark.parametrize("block_dim", [0, 1, -1])
@pytest.mark.parametrize("n_groups", [16])
@pytest.mark.parametrize(
    "mxfp_format",
    [OCP_MXFP8_E4M3, OCP_MXFP8_E5M2, OCP_MXFP6_E2M3, OCP_MXFP6_E3M2, OCP_MXFP4_E2M1],
)
def test_quantize_dequantize_2d(mxfp_format: MXFPMeta, n_groups: int, block_dim: int):
    n_elements = mxfp_format.block_size * n_groups * 3
    w = torch.randn(n_elements, dtype=torch.bfloat16, device="cuda") * 50.0

    if block_dim % 2 == 0:
        w = w.reshape(-1, 3)
    else:
        w = w.reshape(3, -1)

    scales, elements, tensor_meta = extract_mxfp_components(
        w, block_dim=block_dim, mxfp_meta=mxfp_format
    )
    w_dq = compose_mxfp_tensor(
        scales=scales, elements=elements, tensor_meta=tensor_meta
    )
    assert w_dq.shape == w.shape, (
        f"Dequantized tensor shape {w_dq.shape} does not match original shape {w.shape}."
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
        assert avg_err_ratio < 0.5, (
            f"Average error ratio {avg_err_ratio} is too high for {mxfp_format} format."
        )


@pytest.mark.parametrize(
    "x_meta",
    [
        None,
        OCP_MXFP8_E4M3,
        OCP_MXFP8_E5M2,
        OCP_MXFP6_E2M3,
        OCP_MXFP6_E3M2,
        OCP_MXFP4_E2M1,
    ],
)
@pytest.mark.parametrize(
    "y_meta",
    [
        None,
        OCP_MXFP8_E4M3,
        OCP_MXFP8_E5M2,
        OCP_MXFP6_E2M3,
        OCP_MXFP6_E3M2,
        OCP_MXFP4_E2M1,
    ],
)
@pytest.mark.parametrize("device", [torch.device("cuda"), torch.device("cpu")])
@pytest.mark.parametrize("backend", ["separate"])
def test_mxfp_matmul(x_meta: MXFPMeta, y_meta: MXFPMeta, device, backend: str):
    func_type = ""
    if x_meta is not None:
        func_type += "Xq"
    else:
        func_type += "X"
    if y_meta is not None:
        func_type += "Yq"
    else:
        func_type += "Y"

    a = torch.randn((2, 4, 512, 256), dtype=torch.bfloat16, device=device)
    b = torch.randn((2, 4, 256, 128), dtype=torch.bfloat16, device=device)

    y_ref = torch.matmul(a, b)
    y = mxfp_matmul(
        a, b, input_meta=x_meta, other_meta=y_meta, func_type=func_type, backend=backend
    )

    assert y.shape == y_ref.shape, (
        f"Output shape {y.shape} does not match reference shape {y_ref.shape}."
    )
    avg_err = (y - y_ref).abs().mean()
    avg_err_ratio = avg_err / y_ref.abs().mean()
    logger.info(
        f"Average error ratio for {func_type} with {x_meta} and {y_meta}: {avg_err_ratio:.4f}"
    )
    if x_meta is None and y_meta is None:
        assert avg_err_ratio == 0
    elif x_meta is OCP_MXFP8_E4M3 and y_meta is OCP_MXFP8_E4M3:
        assert avg_err_ratio < 0.05, (
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
        assert avg_err_ratio < 0.2, (
            f"Average error ratio {avg_err_ratio} is too high for {func_type}."
        )


def test_mxfp4_all_normal():
    exp_bias = -120
    x = torch.tensor(
        [
            [6.0, 6.0, 6.0, 6.0, -6.0, -6.0, -6.0, -6.0],
            [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
            [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )
    x = x * (2**exp_bias)
    mxfp_meta = MXFPMeta(
        block_size=8,
        scale_exp_bits=8,
        element_exp_bits=2,
        element_frac_bits=1,
    )
    scales, elements, tensor_meta = extract_mxfp_components(
        x, block_dim=-1, mxfp_meta=mxfp_meta
    )
    x_dq = compose_mxfp_tensor(
        scales=scales, elements=elements, tensor_meta=tensor_meta
    )
    assert (x == x_dq).all()


def test_mxfp4_all_subnormal():
    exp_bias = -126
    x = torch.tensor(
        [
            [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )
    x = x * (2**exp_bias)
    mxfp_meta = MXFPMeta(
        block_size=8,
        scale_exp_bits=8,
        element_exp_bits=2,
        element_frac_bits=1,
    )
    scales, elements, tensor_meta = extract_mxfp_components(
        x, block_dim=-1, mxfp_meta=mxfp_meta
    )
    x_dq = compose_mxfp_tensor(
        scales=scales, elements=elements, tensor_meta=tensor_meta
    )
    assert (x == x_dq).all()


def test_mxfp4_mixture_of_normal_subnormal():
    exp_bias = -120
    x = torch.tensor(
        [
            [1.0, 0.0, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
        device="cpu",
    )
    x = x * (2**exp_bias)
    mxfp_meta = MXFPMeta(
        block_size=8,
        scale_exp_bits=8,
        element_exp_bits=2,
        element_frac_bits=1,
    )
    scales, elements, tensor_meta = extract_mxfp_components(
        x, block_dim=-1, mxfp_meta=mxfp_meta
    )
    x_dq = compose_mxfp_tensor(
        scales=scales, elements=elements, tensor_meta=tensor_meta
    )
    assert (x == x_dq).all()


if __name__ == "__main__":
    set_ipdb_breakpoint()
    test_mxfp4_all_normal()
    test_mxfp4_all_subnormal()
    test_mxfp4_mixture_of_normal_subnormal()
