import pytest
import torch

from mase_triton.mxfp.functional import extract_mxfp_components, quantize_dequantize
from mase_triton.mxfp.layers import MXFPLinearPTQ
from mase_triton.mxfp.meta import (
    OCP_MXFP4_E2M1,
    OCP_MXFP6_E2M3,
    OCP_MXFP6_E3M2,
    MXFP8_E4M3_fn,
    MXFP8_E5M2_fn,
    MXFPMeta,
)
from mase_triton.utils.meta import device_str, dtype_str, shape_tuple
from mase_triton.utils.qlayer import ChangeDtypeError, devices_equal
from mase_triton.utils.train_utils import set_seed

set_seed(42)


@pytest.mark.parametrize("MNK", [(128, 512, 1024)])
@pytest.mark.parametrize("backend", ["separate"])
@pytest.mark.parametrize(
    "x_meta",
    [
        MXFP8_E4M3_fn,
        MXFP8_E5M2_fn,
        OCP_MXFP6_E2M3,
        OCP_MXFP6_E3M2,
        OCP_MXFP4_E2M1,
        None,
    ],
)
@pytest.mark.parametrize(
    "w_meta",
    [
        MXFP8_E4M3_fn,
        MXFP8_E5M2_fn,
        OCP_MXFP6_E2M3,
        OCP_MXFP6_E3M2,
        OCP_MXFP4_E2M1,
        None,
    ],
)
@pytest.mark.parametrize("b_meta", [MXFP8_E4M3_fn, None])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_mxfp_linear_ptq_from_linear(
    MNK,
    backend: str,
    x_meta: MXFPMeta,
    w_meta: MXFPMeta,
    b_meta: MXFPMeta,
    bias: bool,
    dtype: torch.dtype,
):
    M, N, K = MNK
    layer_type = ""
    if x_meta is None:
        layer_type += "X"
    else:
        layer_type += "Xq"
    if w_meta is None:
        layer_type += "W"
    else:
        layer_type += "Wq"
    if b_meta is None:
        layer_type += "B"
    else:
        layer_type += "Bq"

    if not bias:
        b_meta = None

    in_features = K
    out_features = N
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    fc_ref = torch.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        device=device,
        dtype=dtype,
    )
    fc_ref.to(device=device)
    fc_mxfp = MXFPLinearPTQ.from_linear(
        layer=fc_ref,
        x_mxfp_meta=x_meta,
        w_mxfp_meta=w_meta,
        b_mxfp_meta=b_meta,
        layer_type=layer_type,
        backend=backend,
    )

    x = torch.randn(M, K, device=device, dtype=dtype) * 3
    y_ref = fc_ref(x)
    y_mxfp = fc_mxfp(x)

    avg_err = (y_ref - y_mxfp).abs().mean().item()
    avg_err_ratio = avg_err / y_ref.abs().mean().item()

    print(
        f"Average error ratio for {layer_type} with {x_meta}, {w_meta}, {b_meta}: {avg_err_ratio:.4f}"
    )
    if x_meta is OCP_MXFP4_E2M1 or w_meta is OCP_MXFP4_E2M1 or b_meta is OCP_MXFP4_E2M1:
        assert avg_err_ratio < 0.4
    else:
        assert avg_err_ratio < 0.25


@pytest.mark.parametrize("MNK", [(128, 512, 1024)])
@pytest.mark.parametrize("backend", ["separate"])
@pytest.mark.parametrize("x_meta", [None])
@pytest.mark.parametrize(
    "w_meta",
    [
        MXFP8_E4M3_fn,
        MXFP8_E5M2_fn,
        OCP_MXFP6_E3M2,
        OCP_MXFP6_E2M3,
        OCP_MXFP4_E2M1,
    ],
)
@pytest.mark.parametrize("b_meta", [OCP_MXFP4_E2M1, None])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.no_grad()
def test_mxfp_linear_ptq_from_quantized(
    MNK,
    backend: str,
    x_meta: MXFPMeta,
    w_meta: MXFPMeta,
    b_meta: MXFPMeta,
    bias: bool,
    dtype: torch.dtype,
):
    M, N, K = MNK
    layer_type = ""
    if x_meta is None:
        layer_type += "X"
    else:
        layer_type += "Xq"
    if w_meta is None:
        layer_type += "W"
    else:
        layer_type += "Wq"
    if b_meta is None:
        layer_type += "B"
    else:
        layer_type += "Bq"

    assert "Wq" in layer_type

    if not bias:
        b_meta = None

    in_features = K
    out_features = N
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    fc_ref = torch.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        device=device,
        dtype=dtype,
    )
    fc_ref.to(device=device)
    fc_from_linear = MXFPLinearPTQ.from_linear(
        layer=fc_ref,
        x_mxfp_meta=x_meta,
        w_mxfp_meta=w_meta,
        b_mxfp_meta=b_meta,
        layer_type=layer_type,
        backend=backend,
    )
    w_scales, w_elements, w_tensor_meta = extract_mxfp_components(
        tensor=fc_ref.weight, block_dim=1, mxfp_meta=w_meta
    )
    b_scales, b_elements, b_tensor_meta = None, None, None
    if bias and b_meta is not None:
        b_scales, b_elements, b_tensor_meta = extract_mxfp_components(
            tensor=fc_ref.bias, block_dim=0, mxfp_meta=b_meta
        )
    fc_from_pre_quantized = MXFPLinearPTQ.from_quantized(
        w_scales=w_scales,
        w_elements=w_elements,
        w_tensor_meta=w_tensor_meta,
        bias=fc_from_linear.bias,
        b_scales=b_scales,
        b_elements=b_elements,
        b_tensor_meta=b_tensor_meta,
        x_mxfp_meta=x_meta,
        layer_type=layer_type,
        backend=backend,
    )

    x = torch.randn(M, K, device=device, dtype=dtype) * 3
    y_from_linear = fc_from_linear(x)
    y_from_pre_quantized = fc_from_pre_quantized(x)
    assert torch.allclose(y_from_linear, y_from_pre_quantized)


@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("ori_device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("new_device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("new_dtype", [None, torch.float16])
@pytest.mark.parametrize("x_meta", [MXFP8_E4M3_fn, None])
@pytest.mark.parametrize("w_meta", [MXFP8_E4M3_fn, None])
@pytest.mark.parametrize("b_meta", [MXFP8_E4M3_fn, None])
def test_mxfp_linear_to(
    has_bias, ori_device, new_device, new_dtype, x_meta, w_meta, b_meta
):
    layer_type = ""
    if x_meta is None:
        layer_type += "X"
    else:
        layer_type += "Xq"
    if w_meta is None:
        layer_type += "W"
    else:
        layer_type += "Wq"
    if b_meta is None:
        layer_type += "B"

    fc1 = torch.nn.Linear(64, 64, bias=has_bias, device=ori_device)
    fc2 = MXFPLinearPTQ.from_linear(
        layer=torch.nn.Linear(64, 64, bias=True, device=ori_device),
        x_mxfp_meta=x_meta,
        w_mxfp_meta=w_meta,
        b_mxfp_meta=b_meta,
        layer_type=layer_type,
        backend="separate",
    )
    model = torch.nn.Sequential(fc1, fc2)
    # fmt: off
    w_ori_dtype = fc2.weight.dtype if fc2.weight is not None else None
    b_ori_dtype = fc2.bias.dtype if fc2.bias is not None else None
    w_sc_ori_dtype = fc2.w_scales.dtype if fc2.w_scales is not None else None
    w_el_ori_dtype = fc2.w_elements.dtype if fc2.w_elements is not None else None
    b_sc_ori_dtype = fc2.b_scales.dtype if fc2.b_scales is not None else None
    b_el_ori_dtype = fc2.b_elements.dtype if fc2.b_elements is not None else None
    # fmt: on

    try:
        model.to(device=new_device, dtype=new_dtype)
    except ChangeDtypeError as e:
        if new_dtype is None:
            raise RuntimeError(
                "ChangeDtypeError should not be raised when new_dtype is None."
            ) from e

    if fc2.weight is not None:
        assert devices_equal(fc2.weight.device, torch.device(new_device))
        assert fc2.weight.dtype == new_dtype if new_dtype is not None else w_ori_dtype
    if fc2.w_scales is not None:
        assert devices_equal(fc2.w_scales.device, torch.device(new_device))
        assert fc2.w_scales.dtype == w_sc_ori_dtype
    if fc2.w_elements is not None:
        assert devices_equal(fc2.w_elements.device, torch.device(new_device))
        assert fc2.w_elements.dtype == w_el_ori_dtype
    if fc2.bias is not None:
        assert devices_equal(fc2.bias.device, torch.device(new_device))
        assert fc2.bias.dtype == new_dtype if new_dtype is not None else b_ori_dtype
    if fc2.b_scales is not None:
        assert devices_equal(fc2.b_scales.device, torch.device(new_device))
        assert fc2.b_scales.dtype == b_sc_ori_dtype
    if fc2.b_elements is not None:
        assert devices_equal(fc2.b_elements.device, torch.device(new_device))
        assert fc2.b_elements.dtype == b_el_ori_dtype


@pytest.mark.parametrize("MNK", [(128, 128, 128)])
@pytest.mark.parametrize("backend", ["separate"])
@pytest.mark.parametrize("x_meta", [None])
@pytest.mark.parametrize("w_meta", [MXFP8_E4M3_fn, None])
@pytest.mark.parametrize("b_meta", [MXFP8_E4M3_fn, None])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("src_device", [torch.device("cpu"), torch.device("cuda:0")])
@pytest.mark.parametrize("dst_device", [torch.device("cpu"), torch.device("cuda:0")])
def test_mxfp_linear_ptq_to(
    MNK,
    backend: str,
    x_meta: MXFPMeta,
    w_meta: MXFPMeta,
    b_meta: MXFPMeta,
    bias: bool,
    src_device: str,
    dst_device: str,
):
    M, N, K = MNK
    layer_type = ""
    if x_meta is None:
        layer_type += "X"
    else:
        layer_type += "Xq"
    if w_meta is None:
        layer_type += "W"
    else:
        layer_type += "Wq"
    if b_meta is None:
        layer_type += "B"
    else:
        layer_type += "Bq"

    if not bias:
        b_meta = None

    in_features = K
    out_features = N

    fc_ref = torch.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        device=src_device,
    )
    fc_mxfp = MXFPLinearPTQ.from_linear(
        layer=fc_ref,
        x_mxfp_meta=x_meta,
        w_mxfp_meta=w_meta,
        b_mxfp_meta=b_meta,
        layer_type=layer_type,
        backend=backend,
    )
    if fc_mxfp.w_scales is not None:
        assert device_str(fc_mxfp.w_scales.device) == device_str(src_device)
    if fc_mxfp.w_elements is not None:
        assert device_str(fc_mxfp.w_elements.device) == device_str(src_device)
    if fc_mxfp.b_scales is not None:
        assert device_str(fc_mxfp.b_scales.device) == device_str(src_device)
    if fc_mxfp.b_elements is not None:
        assert device_str(fc_mxfp.b_elements.device) == device_str(src_device)
    if fc_mxfp.b_tensor_meta is not None:
        assert device_str(fc_mxfp.b_tensor_meta.device) == device_str(src_device)

    fc_mxfp.to(device=dst_device)

    if fc_mxfp.w_scales is not None:
        assert device_str(fc_mxfp.w_scales.device) == device_str(dst_device)
    if fc_mxfp.w_elements is not None:
        assert device_str(fc_mxfp.w_elements.device) == device_str(dst_device)
    if fc_mxfp.w_tensor_meta is not None:
        assert fc_mxfp.w_tensor_meta.device == device_str(dst_device)
    if fc_mxfp.b_scales is not None:
        assert device_str(fc_mxfp.b_scales.device) == device_str(dst_device)
    if fc_mxfp.b_elements is not None:
        assert device_str(fc_mxfp.b_elements.device) == device_str(dst_device)
    if fc_mxfp.b_tensor_meta is not None:
        assert fc_mxfp.b_tensor_meta.device == device_str(dst_device)


if __name__ == "__main__":
    test_mxfp_linear_ptq_from_quantized(
        MNK=(128, 512, 1024),
        backend="separate",
        x_meta=None,
        w_meta=MXFP8_E4M3_fn,
        b_meta=None,
        bias=False,
        dtype=torch.float32,
    )
