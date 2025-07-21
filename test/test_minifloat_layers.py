import pytest
import torch

from mase_triton.minifloat.functional import extract_minifloat_component
from mase_triton.minifloat.layers import MinifloatLinearPTQ
from mase_triton.minifloat.meta import (
    FP4_E2M1_fn,
    FP6_E2M3_fn,
    FP6_E3M2_fn,
    FP8_E4M3_fn,
    FP8_E5M2_fn,
    MinifloatMeta,
)
from mase_triton.utils.meta import device_str
from mase_triton.utils.qlayer import ChangeDtypeError, devices_equal
from mase_triton.utils.train_utils import set_seed

set_seed(42)


@pytest.mark.parametrize("MNK", [(128, 512, 1024)])
@pytest.mark.parametrize("backend", ["separate"])
@pytest.mark.parametrize(
    "x_meta",
    [
        FP8_E4M3_fn,
        FP8_E5M2_fn,
        None,
    ],
)
@pytest.mark.parametrize(
    "w_meta",
    [
        FP8_E4M3_fn,
        FP8_E5M2_fn,
        None,
    ],
)
@pytest.mark.parametrize("b_meta", [FP8_E4M3_fn, None])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_minifloat_linear_ptq_from_linear(
    MNK,
    backend: str,
    x_meta: MinifloatMeta,
    w_meta: MinifloatMeta,
    b_meta: MinifloatMeta,
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
    fc_minifloat = MinifloatLinearPTQ.from_linear(
        layer=fc_ref,
        x_minifloat_meta=x_meta,
        w_minifloat_meta=w_meta,
        b_minifloat_meta=b_meta,
        layer_type=layer_type,
        backend=backend,
    )

    x = torch.randn(M, K, device=device, dtype=dtype) * 3
    y_ref = fc_ref(x)
    y_minifloat = fc_minifloat(x)

    avg_err = (y_ref - y_minifloat).abs().mean().item()
    avg_err_ratio = avg_err / y_ref.abs().mean().item()

    print(
        f"Average error ratio for {layer_type} with {x_meta}, {w_meta}, {b_meta}: {avg_err_ratio:.4f}"
    )
    assert avg_err_ratio < 0.25


@pytest.mark.parametrize("MNK", [(128, 512, 1024)])
@pytest.mark.parametrize("backend", ["separate"])
@pytest.mark.parametrize("x_meta", [None])
@pytest.mark.parametrize(
    "w_meta",
    [
        FP8_E4M3_fn,
        FP8_E5M2_fn,
        FP6_E3M2_fn,
        FP6_E2M3_fn,
        FP4_E2M1_fn,
    ],
)
@pytest.mark.parametrize("b_meta", [FP4_E2M1_fn, None])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.no_grad()
def test_minifloat_linear_ptq_from_quantized(
    MNK,
    backend: str,
    x_meta: MinifloatMeta,
    w_meta: MinifloatMeta,
    b_meta: MinifloatMeta,
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
    fc_from_linear = MinifloatLinearPTQ.from_linear(
        layer=fc_ref,
        x_minifloat_meta=x_meta,
        w_minifloat_meta=w_meta,
        b_minifloat_meta=b_meta,
        layer_type=layer_type,
        backend=backend,
    )
    w_element, w_tensor_meta = extract_minifloat_component(
        tensor=fc_ref.weight, minifloat_meta=w_meta
    )
    b_element, b_tensor_meta = None, None
    if bias and b_meta is not None:
        b_element, b_tensor_meta = extract_minifloat_component(
            tensor=fc_ref.bias, minifloat_meta=b_meta
        )
    fc_from_pre_quantized = MinifloatLinearPTQ.from_quantized(
        w_element=w_element,
        w_tensor_meta=w_tensor_meta,
        bias=fc_from_linear.bias,
        b_element=b_element,
        b_tensor_meta=b_tensor_meta,
        x_minifloat_meta=x_meta,
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
@pytest.mark.parametrize("x_meta", [FP8_E4M3_fn, None])
@pytest.mark.parametrize("w_meta", [FP8_E4M3_fn, None])
@pytest.mark.parametrize("b_meta", [FP8_E4M3_fn, None])
def test_minifloat_linear_to(
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
    fc2 = MinifloatLinearPTQ.from_linear(
        layer=torch.nn.Linear(64, 64, bias=True, device=ori_device),
        x_minifloat_meta=x_meta,
        w_minifloat_meta=w_meta,
        b_minifloat_meta=b_meta,
        layer_type=layer_type,
        backend="separate",
    )
    model = torch.nn.Sequential(fc1, fc2)
    # fmt: off
    w_ori_dtype = fc2.weight.dtype if fc2.weight is not None else None
    b_ori_dtype = fc2.bias.dtype if fc2.bias is not None else None
    w_el_ori_dtype = fc2.w_element.dtype if fc2.w_element is not None else None
    b_el_ori_dtype = fc2.b_element.dtype if fc2.b_element is not None else None
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
    if fc2.w_element is not None:
        assert devices_equal(fc2.w_element.device, torch.device(new_device))
        assert fc2.w_element.dtype == w_el_ori_dtype
    if fc2.bias is not None:
        assert devices_equal(fc2.bias.device, torch.device(new_device))
        assert fc2.bias.dtype == new_dtype if new_dtype is not None else b_ori_dtype
    if fc2.b_element is not None:
        assert devices_equal(fc2.b_element.device, torch.device(new_device))
        assert fc2.b_element.dtype == b_el_ori_dtype


@pytest.mark.parametrize("MNK", [(128, 128, 128)])
@pytest.mark.parametrize("backend", ["separate"])
@pytest.mark.parametrize("x_meta", [None])
@pytest.mark.parametrize("w_meta", [FP8_E4M3_fn, None])
@pytest.mark.parametrize("b_meta", [FP8_E4M3_fn, None])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("src_device", [torch.device("cpu"), torch.device("cuda:0")])
@pytest.mark.parametrize("dst_device", [torch.device("cpu"), torch.device("cuda:0")])
def test_minifloat_linear_ptq_to(
    MNK,
    backend: str,
    x_meta: MinifloatMeta,
    w_meta: MinifloatMeta,
    b_meta: MinifloatMeta,
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
    fc_minifloat = MinifloatLinearPTQ.from_linear(
        layer=fc_ref,
        x_minifloat_meta=x_meta,
        w_minifloat_meta=w_meta,
        b_minifloat_meta=b_meta,
        layer_type=layer_type,
        backend=backend,
    )
    if fc_minifloat.w_element is not None:
        assert device_str(fc_minifloat.w_element.device) == device_str(src_device)
    if fc_minifloat.b_element is not None:
        assert device_str(fc_minifloat.b_element.device) == device_str(src_device)
    if fc_minifloat.b_tensor_meta is not None:
        assert device_str(fc_minifloat.b_tensor_meta.device) == device_str(src_device)

    fc_minifloat.to(device=dst_device)

    if fc_minifloat.w_element is not None:
        assert device_str(fc_minifloat.w_element.device) == device_str(dst_device)
    if fc_minifloat.w_tensor_meta is not None:
        assert fc_minifloat.w_tensor_meta.device == device_str(dst_device)
    if fc_minifloat.b_element is not None:
        assert device_str(fc_minifloat.b_element.device) == device_str(dst_device)
    if fc_minifloat.b_tensor_meta is not None:
        assert fc_minifloat.b_tensor_meta.device == device_str(dst_device)


if __name__ == "__main__":
    test_minifloat_linear_ptq_from_quantized(
        MNK=(128, 512, 1024),
        backend="separate",
        x_meta=None,
        w_meta=FP8_E4M3_fn,
        b_meta=None,
        bias=False,
        dtype=torch.float32,
    )
