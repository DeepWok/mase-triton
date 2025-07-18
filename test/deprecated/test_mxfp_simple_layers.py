import pytest
import torch

from mase_triton._mxfp_simple.layers import MXFPLinearPTQ
from mase_triton._mxfp_simple.meta import OCP_MXFP8_E4M3, MXFPMeta
from mase_triton._mxfp_simple.utils import ChangeDtypeError, devices_equal
from mase_triton.utils.train_utils import set_seed

set_seed(42)


@pytest.mark.parametrize("backend", ["separate"])
@pytest.mark.parametrize("x_meta", [OCP_MXFP8_E4M3, None])
@pytest.mark.parametrize("w_meta", [OCP_MXFP8_E4M3, None])
@pytest.mark.parametrize("b_meta", [OCP_MXFP8_E4M3, None])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_mxfp_linear_ptq(
    backend: str,
    x_meta: MXFPMeta,
    w_meta: MXFPMeta,
    b_meta: MXFPMeta,
    bias: bool,
    dtype: torch.dtype,
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
    else:
        layer_type += "Bq"

    if not bias:
        b_meta = None

    in_features = 512
    out_features = 64
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

    x = torch.randn(8, in_features, device=device, dtype=dtype)
    y_ref = fc_ref(x)
    y_mxfp = fc_mxfp(x)

    avg_err = (y_ref - y_mxfp).abs().mean().item()
    avg_err_ratio = avg_err / y_ref.abs().mean().item()

    assert avg_err_ratio < 0.15


@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("ori_device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("new_device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("new_dtype", [None, torch.float16])
@pytest.mark.parametrize("x_meta", [OCP_MXFP8_E4M3, None])
@pytest.mark.parametrize("w_meta", [OCP_MXFP8_E4M3, None])
@pytest.mark.parametrize("b_meta", [OCP_MXFP8_E4M3, None])
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


if __name__ == "__main__":
    # test_mxfp_linear_ptq(
    # backend="separate",
    # x_meta=None,
    # w_meta=OCP_MXFP8_E4M3,
    # b_meta=None,
    # bias=False,
    # dtype=torch.bfloat16,
    # )
    test_mxfp_linear_to(
        has_bias=True,
        ori_device="cpu",
        new_device="cuda",
        new_dtype=torch.float16,
        x_meta=OCP_MXFP8_E4M3,
        w_meta=OCP_MXFP8_E4M3,
        b_meta=OCP_MXFP8_E4M3,
    )
