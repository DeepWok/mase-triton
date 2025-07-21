from typing import Literal

import torch
from torch import Tensor, nn

from ..utils.qlayer import ChangeDtypeError
from . import functional as MXFP_F
from .meta import MXFPMeta, MXFPTensorMeta


class MXFPLinearPTQ(nn.Module):
    in_features: int
    out_features: int

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor | None,
        x_mxfp_meta: MXFPMeta | None,
        w_mxfp_meta: MXFPMeta | None,
        b_mxfp_meta: MXFPMeta | None,
        layer_type: Literal[
            "XWB", "XWBq", "XWqB", "XWqBq", "XqWB", "XqWBq", "XqWqB", "XqWqBq"
        ],
        backend: Literal["separate", "fused"] = "fused",
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        w_scales: Tensor | None = None,
        w_elements: Tensor | None = None,
        w_tensor_meta: MXFPTensorMeta | None = None,
        b_scales: Tensor | None = None,
        b_elements: Tensor | None = None,
        b_tensor_meta: MXFPTensorMeta | None = None,
    ):
        super().__init__()
        t_args = {"dtype": dtype, "device": device}
        assert weight is None or weight.ndim == 2
        assert bias is None or bias.ndim == 1

        assert (weight is not None) ^ (
            w_scales is not None
            and w_elements is not None
            and w_tensor_meta is not None
        )
        if (bias is not None) and (
            b_scales is not None
            and b_elements is not None
            and b_tensor_meta is not None
        ):
            raise ValueError(
                "Either bias or b_scales, b_elements, b_tensor_meta must be None, not both."
            )
        if weight is None:
            out_features, in_features = w_tensor_meta.shape
        else:
            in_features, out_features = weight.shape[1], weight.shape[0]
        assert bias is None or bias.shape[0] == out_features
        self.in_features = in_features
        self.out_features = out_features
        self.x_mxfp_meta = x_mxfp_meta
        self.w_mxfp_meta = w_mxfp_meta
        self.b_mxfp_meta = b_mxfp_meta
        self.layer_type = layer_type
        self.backend = backend
        self.dtype = dtype
        self.device = device

        self.weight = None
        self.w_scales, self.w_elements, self.w_tensor_meta = None, None, None
        self.bias = None
        self.b_scales, self.b_elements, self.b_tensor_meta = None, None, None

        if "Wq" in layer_type:
            if w_scales is None:
                w_scales, w_elements, w_tensor_meta = MXFP_F.extract_mxfp_components(
                    weight, block_dim=1, mxfp_meta=w_mxfp_meta
                )
            self.w_scales = nn.Parameter(w_scales, requires_grad=False)
            self.w_elements = nn.Parameter(w_elements, requires_grad=False)
            self.w_tensor_meta = w_tensor_meta
        else:
            assert weight is not None
            self.weight = nn.Parameter(weight.to(**t_args), requires_grad=False)

        if "Bq" in layer_type:
            if bias is not None or b_scales is not None:
                if b_scales is None:
                    b_scales, b_elements, b_tensor_meta = (
                        MXFP_F.extract_mxfp_components(
                            bias, block_dim=0, mxfp_meta=b_mxfp_meta
                        )
                    )
                self.b_scales = nn.Parameter(b_scales, requires_grad=False)
                self.b_elements = nn.Parameter(b_elements, requires_grad=False)
                self.b_tensor_meta = b_tensor_meta
        else:
            if bias is not None:
                self.bias = nn.Parameter(bias.to(**t_args), requires_grad=False)

    def _apply(self, fn, recurse=True):
        # fmt: off
        w_sc_ori_dtype = self.w_scales.dtype if self.w_scales is not None else None
        w_el_ori_dtype = self.w_elements.dtype if self.w_elements is not None else None
        b_sc_ori_dtype = self.b_scales.dtype if self.b_scales is not None else None
        b_el_ori_dtype = self.b_elements.dtype if self.b_elements is not None else None
        # fmt: on
        r_val = super()._apply(fn, recurse)
        for t_name, ori_type in zip(
            ["w_scales", "w_elements", "b_scales", "b_elements"],
            [w_sc_ori_dtype, w_el_ori_dtype, b_sc_ori_dtype, b_el_ori_dtype],
        ):
            t = getattr(self, t_name)
            if t is not None and t.dtype != ori_type:
                raise ChangeDtypeError(
                    f"Changing dtype of {t_name} from {ori_type} to {t.dtype} is not allowed."
                )
        if self.w_tensor_meta is not None and self.w_scales is not None:
            self.w_tensor_meta = self.w_tensor_meta.create(device=self.w_scales.device)
        if self.b_tensor_meta is not None and self.b_scales is not None:
            self.b_tensor_meta = self.b_tensor_meta.create(device=self.b_scales.device)
        return r_val

    @torch.no_grad()
    def forward(self, input: Tensor) -> Tensor:
        return MXFP_F.mxfp_linear(
            x=input,
            x_meta=self.x_mxfp_meta,
            w=self.weight,
            w_scales=self.w_scales,
            w_elements=self.w_elements,
            w_tensor_meta=self.w_tensor_meta,
            b=self.bias,
            b_scales=self.b_scales,
            b_elements=self.b_elements,
            b_tensor_meta=self.b_tensor_meta,
            layer_type=self.layer_type,
            backend=self.backend,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
            f"layer_type={self.layer_type}, "
            f"w_mxfp_meta={self.w_mxfp_meta}, x_mxfp_meta={self.x_mxfp_meta}, "
            f"b_mxfp_meta={self.b_mxfp_meta}"
        )

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        x_mxfp_meta: MXFPMeta | None,
        w_mxfp_meta: MXFPTensorMeta | None,
        b_mxfp_meta: MXFPTensorMeta | None,
        layer_type: Literal[
            "XWB", "XWBq", "XWqB", "XWqBq", "XqWB", "XqWBq", "XqWqB", "XqWqBq"
        ],
        backend: Literal["separate", "fused"],
    ):
        """
        Create an MXFPLinearPTQ instance from a PyTorch Linear layer.
        """
        assert isinstance(layer, nn.Linear), "layer must be an instance of nn.Linear"
        with torch.no_grad():
            return cls(
                weight=layer.weight.clone(),
                bias=layer.bias.clone() if layer.bias is not None else None,
                x_mxfp_meta=x_mxfp_meta,
                w_mxfp_meta=w_mxfp_meta,
                b_mxfp_meta=b_mxfp_meta,
                layer_type=layer_type,
                backend=backend,
            )

    @classmethod
    def from_quantized(
        cls,
        w_scales: Tensor | None,
        w_elements: Tensor | None,
        w_tensor_meta: MXFPTensorMeta | None,
        bias: Tensor | None,
        b_scales: Tensor | None,
        b_elements: Tensor | None,
        b_tensor_meta: MXFPTensorMeta | None,
        x_mxfp_meta: MXFPMeta | None,
        layer_type: Literal["XWqB", "XWqBq", "XqWqB", "XqWqBq"],
        backend: Literal["separate", "fused"],
    ):
        assert (
            w_scales is not None
            and w_elements is not None
            and w_tensor_meta is not None
        )

        return cls(
            weight=None,
            bias=bias,
            x_mxfp_meta=x_mxfp_meta,
            w_mxfp_meta=None,
            b_mxfp_meta=None,
            layer_type=layer_type,
            backend=backend,
            dtype=getattr(torch, w_tensor_meta.dtype),
            device=torch.device(w_tensor_meta.device),
            w_scales=w_scales,
            w_elements=w_elements,
            w_tensor_meta=w_tensor_meta,
            b_scales=b_scales,
            b_elements=b_elements,
            b_tensor_meta=b_tensor_meta,
        )
