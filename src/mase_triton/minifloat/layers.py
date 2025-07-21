from typing import Literal

import torch
from torch import Tensor, nn

from ..utils.qlayer import ChangeDtypeError
from . import functional as MinifloatF
from .meta import MinifloatMeta, MinifloatTensorMeta


class MinifloatLinearPTQ(nn.Module):
    in_features: int
    out_features: int

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor | None,
        x_minifloat_meta: MinifloatMeta | None,
        w_minifloat_meta: MinifloatMeta | None,
        b_minifloat_meta: MinifloatMeta | None,
        layer_type: Literal[
            "XWB", "XWBq", "XWqB", "XWqBq", "XqWB", "XqWBq", "XqWqB", "XqWqBq"
        ],
        backend: Literal["separate", "fused"] = "fused",
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        w_element: Tensor | None = None,
        w_tensor_meta: MinifloatTensorMeta | None = None,
        b_element: Tensor | None = None,
        b_tensor_meta: MinifloatTensorMeta | None = None,
    ):
        super().__init__()
        t_args = {"dtype": dtype, "device": device}
        assert weight is None or weight.ndim == 2
        assert bias is None or bias.ndim == 1

        assert (weight is not None) ^ (
            w_element is not None and w_tensor_meta is not None
        )
        if (bias is not None) and (b_element is not None and b_tensor_meta is not None):
            raise ValueError(
                "Either bias or b_element, b_tensor_meta must be None, not both."
            )
        if weight is None:
            out_features, in_features = w_tensor_meta.shape
        else:
            in_features, out_features = weight.shape[1], weight.shape[0]
        assert bias is None or bias.shape[0] == out_features

        self.in_features = in_features
        self.out_features = out_features
        self.x_minifloat_meta = x_minifloat_meta
        self.w_minifloat_meta = w_minifloat_meta
        self.b_minifloat_meta = b_minifloat_meta
        self.layer_type = layer_type
        self.backend = backend
        self.dtype = dtype
        self.device = device

        self.weight = None
        self.w_element, self.w_tensor_meta = None, None
        self.bias = None
        self.b_element, self.b_tensor_meta = None, None

        if "Wq" in layer_type:
            if w_element is None:
                w_element, w_tensor_meta = MinifloatF.extract_minifloat_component(
                    weight, w_minifloat_meta
                )
            self.w_element = nn.Parameter(w_element, requires_grad=False)
            self.w_tensor_meta = w_tensor_meta
        else:
            assert weight is not None
            self.weight = nn.Parameter(weight.to(**t_args), requires_grad=False)

        if "Bq" in layer_type:
            if bias is not None or b_element is not None:
                if b_element is None:
                    b_element, b_tensor_meta = MinifloatF.extract_minifloat_component(
                        bias, b_minifloat_meta
                    )
                self.b_element = nn.Parameter(b_element, requires_grad=False)
                self.b_tensor_meta = b_tensor_meta
        else:
            if bias is not None:
                self.bias = nn.Parameter(bias.to(**t_args), requires_grad=False)

    def _apply(self, fn, recurse=True):
        # fmt: off
        w_el_ori_dtype = self.w_element.dtype if self.w_element is not None else None
        b_el_ori_dtype = self.b_element.dtype if self.b_element is not None else None
        # fmt: on
        r_val = super()._apply(fn, recurse)
        for t_name, ori_type in zip(
            ["w_element", "b_element"],
            [w_el_ori_dtype, b_el_ori_dtype],
        ):
            t = getattr(self, t_name)
            if t is not None and t.dtype != ori_type:
                raise ChangeDtypeError(
                    f"Changing dtype of {t_name} from {ori_type} to {t.dtype} is not allowed."
                )
        if self.w_tensor_meta is not None and self.w_element is not None:
            self.w_tensor_meta = self.w_tensor_meta.create(device=self.w_element.device)
        if self.b_tensor_meta is not None and self.b_element is not None:
            self.b_tensor_meta = self.b_tensor_meta.create(device=self.b_element.device)
        return r_val

    @torch.no_grad()
    def forward(self, input: Tensor) -> Tensor:
        return MinifloatF.minifloat_linear(
            x=input,
            x_meta=self.x_minifloat_meta,
            w=self.weight,
            w_element=self.w_element,
            w_tensor_meta=self.w_tensor_meta,
            b=self.bias,
            b_element=self.b_element,
            b_tensor_meta=self.b_tensor_meta,
            layer_type=self.layer_type,
            backend=self.backend,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
            f"layer_type={self.layer_type}, "
            f"w_minifloat_meta={self.w_minifloat_meta}, x_minifloat_meta={self.x_minifloat_meta}, "
            f"b_minifloat_meta={self.b_minifloat_meta}"
        )

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        x_minifloat_meta: MinifloatMeta | None,
        w_minifloat_meta: MinifloatMeta | None,
        b_minifloat_meta: MinifloatMeta | None,
        layer_type: Literal[
            "XWB", "XWBq", "XWqB", "XWqBq", "XqWB", "XqWBq", "XqWqB", "XqWqBq"
        ],
        backend: Literal["separate", "fused"],
    ):
        """
        Create a MinifloatLinearPTQ instance from a PyTorch Linear layer.
        """
        assert isinstance(layer, nn.Linear), "layer must be an instance of nn.Linear"
        with torch.no_grad():
            return cls(
                weight=layer.weight.clone(),
                bias=layer.bias.clone() if layer.bias is not None else None,
                x_minifloat_meta=x_minifloat_meta,
                w_minifloat_meta=w_minifloat_meta,
                b_minifloat_meta=b_minifloat_meta,
                layer_type=layer_type,
                backend=backend,
            )

    @classmethod
    def from_quantized(
        cls,
        w_element: Tensor | None,
        w_tensor_meta: MinifloatTensorMeta | None,
        bias: Tensor | None,
        b_element: Tensor | None,
        b_tensor_meta: MinifloatTensorMeta | None,
        x_minifloat_meta: MinifloatMeta | None,
        layer_type: Literal["XWqB", "XWqBq", "XqWqB", "XqWqBq"],
        backend: Literal["separate", "fused"],
    ):
        assert w_element is not None and w_tensor_meta is not None

        return cls(
            weight=None,
            bias=bias,
            x_minifloat_meta=x_minifloat_meta,
            w_minifloat_meta=None,
            b_minifloat_meta=None,
            layer_type=layer_type,
            backend=backend,
            dtype=getattr(torch, w_tensor_meta.dtype),
            device=torch.device(w_tensor_meta.device),
            w_element=w_element,
            w_tensor_meta=w_tensor_meta,
            b_element=b_element,
            b_tensor_meta=b_tensor_meta,
        )
