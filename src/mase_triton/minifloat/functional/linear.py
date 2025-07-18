from typing import Literal

import torch
from torch import Tensor

from ..meta import minifloatTensorMeta
from .cast import compose_minifloat_tensor, extract_minifloat_components


def minifloat_linear_XWq(
    x: Tensor,
    w_elements: Tensor,
    w_tensor_meta: minifloatTensorMeta,
    b: Tensor | None,
    b_elements: Tensor | None,
    b_tensor_meta: minifloatTensorMeta | None,
    layer_type: Literal["XWqB", "XWqBq"],
    backend: Literal["separate", "fused"],
) -> Tensor:
    if backend == "separate":
        w_dq = compose_minifloat_tensor(w_elements, w_tensor_meta)
        if "Bq" in layer_type:
            b_dq = None
            if b is not None:
                b_dq = compose_minifloat_tensor(b_elements, b_tensor_meta)
        else:
            b_dq = b
        out = torch.nn.functional.linear(x, w_dq, b_dq)
    else:
        raise NotImplementedError("'fused' not implemented")
    return out

def minifloat_linear_XW(
    x: Tensor,
    w: Tensor | None,
    b: Tensor | None,
    b_elements: Tensor | None,
    b_tensor_meta: minifloatTensorMeta | None,
    layer_type: Literal["XWB", "XWBq"],
    backend: Literal["separate", "fused"],
) -> Tensor:
    if backend == "separate":
        if "Bq" in layer_type:
            b_dq = None
            if b is not None:
                b_dq = compose_minifloat_tensor(b_elements, b_tensor_meta)
        else:
            b_dq = b
        out = torch.nn.functional.linear(x, w, b_dq)
    else:
        raise NotImplementedError("'fused' not implemented")
    return out

def parse_minifloat_linear_type(
    x_meta,
    w_tensor_meta: minifloatTensorMeta | None,
    b_tensor_meta: minifloatTensorMeta | None,
) -> str:
    layer_type = ""
    # if x_meta is None:
    #     layer_type += "X"
    # else:
    #     layer_type += "Xq"
    if w_tensor_meta is None:
        layer_type += "W"
    else:
        layer_type += "Wq"
    if b_tensor_meta is None:
        layer_type += "B"
    else:
        layer_type += "Bq"
    return layer_type


def minifloat_linear(
    x: Tensor,
    w: Tensor | None,
    w_scales: Tensor | None,
    w_elements: Tensor | None,
    w_tensor_meta: minifloatTensorMeta | None,
    b: Tensor | None,
    b_scales: Tensor | None,
    b_elements: Tensor | None,
    b_tensor_meta: minifloatTensorMeta | None,
    layer_type: Literal[
        "XWB", "XWBq", "XWqB", "XWqBq"
    ],
    backend: Literal["separate", "fused"],
) -> Tensor:
    """
    Perform minifloat linear operation based on the layer type.
    """
    if "XWq" in layer_type:
        out = minifloat_linear_XWq(
            x=x,
            w_elements=w_elements,
            w_tensor_meta=w_tensor_meta,
            b=b,
            b_elements=b_elements,
            b_tensor_meta=b_tensor_meta,
            backend=backend,
            layer_type=layer_type,
        )
    elif "XW" in layer_type:
        out = mxfp_linear_XW(
            x=x,
            w=w,
            b=b,
            b_elements=b_elements,
            b_tensor_meta=b_tensor_meta,
            backend=backend,
            layer_type=layer_type,
        )
    else:
        raise NotImplementedError(f"Layer type {layer_type} not implemented")
    return out
