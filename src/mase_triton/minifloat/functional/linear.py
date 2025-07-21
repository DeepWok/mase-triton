from typing import Literal

import torch
from torch import Tensor

from ..meta import MinifloatMeta, MinifloatTensorMeta
from .cast import compose_minifloat_component, extract_minifloat_component


def minifloat_linear_XWq(
    x: Tensor,
    w_element: Tensor,
    w_tensor_meta: MinifloatTensorMeta,
    b: Tensor | None,
    b_element: Tensor | None,
    b_tensor_meta: MinifloatTensorMeta | None,
    layer_type: Literal["XWqB", "XWqBq"],
    backend: Literal["separate", "fused"],
) -> Tensor:
    if backend == "separate":
        w_dq = compose_minifloat_component(w_element, w_tensor_meta)
        if "Bq" in layer_type:
            b_dq = None
            if b is not None:
                b_dq = compose_minifloat_component(b_element, b_tensor_meta)
        else:
            b_dq = b
        out = torch.nn.functional.linear(x, w_dq, b_dq)
    else:
        raise NotImplementedError("'fused' not implemented")
    return out


def minifloat_linear_XqWq(
    x: Tensor,
    x_meta: MinifloatMeta,
    w_element: Tensor,
    w_tensor_meta: MinifloatTensorMeta,
    b: Tensor | None,
    b_element: Tensor | None,
    b_tensor_meta: MinifloatTensorMeta | None,
    layer_type: Literal["XqWqB", "XqWqBq"],
    backend: Literal["separate", "fused"],
) -> Tensor:
    if backend == "separate":
        x_element, x_tensor_meta = extract_minifloat_component(x, x_meta)
        x_dq = compose_minifloat_component(x_element, x_tensor_meta)
        w_dq = compose_minifloat_component(w_element, w_tensor_meta)
        if "Bq" in layer_type:
            b_dq = None
            if b is not None:
                b_dq = compose_minifloat_component(b_element, b_tensor_meta)
        else:
            b_dq = b
        out = torch.nn.functional.linear(x_dq, w_dq, b_dq)
    else:
        raise NotImplementedError("'fused' not implemented")
    return out


def minifloat_linear_XqW(
    x: Tensor,
    x_meta: MinifloatMeta,
    w: Tensor | None,
    b: Tensor | None,
    b_element: Tensor | None,
    b_tensor_meta: MinifloatTensorMeta | None,
    layer_type: Literal["XqWB", "XqWBq"],
    backend: Literal["separate", "fused"],
) -> Tensor:
    if backend == "separate":
        x_element, x_tensor_meta = extract_minifloat_component(x, x_meta)
        x_dq = compose_minifloat_component(x_element, x_tensor_meta)
        if "Bq" in layer_type:
            b_dq = None
            if b is not None:
                b_dq = compose_minifloat_component(b_element, b_tensor_meta)
        else:
            b_dq = b
        out = torch.nn.functional.linear(x_dq, w, b_dq)
    else:
        raise NotImplementedError("'fused' not implemented")
    return out


def minifloat_linear_XW(
    x: Tensor,
    w: Tensor | None,
    b: Tensor | None,
    b_element: Tensor | None,
    b_tensor_meta: MinifloatTensorMeta | None,
    layer_type: Literal["XWB", "XWBq"],
    backend: Literal["separate", "fused"],
) -> Tensor:
    if backend == "separate":
        if "Bq" in layer_type:
            b_dq = None
            if b is not None:
                b_dq = compose_minifloat_component(b_element, b_tensor_meta)
        else:
            b_dq = b
        out = torch.nn.functional.linear(x, w, b_dq)
    else:
        raise NotImplementedError("'fused' not implemented")
    return out


def minifloat_linear(
    x: Tensor,
    x_meta: MinifloatMeta | None,
    w: Tensor | None,
    w_element: Tensor | None,
    w_tensor_meta: MinifloatTensorMeta | None,
    b: Tensor | None,
    b_element: Tensor | None,
    b_tensor_meta: MinifloatTensorMeta | None,
    layer_type: Literal[
        "XWB", "XWBq", "XWqB", "XWqBq", "XqWB", "XqWBq", "XqWqB", "XqWqBq"
    ],
    backend: Literal["separate", "fused"],
) -> Tensor:
    """
    Perform minifloat linear operation based on the layer type.
    """
    if "XWq" in layer_type:
        out = minifloat_linear_XWq(
            x=x,
            w_element=w_element,
            w_tensor_meta=w_tensor_meta,
            b=b,
            b_element=b_element,
            b_tensor_meta=b_tensor_meta,
            backend=backend,
            layer_type=layer_type,
        )
    elif "XqWq" in layer_type:
        out = minifloat_linear_XqWq(
            x=x,
            x_meta=x_meta,
            w_element=w_element,
            w_tensor_meta=w_tensor_meta,
            b=b,
            b_element=b_element,
            b_tensor_meta=b_tensor_meta,
            backend=backend,
            layer_type=layer_type,
        )
    elif "XqW" in layer_type:
        out = minifloat_linear_XqW(
            x=x,
            x_meta=x_meta,
            w=w,
            b=b,
            b_element=b_element,
            b_tensor_meta=b_tensor_meta,
            backend=backend,
            layer_type=layer_type,
        )
    elif "XW" in layer_type:
        out = minifloat_linear_XW(
            x=x,
            w=w,
            b=b,
            b_element=b_element,
            b_tensor_meta=b_tensor_meta,
            backend=backend,
            layer_type=layer_type,
        )
    else:
        raise NotImplementedError(f"Layer type {layer_type} not implemented")
    return out
