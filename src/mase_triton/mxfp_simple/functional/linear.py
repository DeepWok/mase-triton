from typing import Literal

import torch
from torch import Tensor

from ..meta import MXFPMeta, MXFPTensorMeta
from .cast import compose_mxfp_tensor, extract_mxfp_components


def mxfp_linear_XWq(
    x: Tensor,
    w_scales: Tensor,
    w_elements: Tensor,
    w_tensor_meta: MXFPTensorMeta,
    b: Tensor | None,
    b_scales: Tensor | None,
    b_elements: Tensor | None,
    b_tensor_meta: MXFPTensorMeta | None,
    layer_type: Literal["XWqB", "XWqBq"],
    backend: Literal["separate", "fused"],
) -> Tensor:
    if backend == "separate":
        w_dq = compose_mxfp_tensor(w_scales, w_elements, w_tensor_meta)
        if "Bq" in layer_type:
            b_dq = None
            if b is not None:
                b_dq = compose_mxfp_tensor(b_scales, b_elements, b_tensor_meta)
        else:
            b_dq = b
        out = torch.nn.functional.linear(x, w_dq, b_dq)
    else:
        raise NotImplementedError("'fused' not implemented")
    return out


def mxfp_linear_XqWq(
    x: Tensor,
    x_meta: MXFPMeta,
    w_scales: Tensor,
    w_elements: Tensor,
    w_tensor_meta: MXFPTensorMeta,
    b: Tensor | None,
    b_scales: Tensor | None,
    b_elements: Tensor | None,
    b_tensor_meta: MXFPTensorMeta | None,
    layer_type: Literal["XqWqB", "XqWqBq"],
    backend: Literal["separate", "fused"],
) -> Tensor:
    if backend == "separate":
        x_scales, x_elements, x_tensor_meta = extract_mxfp_components(
            x, block_dim=-1, mxfp_meta=x_meta
        )
        x_dq = compose_mxfp_tensor(x_scales, x_elements, x_tensor_meta)
        w_dq = compose_mxfp_tensor(w_scales, w_elements, w_tensor_meta)
        if "Bq" in layer_type:
            b_dq = None
            if b is not None:
                b_dq = compose_mxfp_tensor(b_scales, b_elements, b_tensor_meta)
        else:
            b_dq = b
        out = torch.nn.functional.linear(x_dq, w_dq, b_dq)
    else:
        raise NotImplementedError("'fused' not implemented")
    return out


def mxfp_linear_XqW(
    x: Tensor,
    x_meta: MXFPMeta,
    w: Tensor | None,
    b: Tensor | None,
    b_scales: Tensor | None,
    b_elements: Tensor | None,
    b_tensor_meta: MXFPTensorMeta | None,
    layer_type: Literal["XqWB", "XqWBq"],
    backend: Literal["separate", "fused"],
) -> Tensor:
    if backend == "separate":
        x_scales, x_elements, x_tensor_meta = extract_mxfp_components(
            x, block_dim=-1, mxfp_meta=x_meta
        )
        x_dq = compose_mxfp_tensor(x_scales, x_elements, x_tensor_meta)
        if "Bq" in layer_type:
            b_dq = None
            if b is not None:
                b_dq = compose_mxfp_tensor(b_scales, b_elements, b_tensor_meta)
        else:
            b_dq = b
        out = torch.nn.functional.linear(x_dq, w, b_dq)
    else:
        raise NotImplementedError("'fused' not implemented")
    return out


def mxfp_linear_XW(
    x: Tensor,
    w: Tensor | None,
    b: Tensor | None,
    b_scales: Tensor | None,
    b_elements: Tensor | None,
    b_tensor_meta: MXFPTensorMeta | None,
    layer_type: Literal["XWB", "XWBq"],
    backend: Literal["separate", "fused"],
) -> Tensor:
    if backend == "separate":
        if "Bq" in layer_type:
            b_dq = None
            if b is not None:
                b_dq = compose_mxfp_tensor(b_scales, b_elements, b_tensor_meta)
        else:
            b_dq = b
        out = torch.nn.functional.linear(x, w, b_dq)
    else:
        raise NotImplementedError("'fused' not implemented")
    return out


def parse_mxfp_linear_type(
    x_meta: MXFPMeta | None,
    w_tensor_meta: MXFPTensorMeta | None,
    b_tensor_meta: MXFPTensorMeta | None,
) -> str:
    layer_type = ""
    if x_meta is None:
        layer_type += "X"
    else:
        layer_type += "Xq"
    if w_tensor_meta is None:
        layer_type += "W"
    else:
        layer_type += "Wq"
    if b_tensor_meta is None:
        layer_type += "B"
    else:
        layer_type += "Bq"
    return layer_type


def mxfp_linear(
    x: Tensor,
    x_meta: MXFPMeta | None,
    w: Tensor | None,
    w_scales: Tensor | None,
    w_elements: Tensor | None,
    w_tensor_meta: MXFPTensorMeta | None,
    b: Tensor | None,
    b_scales: Tensor | None,
    b_elements: Tensor | None,
    b_tensor_meta: MXFPTensorMeta | None,
    layer_type: Literal[
        "XWB", "XWBq", "XWqB", "XWqBq", "XqWB", "XqWBq", "XqWqB", "XqWqBq"
    ],
    backend: Literal["separate", "fused"],
) -> Tensor:
    """
    Perform MXFP linear operation based on the layer type.
    """
    if "XWq" in layer_type:
        out = mxfp_linear_XWq(
            x=x,
            w_scales=w_scales,
            w_elements=w_elements,
            w_tensor_meta=w_tensor_meta,
            b=b,
            b_scales=b_scales,
            b_elements=b_elements,
            b_tensor_meta=b_tensor_meta,
            backend=backend,
            layer_type=layer_type,
        )
    elif "XqWq" in layer_type:
        out = mxfp_linear_XqWq(
            x=x,
            x_meta=x_meta,
            w_scales=w_scales,
            w_elements=w_elements,
            w_tensor_meta=w_tensor_meta,
            b=b,
            b_scales=b_scales,
            b_elements=b_elements,
            b_tensor_meta=b_tensor_meta,
            backend=backend,
            layer_type=layer_type,
        )
    elif "XqW" in layer_type:
        out = mxfp_linear_XqW(
            x=x,
            x_meta=x_meta,
            w=w,
            b=b,
            b_scales=b_scales,
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
            b_scales=b_scales,
            b_elements=b_elements,
            b_tensor_meta=b_tensor_meta,
            backend=backend,
            layer_type=layer_type,
        )
    else:
        raise NotImplementedError(f"Layer type {layer_type} not implemented")
    return out
