from typing import Literal

import torch
from torch import Tensor

from ..meta import MinifloatMeta
from .cast import compose_minifloat_component, extract_minifloat_component


def minifloat_matmul(
    input: Tensor,
    other: Tensor,
    input_meta: MinifloatMeta | None,
    other_meta: MinifloatMeta | None,
    func_type: Literal["XW", "XqW", "XWq", "XqWq"],
    backend: Literal["separate", "fused"],
) -> Tensor:
    if "Xq" in func_type:
        assert input_meta is not None
        input_elements, input_tensor_meta = extract_minifloat_component(
            input, minifloat_meta=input_meta
        )
        input = compose_minifloat_component(input_elements, input_tensor_meta)
    if "Wq" in func_type:
        assert other_meta is not None
        other_elements, other_tensor_meta = extract_minifloat_component(
            other, minifloat_meta=other_meta
        )
        other = compose_minifloat_component(other_elements, other_tensor_meta)

    if backend == "separate":
        out = torch.matmul(input, other)
    else:
        raise NotImplementedError("'fused' not implemented")
    return out
