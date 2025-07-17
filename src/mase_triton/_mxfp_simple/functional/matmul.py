from typing import Literal

import torch
from torch import Tensor

from ..meta import MXFPMeta
from .cast import compose_mxfp_tensor, extract_mxfp_components


def mxfp_matmul(
    input: Tensor,
    other: Tensor,
    input_meta: MXFPMeta | None,
    other_meta: MXFPMeta | None,
    func_type: Literal["XW", "XqW", "XWq", "XqWq"],
    backend: Literal["separate", "fused"],
) -> Tensor:
    if "Xq" in func_type:
        assert input_meta is not None
        input_scales, input_elements, input_tensor_meta = extract_mxfp_components(
            input, block_dim=-1, mxfp_meta=input_meta
        )
        input = compose_mxfp_tensor(input_scales, input_elements, input_tensor_meta)
    if "Wq" in func_type:
        assert other_meta is not None
        other_scales, other_elements, other_tensor_meta = extract_mxfp_components(
            other, block_dim=-2, mxfp_meta=other_meta
        )
        other = compose_mxfp_tensor(other_scales, other_elements, other_tensor_meta)

    if backend == "separate":
        out = torch.matmul(input, other)
    else:
        raise NotImplementedError("'fused' not implemented")
    return out
