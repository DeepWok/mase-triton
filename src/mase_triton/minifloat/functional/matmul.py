from typing import Literal

import torch
from torch import Tensor

from .cast import compose_minifloat_tensor, extract_minifloat_components


def minifloat_matmul(
    input: Tensor,
    other: Tensor,
    func_type: Literal["XW", "XqW", "XWq", "XqWq"],
    backend: Literal["separate", "fused"],
) -> Tensor:
    if "Xq" in func_type:
        input_elements, input_tensor_meta = extract_minifloat_components(input, block_dim=-1)
        input = compose_minifloat_tensor(input_elements, input_tensor_meta)
    if "Wq" in func_type:
        other_elements, other_tensor_meta = extract_minifloat_components(other, block_dim=-2)
        other = compose_minifloat_tensor(other_elements, other_tensor_meta)

    if backend == "separate":
        out = torch.matmul(input, other)
    else:
        raise NotImplementedError("'fused' not implemented")
    return out
