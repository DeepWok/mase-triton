from typing import Literal

import torch
from torch import Tensor

from .. import fake as minifloat_fake
from .. import kernels as minifloat_kernels
from ..helpers import flatten_for_quantize, permute_for_dequantize
from ..meta import minifloatTensorMeta


def extract_minifloat_components(tensor: Tensor, block_dim: int) -> tuple[Tensor, minifloatTensorMeta]:
    device = str(tensor.device)
    ori_shape = tuple(tensor.shape)
    ori_dtype = str(tensor.dtype).removeprefix("torch.")
    ndim = len(ori_shape)
    assert block_dim < ndim and block_dim >= -ndim

    assert device.startswith("cpu") or device.startswith("cuda"), (
        f"Unsupported device: {device}. Only 'cpu' and 'cuda' are supported."
    )
    tensor = tensor.to(torch.bfloat16)
    tensor = flatten_for_quantize(tensor, block_dim)
    if device == "cpu":
        elements = minifloat_fake.extract_minifloat_components(tensor)
    else:
        elements = minifloat_kernels.extract_minifloat_components(tensor)
    tensor_meta = minifloatTensorMeta(
        device=device,
        dtype=ori_dtype,
        shape=ori_shape,
        block_dim=block_dim,
    )
    return elements, tensor_meta


def compose_minifloat_tensor(elements,tensor_meta: minifloatTensorMeta, dtype: torch.dtype | None = None) -> Tensor:
    device = tensor_meta.device
    dtype = getattr(torch, tensor_meta.dtype) if dtype is None else dtype

    if device == "cpu":
        tensor = minifloat_fake.compose_minifloat_tensor(elements)
    else:
        tensor = minifloat_kernels.compose_minifloat_tensor(elements)
    tensor = permute_for_dequantize(
        tensor, ori_shape=tensor_meta.shape, block_dim=tensor_meta.block_dim
    )
    tensor = tensor.to(dtype=dtype)
    return tensor


def quantize_dequantize(tensor: Tensor, block_dim: int, dtype: torch.dtype | None = None) -> Tensor:
    elements, tensor_meta = extract_minifloat_components(tensor, block_dim)
    tensor_dq = compose_minifloat_tensor(elements, tensor_meta, dtype=dtype)
    return tensor_dq
