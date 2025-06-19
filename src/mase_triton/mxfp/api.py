import torch
from torch import Tensor

from . import core as mxfp_core
from . import fake as mxfp_fake
from .helpers import flatten_for_quantize, permute_for_dequantize
from .meta import MXFPMeta, MXFPTensorMeta


def extract_mxfp_components(
    tensor: Tensor, block_dim: int, mxfp_meta: MXFPMeta
) -> tuple[Tensor, Tensor, MXFPTensorMeta]:
    device = str(tensor.device)
    ori_shape = tuple(tensor.shape)
    ndim = len(ori_shape)
    assert block_dim < ndim and block_dim >= -ndim

    assert device.startswith("cpu") or device.startswith("cuda"), (
        f"Unsupported device: {device}. Only 'cpu' and 'cuda' are supported."
    )
    tensor = tensor.to(torch.bfloat16)
    tensor = flatten_for_quantize(tensor, block_dim)
    if device == "cpu":
        scales, elements = mxfp_fake.extract_mxfp_components(
            tensor, mxfp_meta=mxfp_meta
        )
    else:
        scales, elements = mxfp_core.extract_mxfp_components(
            tensor, mxfp_meta=mxfp_meta
        )
    tensor_meta = MXFPTensorMeta(
        device=device,
        shape=ori_shape,
        block_dim=block_dim,
        meta=mxfp_meta,
    )
    return scales, elements, tensor_meta


def compose_mxfp_tensor(
    scales,
    elements,
    tensor_meta: MXFPTensorMeta,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    device = tensor_meta.device

    if device == "cpu":
        tensor = mxfp_fake.compose_mxfp_tensor(
            shared_scales=scales,
            elements=elements,
            mxfp_meta=tensor_meta.meta,
        )
    else:
        tensor = mxfp_core.compose_mxfp_tensor(
            shared_scales=scales,
            elements=elements,
            mxfp_meta=tensor_meta.meta,
        )
    tensor = permute_for_dequantize(
        tensor, ori_shape=tensor_meta.shape, block_dim=tensor_meta.block_dim
    )
    tensor = tensor.to(dtype=dtype)
    return tensor
