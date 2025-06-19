import torch
from torch import Tensor

from . import core as mxfp_core
from . import fake as mxfp_fake
from .helpers import flatten_for_quantize, permute_for_dequantize
from .meta import MXFPMeta, MXFPTensorMeta


def extract_mxfp_components(
    tensor: Tensor, block_dim: int, mxfp_meta: MXFPMeta
) -> tuple[Tensor, Tensor, MXFPTensorMeta]:
    """
    Extracts the MXFP components from a tensor.

    .. note::
        The block for exponent sharing is a 1D vector instead of a 2D matrix.

    :param tensor: The input tensor to be quantized.
    :type tensor: torch.Tensor
    :param block_dim: The dimension to group the tensor elements into blocks.
    :type block_dim: int
    :param mxfp_meta: The metadata for the MXFP format.
    :type mxfp_meta: MXFPMeta

    :returns: The extracted scales, elements, and tensor metadata.
    :rtype: tuple[torch.Tensor, torch.Tensor, MXFPTensorMeta]
    """
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
    """
    Compose a tensor from MXFP components.

    :param scales: The shared scales for exponent sharing.
    :type scales: torch.Tensor
    :param elements: The elements of the tensor.
    :type elements: torch.Tensor
    :param tensor_meta: The metadata for the MXFP tensor.
    :type tensor_meta: MXFPTensorMeta
    :param dtype: The desired data type of the output tensor, by default torch.bfloat16.
    :type dtype: torch.dtype, optional

    :returns: The dequantized tensor.
    :rtype: torch.Tensor
    """
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
