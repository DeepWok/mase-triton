import torch
from torch import Tensor

from .. import fake as fp_fake
from ..meta import MinifloatMeta, MinifloatTensorMeta


def extract_minifloat_component(
    tensor: Tensor, minifloat_meta: MinifloatMeta
) -> tuple[Tensor, MinifloatTensorMeta]:
    """
    Extract the minifloat component from a tensor.

    :param tensor: The input tensor to extract minifloat components from.
    :type tensor: torch.Tensor
    :param minifloat_meta: The metadata for the minifloat format.
    :type minifloat_meta: MinifloatMeta
    :returns: The extracted element (uint16 tensor) and tensor metadata.
    :rtype: tuple[torch.Tensor, MinifloatTensorMeta]
    """
    device = str(tensor.device)
    ori_shape = tuple(tensor.shape)
    ori_dtype = str(tensor.dtype).removeprefix("torch.")

    tensor = tensor.to(torch.float32)

    if device.startswith("cuda"):
        # TODO: Implement triton kernel for minifloat extraction
        element = fp_fake.extract_minifloat_component(tensor, minifloat_meta)
    else:
        element = fp_fake.extract_minifloat_component(tensor, minifloat_meta)
    tensor_meta = MinifloatTensorMeta(
        device=device, dtype=ori_dtype, shape=ori_shape, meta=minifloat_meta
    )
    return element, tensor_meta


def compose_minifloat_component(
    element: Tensor, tensor_meta: MinifloatTensorMeta, dtype: torch.dtype | None = None
) -> Tensor:
    """
    Compose a tensor from minifloat components.
    :param element: The element of the minifloat tensor.
    :type element: torch.Tensor
    :param tensor_meta: The metadata for the minifloat tensor.
    :type tensor_meta: MinifloatTensorMeta
    :param dtype: The desired data type of the output tensor, by default None, which
        uses the dtype from tensor_meta.
    :type dtype: torch.dtype, optional
    :returns: The dequantized tensor.
    :rtype: torch.Tensor
    """
    device = tensor_meta.device
    dtype = getattr(torch, tensor_meta.dtype) if dtype is None else dtype

    if device.startswith("cuda"):
        # TODO: Implement triton kernel for minifloat composition
        tensor = fp_fake.compose_minifloat_component(element, tensor_meta.meta)
    else:
        tensor = fp_fake.compose_minifloat_component(element, tensor_meta.meta)
    tensor = tensor.to(dtype)
    return tensor


def quantize_dequantize(
    tensor: Tensor,
    minifloat_meta: MinifloatMeta,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Quantize and dequantize a tensor using minifloat format.

    :param tensor: The input tensor to quantize and dequantize.
    :type tensor: torch.Tensor
    :param minifloat_meta: The metadata for the minifloat format.
    :type minifloat_meta: MinifloatMeta
    :param dtype: The desired data type of the output tensor, by default None, which uses the dtype from tensor_meta.
    :type dtype: torch.dtype, optional
    :returns: The dequantized tensor.
    :rtype: torch.Tensor
    """
    element, tensor_meta = extract_minifloat_component(tensor, minifloat_meta)
    return compose_minifloat_component(element, tensor_meta, dtype=dtype)
