from typing import Literal

import torch
from torch import Tensor

from .. import fake as mxfp_fake
from .. import kernels as mxfp_kernels
from ..helpers import flatten_for_quantize, permute_for_dequantize
from ..meta import MXFPMeta, MXFPTensorMeta


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
    ori_dtype = str(tensor.dtype).removeprefix("torch.")
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
        scales, elements = mxfp_kernels.extract_mxfp_components(
            tensor, mxfp_meta=mxfp_meta
        )
    tensor_meta = MXFPTensorMeta(
        device=device,
        dtype=ori_dtype,
        shape=ori_shape,
        block_dim=block_dim,
        meta=mxfp_meta,
    )
    return scales, elements, tensor_meta


def compose_mxfp_tensor(
    scales,
    elements,
    tensor_meta: MXFPTensorMeta,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Compose a tensor from MXFP components.

    :param scales: The shared scales for exponent sharing.
    :type scales: torch.Tensor
    :param elements: The elements of the tensor.
    :type elements: torch.Tensor
    :param tensor_meta: The metadata for the MXFP tensor.
    :type tensor_meta: MXFPTensorMeta
    :param dtype: The desired data type of the output tensor, by default None, which uses the dtype from tensor_meta.
    :type dtype: torch.dtype, optional

    :returns: The dequantized tensor.
    :rtype: torch.Tensor
    """
    device = tensor_meta.device
    dtype = getattr(torch, tensor_meta.dtype) if dtype is None else dtype

    if device == "cpu":
        tensor = mxfp_fake.compose_mxfp_tensor(
            shared_scales=scales,
            elements=elements,
            mxfp_meta=tensor_meta.meta,
        )
    else:
        tensor = mxfp_kernels.compose_mxfp_tensor(
            shared_scales=scales,
            elements=elements,
            mxfp_meta=tensor_meta.meta,
        )
    tensor = permute_for_dequantize(
        tensor, ori_shape=tensor_meta.shape, block_dim=tensor_meta.block_dim
    )
    tensor = tensor.to(dtype=dtype)
    return tensor


def mxfp_linear_XWqB(
    x: Tensor,
    w_scales: Tensor,
    w_elements: Tensor,
    w_tensor_meta: MXFPTensorMeta,
    b: Tensor | None,
    backend: Literal["separate", "fused"],
):
    if backend == "separate":
        w_dq = compose_mxfp_tensor(w_scales, w_elements, w_tensor_meta)
        out = torch.nn.functional.linear(x, w_dq, b)
    else:
        raise NotImplementedError("'fused' not implemented")
    return out


def parse_mxfp_linear_type(
    x: Tensor,
    x_meta: MXFPMeta | None,
    x_scales: Tensor | None,
    x_elements: Tensor | None,
    x_tensor_meta: Tensor | None,
    w: Tensor,
    w_meta: MXFPMeta | None,
    w_scales: Tensor | None,
    w_elements: Tensor | None,
    w_tensor_meta: MXFPTensorMeta | None,
    b: Tensor | None,
    b_meta: MXFPMeta | None,
    b_scales: Tensor | None,
    b_elements: Tensor | None,
    b_tensor_meta: MXFPTensorMeta | None,
) -> str:
    pattern = ""

    check_list = [
        ["X", x, x_meta, x_scales, x_elements, x_tensor_meta],
        ["W", w, w_meta, w_scales, w_elements, w_tensor_meta],
        ["B", b, b_meta, b_scales, b_elements, b_tensor_meta],
    ]
    for target, t, t_meta, t_scales, t_elements, t_tensor_meta in check_list:
        if (
            t_meta is None
            and t_scales is None
            and t_elements is None
            and t_tensor_meta is None
        ):
            pattern += target
            assert t is not None
        else:
            pattern += f"{target}q"
            assert (w_meta is not None) != (
                w_scales is not None
                and w_elements is not None
                and w_tensor_meta is not None
            ), "One and only one setupsmust be avaialbel"
    return pattern


def mxfp_linear(
    x: Tensor,
    x_meta: MXFPMeta | None,
    x_scales: Tensor | None,
    x_elements: Tensor | None,
    x_tensor_meta: Tensor | None,
    w: Tensor,
    w_meta: MXFPMeta | None,
    w_scales: Tensor | None,
    w_elements: Tensor | None,
    w_tensor_meta: MXFPTensorMeta | None,
    b: Tensor | None,
    b_meta: MXFPMeta | None,
    b_scales: Tensor | None,
    b_elements: Tensor | None,
    b_tensor_meta: MXFPTensorMeta | None,
    backend: Literal["separate", "fused"],
) -> Tensor:
    linear_type = parse_mxfp_linear_type(
        x=x,
        x_meta=x_meta,
        x_scales=x_scales,
        x_elements=x_elements,
        x_tensor_meta=x_tensor_meta,
        w=w,
        w_meta=w_meta,
        w_scales=w_scales,
        w_elements=w_elements,
        w_tensor_meta=w_tensor_meta,
        b=b,
        b_meta=b_meta,
        b_scales=b_scales,
        b_elements=b_elements,
        b_tensor_meta=b_tensor_meta,
    )
    match linear_type:
        case "XWqB":
            out = mxfp_linear_XWqB(
                x=x,
                w_scales=w_scales,
                w_elements=w_elements,
                w_tensor_meta=w_tensor_meta,
                b=b,
                backend=backend,
            )
        case _:
            raise NotImplementedError

    return out
