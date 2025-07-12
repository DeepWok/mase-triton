import torch
import triton
from torch import Tensor
from triton import language as tl
from ..meta import ScaledIntMeta


@triton.jit
def _extract_scaled_int_components_kernel(
    x_ptr,
    output_ptr,
    n_elements: int,
    int_bits: tl.constexpr,
    BLK: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # get x value
    x_offs = pid * BLK + tl.arange(0, BLK)
    x_ptr = x_ptr + x_offs
    x = tl.load(x_ptr, mask=x_offs < n_elements, other=0)

    # compute the maximum value with the representation
    int_max = int(2 ** int_bits - 1) 
    # every row of x is scaled by a learnable scale factor, initized to the maximum of the row
    row_max = tl.max(x, axis=0, keep_dims=True)
    scale = row_max / int_max
    output = x / scale
    # store the result
    el_ptrs = output_ptr + x_offs
    tl.store(el_ptrs, output, mask=x_offs < n_elements)


def extract_int_components(
    x: Tensor,
    scaled_int_meta: ScaledIntMeta,
):
    assert x.dtype == torch.bfloat16
    assert x.ndim == 1
    x = x.contiguous()
    n_elements = x.numel()
    B = scaled_int_meta.block_size
    assert n_elements % B == 0
    n_groups = n_elements // B
    device = x.device
    scales = torch.empty((n_groups, 1), dtype=torch.uint8, device=device)
    elements = torch.empty((n_groups, B), dtype=torch.uint8, device=device)

    output = torch.empty(n_elements, dtype=torch.bfloat16, device=device)


    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLK"]),)

    _extract_scaled_int_components_kernel[grid](
        x,
        output,
        n_elements=n_elements,
        int_bits=scaled_int_meta.bits,
        BLK=128,
    )

    return scales, elements