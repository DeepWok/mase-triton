import torch

from mase_triton.mxfp.api import compose_mxfp_tensor, extract_mxfp_components
from mase_triton.mxfp.meta import (
    OCP_MXFP6_E3M2,
    OCP_MXFP8_E4M3,
    OCP_MXFP8_E5M2,
    MXFPElementMeta,
    MXFPMeta,
    MXFPScaleMeta,
)
from mase_triton.utils.bit_repr import get_binary_repr, get_binary_repr_bf16


def minimal():
    torch.set_printoptions(linewidth=200)
    # create a mxfp format
    mxfp_format = MXFPMeta(
        block_size=2,
        scale=MXFPScaleMeta(exponent_bits=8),
        element=MXFPElementMeta(exponent_bits=4, mantissa_bits=3),
    )

    w = torch.randn((3, 2), dtype=torch.bfloat16, device="cuda") * 100.0
    print("Original tensor:")
    print(get_binary_repr_bf16(w))
    scales, elements, tensor_meta = extract_mxfp_components(
        w, block_dim=1, mxfp_meta=mxfp_format
    )
    print("Tensor Meta data")
    print(tensor_meta)
    print("Shared scales:")
    print(get_binary_repr(scales))
    print("Elements (Minifloat)")
    print(get_binary_repr(elements))
    """outputs:
    Original tensor:
    [['0 10000010 1110010' '0 10000000 1101111']
     ['0 10000101 1100111' '0 10000101 0000010']
     ['1 10000011 1111010' '1 10000110 0111010']]
    Tensor Meta data
    MXFPTensorMeta(device='cuda:0', shape=(3, 2), block_dim=1, meta=MXFPMeta(block_size=2, scale=Scale(exp_bits=8), element=Element(exp_bits=4, frac_bits=3)))
    Shared scales:
    [['1000 0010']
     ['1000 0101']
     ['1000 0110']]
    Elements (Minifloat)
    [['0011 1111' '0010 1110']
     ['0011 1110' '0011 1000']
     ['1010 0111' '1011 1011']]
    """


def mxfp8_example():
    torch.set_printoptions(linewidth=200)
    # create a mxfp format
    mxfp_format = OCP_MXFP8_E5M2

    w = torch.randn((3, 64), dtype=torch.bfloat16, device="cuda") * 100.0
    scales, elements, tensor_meta = extract_mxfp_components(
        w, block_dim=1, mxfp_meta=mxfp_format
    )
    w_mxfp8 = compose_mxfp_tensor(scales, elements, tensor_meta)
    avg_err = (w - w_mxfp8).abs().mean()
    err_ratio = avg_err / w.abs().mean()
    print(f"Tensor Meta data")
    print(mxfp_format)
    print("Mean abs error: ", avg_err.item())
    print("Error ratio: ", err_ratio.item())
    """outputs:
    Tensor Meta data
    MXFPMeta(block_size=32, scale=Scale(exp_bits=8), element=Element(exp_bits=5, frac_bits=2))
    Mean abs error:  6.53125
    Error ratio:  0.08251953125
    """


if __name__ == "__main__":
    minimal()
    mxfp8_example()
