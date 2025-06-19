import torch

from mase_triton.mxfp.api import compose_mxfp_tensor, extract_mxfp_components
from mase_triton.mxfp.meta import (
    OCP_MXFP4_E2M1,
    OCP_MXFP6_E2M3,
    OCP_MXFP6_E3M2,
    OCP_MXFP8_E4M3,
    OCP_MXFP8_E5M2,
    MXFPMeta,
)
from mase_triton.utils.bit_repr import get_binary_repr, get_binary_repr_bf16


def minimal():
    torch.set_printoptions(linewidth=200)
    # create a mxfp format
    mxfp_format = MXFPMeta(
        block_size=2,
        scale_exp_bits=8,
        element_exp_bits=4,
        element_frac_bits=3,
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
    [['1 10000110 0001011' '0 10000010 1010011']
    ['0 10000100 0101111' '0 10000110 1000101']
    ['1 10000111 0001001' '1 10000100 1011111']]
    Tensor Meta data
    MXFPTensorMeta(device='cuda:0', shape=(3, 2), block_dim=1, meta=MXFPMeta(block_size=2, scale_exp_bits=8, element_exp_bits=4, element_frac_bits=3))
    Shared scales:
    [['1000 0110']
    ['1000 0110']
    ['1000 0111']]
    Elements (Minifloat)
    [['1011 1000' '0001 1101']
    ['0010 1010' '0011 1100']
    ['1011 1000' '1010 0101']]
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
    MXFPMeta(block_size=32, scale_exp_bits=8, element_exp_bits=5, element_frac_bits=2)
    Mean abs error:  6.125
    Error ratio:  0.08740234375
    """


if __name__ == "__main__":
    minimal()
    mxfp8_example()
