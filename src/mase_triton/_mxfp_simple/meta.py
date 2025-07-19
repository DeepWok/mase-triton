from dataclasses import dataclass

import torch

from ..utils.meta import device_str, dtype_str, shape_tuple


@dataclass
class MXFPMeta:
    block_size: int
    scale_exp_bits: int
    element_exp_bits: int
    element_frac_bits: int
    tag: str = ""

    def __post_init__(self):
        legal_scale_exp_bits = (8,)
        assert self.scale_exp_bits in legal_scale_exp_bits, (
            f"Invalid exponent bits: {self.scale_exp_bits}. "
            f"Legal values are: {legal_scale_exp_bits}."
        )

        legal_element_exp_frac_bits = ((4, 3), (5, 2), (2, 3), (3, 2), (2, 1))
        el_exp_frac = (self.element_exp_bits, self.element_frac_bits)
        assert el_exp_frac in legal_element_exp_frac_bits, (
            f"Invalid element exponent and fraction bits: {self.element_exp_bits}, {self.element_frac_bits}. "
            f"Legal values are: {legal_element_exp_frac_bits}."
        )
        self.element_bits = self.element_exp_bits + self.element_frac_bits + 1


@dataclass(frozen=True)
class MXFPTensorMeta:
    device: str
    dtype: str
    shape: tuple[int, ...]
    block_dim: int
    meta: MXFPMeta

    def __post_init__(self):
        super().__setattr__("device", device_str(self.device))
        super().__setattr__("dtype", dtype_str(self.dtype))
        super().__setattr__("shape", shape_tuple(self.shape))

    def create(
        self,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
        shape: tuple[int, ...] | torch.Size | None = None,
        block_dim: int | None = None,
        meta: MXFPMeta | None = None,
    ) -> "MXFPTensorMeta":
        device = self.device if device is None else device_str(device)
        dtype = self.dtype if dtype is None else dtype_str(dtype)
        shape = self.shape if shape is None else shape_tuple(shape)
        block_dim = self.block_dim if block_dim is None else block_dim
        meta = self.meta if meta is None else meta
        return MXFPTensorMeta(
            device=device,
            dtype=dtype,
            shape=shape,
            block_dim=block_dim,
            meta=meta,
        )


OCP_MXFP8_E4M3 = MXFPMeta(
    block_size=32,
    scale_exp_bits=8,
    element_exp_bits=4,
    element_frac_bits=3,
    tag="OCP_MXFP8_E4M3",
)
OCP_MXFP8_E5M2 = MXFPMeta(
    block_size=32,
    scale_exp_bits=8,
    element_exp_bits=5,
    element_frac_bits=2,
    tag="OCP_MXFP8_E5M2",
)
OCP_MXFP6_E2M3 = MXFPMeta(
    block_size=32,
    scale_exp_bits=8,
    element_exp_bits=2,
    element_frac_bits=3,
    tag="OCP_MXFP6_E2M3",
)
OCP_MXFP6_E3M2 = MXFPMeta(
    block_size=32,
    scale_exp_bits=8,
    element_exp_bits=3,
    element_frac_bits=2,
    tag="OCP_MXFP6_E3M2",
)
OCP_MXFP4_E2M1 = MXFPMeta(
    block_size=32,
    scale_exp_bits=8,
    element_exp_bits=2,
    element_frac_bits=1,
    tag="OCP_MXFP4_E2M1",
)
