import functools
from dataclasses import dataclass
from typing import Literal

from ..minifloat.meta import MinifloatMeta, MinifloatTensorMeta


@dataclass(frozen=True)
class MXFPMeta:
    block_size: int
    scale_exp_bits: int
    element_exp_bits: int
    element_frac_bits: int
    element_is_finite: bool = True
    tag: str = ""

    def __post_init__(self):
        # check scale exponent bits
        legal_scale_exp_bits = (8,)
        assert self.scale_exp_bits in legal_scale_exp_bits, (
            f"Invalid exponent bits: {self.scale_exp_bits}. "
            f"Legal values are: {legal_scale_exp_bits}."
        )
        # check element exponent and fraction bits
        legal_element_exp_frac_bits = ((4, 3), (5, 2), (2, 3), (3, 2), (2, 1))
        el_exp_frac = (self.element_exp_bits, self.element_frac_bits)
        assert el_exp_frac in legal_element_exp_frac_bits, (
            f"Invalid element exponent and fraction bits: {self.element_exp_bits}, {self.element_frac_bits}. "
            f"Legal values are: {legal_element_exp_frac_bits}."
        )

    @functools.cached_property
    def element_meta(self) -> MinifloatMeta:
        """Returns the metadata for the element (Minifloat) part of the MXFP format."""
        return MinifloatMeta(
            exp_bits=self.element_exp_bits,
            frac_bits=self.element_frac_bits,
            is_finite=self.element_is_finite,
        )


@dataclass
class MXFPTensorMeta:
    device: str
    dtype: str
    shape: tuple[int, ...]
    block_dim: int
    meta: MXFPMeta


MXFP8_E4M3_fn = MXFPMeta(
    block_size=32,
    scale_exp_bits=8,
    element_exp_bits=4,
    element_frac_bits=3,
    element_is_finite=True,
    tag="MXFP8_E4M3_fn",
)
MXFP8_E5M2_fn = MXFPMeta(
    block_size=32,
    scale_exp_bits=8,
    element_exp_bits=5,
    element_frac_bits=2,
    element_is_finite=True,
    tag="MXFP8_E5M2_fn",
)
OCP_MXFP6_E2M3 = MXFPMeta(
    block_size=32,
    scale_exp_bits=8,
    element_exp_bits=2,
    element_frac_bits=3,
    element_is_finite=True,
    tag="OCP_MXFP6_E2M3",
)
OCP_MXFP6_E3M2 = MXFPMeta(
    block_size=32,
    scale_exp_bits=8,
    element_exp_bits=3,
    element_frac_bits=2,
    element_is_finite=True,
    tag="OCP_MXFP6_E3M2",
)
OCP_MXFP4_E2M1 = MXFPMeta(
    block_size=32,
    scale_exp_bits=8,
    element_exp_bits=2,
    element_frac_bits=1,
    element_is_finite=True,
    tag="OCP_MXFP4_E2M1",
)
