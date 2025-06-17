from dataclasses import dataclass


@dataclass
class MXFPScaleMeta:
    exponent_bits: int = 8

    def __post_init__(self):
        legal_exponent_bits = (8,)
        assert self.exponent_bits in legal_exponent_bits, (
            f"Invalid exponent bits: {self.exponent_bits}. "
            f"Legal values are: {legal_exponent_bits}."
        )


@dataclass
class MXFPElementMeta:
    exponent_bits: int
    mantissa_bits: int

    def __post_init__(self):
        legal_exp_mantissa = ((4, 3), (5, 2), (2, 3), (3, 2), (2, 1))
        assert (self.exponent_bits, self.mantissa_bits) in legal_exp_mantissa, (
            f"Invalid exponent and mantissa bits: {self.exponent_bits}, {self.mantissa_bits}. "
            f"Legal values are: {legal_exp_mantissa}."
        )
        self.n_bits = self.exponent_bits + self.mantissa_bits + 1


@dataclass
class MXFPMeta:
    block_size: int
    scale: MXFPScaleMeta
    element: MXFPElementMeta


OCP_MXFP8_E4M3 = MXFPMeta(
    block_size=32,
    scale=MXFPScaleMeta(exponent_bits=8),
    element=MXFPElementMeta(exponent_bits=4, mantissa_bits=3),
)
OCP_MXFP8_E5M2 = MXFPMeta(
    block_size=32,
    scale=MXFPScaleMeta(exponent_bits=8),
    element=MXFPElementMeta(exponent_bits=5, mantissa_bits=2),
)
OCP_MXFP6_E2M3 = MXFPMeta(
    block_size=32,
    scale=MXFPScaleMeta(exponent_bits=8),
    element=MXFPElementMeta(exponent_bits=2, mantissa_bits=3),
)
OCP_MXFP6_E3M2 = MXFPMeta(
    block_size=32,
    scale=MXFPScaleMeta(exponent_bits=8),
    element=MXFPElementMeta(exponent_bits=3, mantissa_bits=2),
)
OCP_MXFP4_E2M1 = MXFPMeta(
    block_size=32,
    scale=MXFPScaleMeta(exponent_bits=8),
    element=MXFPElementMeta(exponent_bits=2, mantissa_bits=1),
)


@dataclass
class MXFPComponents:
    shape: tuple[int, ...]
    scales: int
    elements: int
    meta: MXFPMeta
