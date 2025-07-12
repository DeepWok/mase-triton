from dataclasses import dataclass


@dataclass
class ScaledIntMeta:
    block_size: int
    bits: int

