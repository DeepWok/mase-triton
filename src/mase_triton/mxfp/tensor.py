from dataclasses import dataclass

from .meta import MXFPMeta


@dataclass
class MXFPComponents:
    shape: tuple[int, ...]
    block_dim: int
    scales: int
    elements: int
    meta: MXFPMeta
