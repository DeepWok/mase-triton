from dataclasses import dataclass

@dataclass
class minifloatTensorMeta:
    device: str
    dtype: str
    shape: tuple[int, ...]
    block_dim: int
