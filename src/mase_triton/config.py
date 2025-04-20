import importlib


class AutotuneConfig:
    bitflip: bool = False  # Random Bitflip autotuning
    onn_ot: bool = False  # ONN Optical Transformer autotuning


def set_autotune_config(bitflip: bool = False, onn_ot: bool = False) -> None:
    """
    Set the autotuning configuration.

    Args:
        bitflip (bool): Indicates whether autotuning of bit-flip is enabled. Default is False.
        onn_ot (bool): Indicates whether autotuning of ONN optical transformer is enabled. Default is False.
    """
    AutotuneConfig.bitflip = bitflip
    AutotuneConfig.onn_ot = onn_ot

    from . import optical_compute
    from . import random_bitflip

    if bitflip:
        importlib.reload(random_bitflip)
    if onn_ot:
        importlib.reload(optical_compute)
