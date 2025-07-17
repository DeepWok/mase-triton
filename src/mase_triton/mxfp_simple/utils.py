import torch


class ChangeDtypeError(Exception):
    def __init__(self, message="Changing dtype is not allowed."):
        self.message = message
        super().__init__(self.message)


def devices_equal(device1, device2):
    """Check if two devices are equivalent"""
    d1 = torch.device(device1)
    d2 = torch.device(device2)

    # Normalize cuda to cuda:0
    if d1.type == "cuda" and d1.index is None:
        d1 = torch.device("cuda", 0)
    if d2.type == "cuda" and d2.index is None:
        d2 = torch.device("cuda", 0)

    return d1 == d2
