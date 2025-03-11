import os
import torch
import random

from mase_triton.random_bitflip.core import random_bitflip_fn


@torch.no_grad()
def run_random_bitflip_act():
    num_elements = int(os.environ.get("NUM_ELEMENTS", 64 * 2048 * 1024))
    dtype = getattr(torch, os.environ.get("DTYPE", "float16"))
    enable_zero_out = os.environ.get("ENABLE_ZERO_OUT", "false").lower() == "true"
    n_halves = int(os.environ.get("N_HALVES", 10))

    x = torch.randn(num_elements, device="cuda", dtype=dtype)
    out = random_bitflip_fn(
        x,
        exp_halves=n_halves,
        frac_halves=n_halves,
        seed_exp=random.randint(0, 100000),
        seed_frac=random.randint(0, 100000),
        zero_out_threshold=10 if enable_zero_out else None,
    )


@torch.no_grad()
def run_random_bitflip_weight():
    num_elements = int(os.environ.get("NUM_ELEMENTS", 1024 * 1024))
    dtype = getattr(torch, os.environ.get("DTYPE", "float16"))
    enable_zero_out = os.environ.get("ENABLE_ZERO_OUT", "false").lower() == "true"
    n_halves = int(os.environ.get("N_HALVES", 10))

    x = torch.randn(num_elements, device="cuda", dtype=dtype)
    out = random_bitflip_fn(
        x,
        exp_halves=n_halves,
        frac_halves=n_halves,
        seed_exp=random.randint(0, 100000),
        seed_frac=random.randint(0, 100000),
        zero_out_threshold=10 if enable_zero_out else None,
    )


if __name__ == "__main__":
    run_random_bitflip_act()
    # run_random_bitflip_weight()
