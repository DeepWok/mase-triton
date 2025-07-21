import os

import pytest
import tabulate
import torch
import tqdm

from mase_triton.logging import set_logging_verbosity, test_logger
from mase_triton.random_bitflip import functional as RBFunctions
from mase_triton.random_bitflip.utils import calculate_bit_mismatch_rate

set_logging_verbosity("INFO")
logger = test_logger.getChild(f"{__name__}")

DEVICE = "cuda"


@torch.no_grad()
def test_random_bitflip_forward_simple():
    n_tries = 10
    x = torch.zeros(16, device=DEVICE, dtype=torch.bfloat16)
    exp_halves = 4
    frac_halves = 1
    seed_exp, seed_frac = 0, 0
    found_bitflip = False
    for _ in range(n_tries):
        out, seed_exp, seed_frac = RBFunctions.random_bitflip_fn(
            x,
            exp_halves=exp_halves,
            frac_halves=frac_halves,
            seed_exp=seed_exp,
            seed_frac=seed_frac,
            zero_out_threshold=None,
        )
        assert out.dtype == x.dtype
        assert out.shape == x.shape
        if not (x == out).all():
            found_bitflip = True
            break
    assert found_bitflip, "No bitflip found in the output tensor"


@pytest.mark.slow
def test_random_bitflip_forward_fully_activated_slow():
    helper_random_bitflip_forward_fully_activated(
        input_dtypes=[torch.float16, torch.bfloat16, torch.float32],
        s_exp_halves_frac_halves=[(0.5**n, 0.5**n) for n in range(1, 25)],
        M=2048,
        max_tries=1000,
        num_workers=min(16, os.cpu_count() // 2),
    )


def test_random_bitflip_forward_fully_activated():
    helper_random_bitflip_forward_fully_activated(
        input_dtypes=[torch.float16],
        s_exp_halves_frac_halves=[(0.5**n, 0.5**n) for n in range(4, 8)],
        M=512,
        max_tries=1000,
        num_workers=min(16, os.cpu_count() // 2),
    )


def helper_random_bitflip_forward_fully_activated(
    input_dtypes: tuple[torch.dtype],
    s_exp_halves_frac_halves: tuple[tuple[float, float]],
    M: int = 2048,
    max_tries: int = 1000,
    num_workers: int = 4,
):
    dtype2exp_bits = {
        torch.float32: 9,
        torch.float16: 6,
        torch.bfloat16: 9,
    }
    dtype2frac_bits = {
        torch.float32: 23,
        torch.float16: 10,
        torch.bfloat16: 7,
    }
    num_workers = 16
    rows = []
    headers = [
        "input_dtype",
        "exp_n_halves",
        "exp_p",
        "exp_p*exp_bits",
        "exp_ratio",
        "frac_n_halves",
        "frac_p",
        "frac_p*frac_bits",
        "frac_ratio",
    ]
    for input_dtype in input_dtypes:
        x = torch.randn(M, M, device=DEVICE, dtype=input_dtype)
        cur_try = 0
        for exp_p, frac_p in tqdm.tqdm(s_exp_halves_frac_halves):
            exp_halves = RBFunctions.find_nearest_prob_n_halves(exp_p)
            frac_halves = RBFunctions.find_nearest_prob_n_halves(frac_p)
            seed_exp, seed_frac = 42, 42
            while True:
                with torch.no_grad():
                    out, seed_exp, seed_frac = RBFunctions.random_bitflip_fn(
                        x,
                        exp_halves=exp_halves,
                        frac_halves=frac_halves,
                        seed_exp=seed_exp,
                        seed_frac=seed_frac,
                        zero_out_threshold=None,
                    )
                assert out.dtype == input_dtype
                assert out.shape == x.shape
                find_bitflip = not torch.equal(x, out)
                if find_bitflip:
                    mismatch_rate = calculate_bit_mismatch_rate(
                        x, out, num_workers=num_workers
                    )
                    rows.append(
                        [
                            input_dtype,
                            exp_halves,
                            exp_p,
                            round(exp_p * dtype2exp_bits[input_dtype] * x.numel()),
                            mismatch_rate["sign_exp"],
                            frac_halves,
                            frac_p,
                            round(frac_p * dtype2frac_bits[input_dtype] * x.numel()),
                            mismatch_rate["frac"],
                        ]
                    )
                    break
                cur_try += 1
                if cur_try >= max_tries:
                    logger.error(f"Could not find a bitflip in {max_tries} tries")
                    break

    logger.info("\n" + tabulate.tabulate(rows, headers=headers, tablefmt="pretty"))


@torch.no_grad()
def test_random_bitflip_forward_zero_outed():
    for exp_halves in [1, 2, 3, 4]:
        x = torch.randn(2048, 2048, device=DEVICE, dtype=torch.float32)
        frac_halves = 2
        seed_exp, seed_frac = 0, 0
        zero_out_threshold = 200.0
        out, seed_exp, seed_frac = RBFunctions.random_bitflip_fn(
            x,
            exp_halves=exp_halves,
            frac_halves=frac_halves,
            seed_exp=seed_exp,
            seed_frac=seed_frac,
            zero_out_threshold=zero_out_threshold,
        )
        assert torch.all(torch.isfinite(x))
        zero_out_ratio = (out == 0.0).sum() / out.numel()
        assert zero_out_ratio > 0


def test_random_bitflip_fn_backward():
    n_repeats = 4
    for exp_halves in [1, 2, 3]:
        for _ in range(n_repeats):
            x = torch.rand(8, device=DEVICE, dtype=torch.float32)
            x = x + 0.1
            x.requires_grad_()
            frac_halves = 2
            seed_exp, seed_frac = 0, 0
            zero_out_threshold = 200.0
            out, seed_exp, seed_frac = RBFunctions.random_bitflip_fn(
                x,
                exp_halves=exp_halves,
                frac_halves=frac_halves,
                seed_exp=seed_exp,
                seed_frac=seed_frac,
                zero_out_threshold=zero_out_threshold,
            )

            loss = torch.sum(out)
            loss.backward()
            assert torch.all(torch.isfinite(x))
            assert torch.all((out != 0) == (x.grad == 1.0))


if __name__ == "__main__":
    set_logging_verbosity("info")
    torch.set_printoptions(linewidth=120)
    test_random_bitflip_forward_simple()
    test_random_bitflip_forward_fully_activated()
    test_random_bitflip_forward_zero_outed()
    test_random_bitflip_fn_backward()
