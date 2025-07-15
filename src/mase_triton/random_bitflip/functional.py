from torch import Tensor

from .core import calculate_flip_probability as _calculate_flip_probability
from .core import find_nearest_prob_n_halves as _find_nearest_prob_n_halves
from .core import random_bitflip_fn as _random_bitflip_fn


def random_bitflip_fn(
    x: Tensor,
    exp_halves: int | None,
    frac_halves: int | None,
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float | None,
) -> tuple[Tensor, int, int]:
    """
    Perform random bit flipping on the input tensor.
    This function applies a random bit flip operation to the input tensor `x`, based on the specified
    number of halves for exponent and fraction bits. The operation is controlled by the provided seeds
    for exponent and fraction bits, allowing for reproducibility in the random bit flipping process.

    Args:
        x (Tensor): Input tensor to be processed.
        exp_halves (int | None): Number of halves for exponent bits. If None, no exponent bit flipping is applied.
        frac_halves (int | None): Number of halves for fraction bits. If None, no fraction bit flipping is applied.
        seed_exp (int): Seed for the random number generator for exponent bits.
        seed_frac (int): Seed for the random number generator for fraction bits.
        zero_out_threshold (float | None): Threshold below which elements are set to zero.
    Returns:
        tuple[Tensor, int, int]: A tuple containing:
            - The processed tensor after random bit flipping.
            - Updated seed for exponent bits.
            - Updated seed for fraction bits.
    """
    return _random_bitflip_fn(
        x=x,
        exp_halves=exp_halves,
        frac_halves=frac_halves,
        seed_exp=seed_exp,
        seed_frac=seed_frac,
        zero_out_threshold=zero_out_threshold,
    )


def calculate_flip_probability(prob_halves: int | None) -> float | None:
    """Calculate the flip probability from the number of halves, prob = 0.5^prob_halves.
    Note that current flip kernel uses bitwise-or only (refer to _cta_random_flip).

    Args:
        prob_halves (int | None): The number of halves to calculate the flip probability.

    Returns:
        float | None: The calculated flip probability, or None if prob_halves is None.
    """
    return _calculate_flip_probability(prob_halves=prob_halves)


def find_nearest_prob_n_halves(prob: float | None) -> int | None:
    """
    Calculate the smallest integer n such that (1/2)^n is less than or equal to the given probability.

    This function computes the smallest number of halvings (n) required for the probability to be less than or equal to the given probability.

    Args:
        prob (float): The probability value for which to find the nearest number of halvings.

    Returns:
        int: The smallest integer n such that (1/2)^n <= prob.
    """
    return _find_nearest_prob_n_halves(prob=prob)
