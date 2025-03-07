import torch
from torch import Tensor

from .core import find_nearest_prob_n_halves, random_bitflip_fn, calculate_flip_probability


def _format_prob(prob: float | None) -> str:
    return "None" if prob is None else f"{prob:.2e}"


class RandomBitFlipDropout(torch.nn.Module):
    """Random bit flip layer, which flips the sign-exponent and fraction bits with given probabilities.
    If zero_out_threshold is not None, the flipped element whose absolute value is less than this threshold are zeroed out,
    the gradient of these zeroed out elements are also zeroed out.

    Parameters
    ----------
    p_exp : float | None
        the random bit flip probability for sign-exponent bits = 0.5^find_nearest_prob_n_halves(p_exp)
    p_frac : float | None
        the random bit flip probability for fraction bits = 0.5^find_nearest_prob_n_halves(p_frac)
    zero_out_threshold : float | None
        if not None, zero out the bits whose absolute value is less than this threshold (including NaN).
        if None, no zero out operation is performed.
    seed_exp : int
        the initial random seed for sign-exp bits. Note the same seed generates the same random bits,
        thus the seed is updated after each call.
    seed_frac : int
        the random seed for sign-exp bits. Note the same seed generates the same random bits,
        thus the seed is updated after each call.
    """

    def __init__(
        self,
        p_exp: float | None,
        p_frac: float | None,
        zero_out_t: float | None,
        seed_exp: int = 0,
        seed_frac: int = 0,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        self.p_exp = p_exp
        self.p_frac = p_frac
        self.nearest_exp_halves = find_nearest_prob_n_halves(p_exp)
        self.nearest_frac_halves = find_nearest_prob_n_halves(p_frac)
        self.seed_exp = seed_exp
        self.seed_frac = seed_frac
        self.zero_out_t = zero_out_t

    def forward(self, x: Tensor) -> Tensor:
        if self.p_exp is None and self.p_frac is None and self.zero_out_t is None:
            return x
        else:
            out, seed_exp, seed_frac = random_bitflip_fn(
                x,
                exp_halves=self.nearest_exp_halves,
                frac_halves=self.nearest_frac_halves,
                seed_exp=self.seed_exp,
                seed_frac=self.seed_frac,
                zero_out_threshold=self.zero_out_t,
            )
            self.seed_exp = seed_exp
            self.seed_frac = seed_frac
            return out

    def extra_repr(self) -> str:
        return (
            f"nearest_p_exp={_format_prob(calculate_flip_probability(self.nearest_exp_halves))}, "
            f"nearest_p_frac={_format_prob(calculate_flip_probability(self.nearest_frac_halves))}, "
            f"zero_out_threshold={self.zero_out_t}, "
            f"seed_exp={self.seed_exp}, seed_frac={self.seed_frac}"
        )


class RandomBitFlipLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device,
        dtype,
        x_p_exp: float | None,
        x_p_frac: float | None,
        x_zero_out_t: float | None,
        w_p_exp: float | None,
        w_p_frac: float | None,
        w_zero_out_t: float | None,
        x_seed_exp: int = 0,
        x_seed_frac: int = 0,
        w_seed_exp: int = 0,
        w_seed_frac: int = 0,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.x_p_exp = x_p_exp
        self.x_p_frac = x_p_frac
        self.x_nearest_exp_halves = find_nearest_prob_n_halves(x_p_exp)
        self.x_nearest_frac_halves = find_nearest_prob_n_halves(x_p_frac)
        self.x_seed_exp = x_seed_exp
        self.x_seed_frac = x_seed_frac
        self.x_zero_out_t = x_zero_out_t

        self.w_p_exp = w_p_exp
        self.w_p_frac = w_p_frac
        self.w_nearest_exp_halves = find_nearest_prob_n_halves(w_p_exp)
        self.w_nearest_frac_halves = find_nearest_prob_n_halves(w_p_frac)
        self.w_seed_exp = w_seed_exp
        self.w_seed_frac = w_seed_frac
        self.w_zero_out_t = w_zero_out_t

    def forward(self, x: Tensor) -> Tensor:
        if not (self.x_p_exp is None and self.x_p_frac is None and self.x_zero_out_t is None):
            x, x_seed_exp, x_seed_frac = random_bitflip_fn(
                x,
                exp_halves=self.x_nearest_exp_halves,
                frac_halves=self.x_nearest_frac_halves,
                seed_exp=self.x_seed_exp,
                seed_frac=self.x_seed_frac,
                zero_out_threshold=self.x_zero_out_t,
            )
            self.x_seed_exp = x_seed_exp
            self.x_seed_frac = x_seed_frac

        if self.w_p_exp is None and self.w_p_frac is None and self.w_zero_out_t is None:
            w = self.weight
        else:
            w, w_seed_exp, w_seed_frac = random_bitflip_fn(
                self.weight,
                exp_halves=self.w_nearest_exp_halves,
                frac_halves=self.w_nearest_frac_halves,
                seed_exp=self.w_seed_exp,
                seed_frac=self.w_seed_frac,
                zero_out_threshold=self.w_zero_out_t,
            )
            self.w_seed_exp = w_seed_exp
            self.w_seed_frac = w_seed_frac
        out = torch.nn.functional.linear(x, w, self.bias)
        return out

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        x_p_exp: float | None,
        x_p_frac: float | None,
        x_zero_out_t: float | None,
        w_p_exp: float | None,
        w_p_frac: float | None,
        w_zero_out_t: float | None,
        x_seed_exp: int = 0,
        x_seed_frac: int = 0,
        w_seed_exp: int = 0,
        w_seed_frac: int = 0,
    ) -> "RandomBitFlipLinear":
        new_fc = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            linear.weight.dtype,
            x_p_exp=x_p_exp,
            x_p_frac=x_p_frac,
            x_zero_out_t=x_zero_out_t,
            w_p_exp=w_p_exp,
            w_p_frac=w_p_frac,
            w_zero_out_t=w_zero_out_t,
            x_seed_exp=x_seed_exp,
            x_seed_frac=x_seed_frac,
            w_seed_exp=w_seed_exp,
            w_seed_frac=w_seed_frac,
        )
        with torch.no_grad():
            new_fc.weight.copy_(linear.weight)
            if linear.bias is not None:
                new_fc.bias.copy_(linear.bias)
        return new_fc

    def extra_repr(self) -> str:
        text = (
            f"x_nearest_p_exp={_format_prob(calculate_flip_probability(self.x_nearest_exp_halves))}, "
            f"x_nearest_p_frac={_format_prob(calculate_flip_probability(self.x_nearest_frac_halves))}, "
            f"x_zero_out_threshold={self.x_zero_out_t}, "
            f"x_seed_exp={self.x_seed_exp}, x_seed_frac={self.x_seed_frac}, "
            f"w_nearest_p_exp={_format_prob(calculate_flip_probability(self.w_nearest_exp_halves))}, "
            f"w_nearest_p_frac={_format_prob(calculate_flip_probability(self.w_nearest_frac_halves))}, "
            f"w_zero_out_threshold={self.w_zero_out_t}, "
            f"w_seed_exp={self.w_seed_exp}, w_seed_frac={self.w_seed_frac}"
        )
        return text
