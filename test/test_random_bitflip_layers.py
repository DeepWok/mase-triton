import torch

from mase_triton.logging import set_logging_verbosity, test_logger
from mase_triton.random_bitflip.layers import (
    RandomBitFlipDropout,
    RandomBitFlipLinear,
)
from mase_triton.random_bitflip.utils import calculate_bit_mismatch_rate

logger = test_logger.getChild(f"{__name__}")

DEVICE = "cuda"


def test_random_bitflip_dropout():
    n_passes = 4
    p_exp = 0.5**4
    p_frac = 0.5**5
    zero_out_threshold = 2
    seed_exp, seed_frac = 1, 1
    bitflip = RandomBitFlipDropout(
        p_exp=p_exp,
        p_frac=p_frac,
        zero_out_t=zero_out_threshold,
        seed_exp=seed_exp,
        seed_frac=seed_frac,
    )
    logger.info(bitflip)

    for i in range(n_passes):
        x = torch.randn(8, device=DEVICE, dtype=torch.float32)
        x = x + 0.1
        x.requires_grad_()

        out = bitflip(x)
        loss = torch.sum(out)
        loss.backward()

        assert torch.all(torch.isfinite(x))
        assert torch.all((out != 0) == (x.grad == 1.0))
        logger.info(f"{i}-th pass, {bitflip}")


def test_random_bitflip_linear_act_only():
    dtype = torch.float16
    n_tries = 4
    bs = 8
    in_features = 1024
    out_features = 1024
    x_p_exp = 0.5**3
    x_p_frac = 0.5**4
    x_zero_out_t = 1e3
    w_p_exp = None
    w_p_frac = None
    w_zero_out_t = None

    logger.info(
        f"x_p_exp={x_p_exp}, x_p_frac={x_p_frac}, x_zero_out_t={x_zero_out_t}, bypassing w"
    )

    fc = torch.nn.Linear(
        in_features, out_features, bias=False, device=DEVICE, dtype=dtype
    )
    with torch.no_grad():
        fc.weight.copy_(
            torch.eye(in_features, device=DEVICE, dtype=dtype).transpose(0, 1)
        )

    bitflip_fc = RandomBitFlipLinear.from_linear(
        fc,
        x_p_exp=x_p_exp,
        x_p_frac=x_p_frac,
        x_zero_out_t=x_zero_out_t,
        w_p_exp=w_p_exp,
        w_p_frac=w_p_frac,
        w_zero_out_t=w_zero_out_t,
    )
    logger.info(str(bitflip_fc))
    bitflip_fc.train()

    for i in range(n_tries):
        x = torch.randn(bs, in_features, device=DEVICE, dtype=dtype)
        x.requires_grad_()
        out = bitflip_fc(x)
        assert torch.all(torch.isfinite(x))
        assert x.shape == out.shape
        find_bitflip = not torch.equal(x, out)
        loss = torch.sum(out)
        loss.backward()
        x.grad
        bitflip_fc.weight.grad
        if find_bitflip:
            mismatch_rates = calculate_bit_mismatch_rate(x, out)
            logger.info(f"{i}-th try, mismatch_rates: {mismatch_rates}")
            break


if __name__ == "__main__":
    from mase_triton.utils.debug import set_ipdb_breakpoint

    set_ipdb_breakpoint()
    set_logging_verbosity("info")
    test_random_bitflip_dropout()
    test_random_bitflip_linear_act_only()
