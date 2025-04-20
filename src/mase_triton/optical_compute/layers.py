import torch
from torch import Tensor

from .core import OpticalTransformerFunctions as OTFunctions


@torch.no_grad()
def optical_transformer_update_qstats(
    x: Tensor, min_max: Tensor, min_max_quantile: Tensor | None, stat_smooth_factor: float
) -> None:
    dtype = x.dtype
    if min_max_quantile is not None:
        min_max_quantile = min_max_quantile.float()
        if min_max.isinf().any():
            min_max = x.flatten().float().quantile(min_max_quantile).to(dtype)
        else:
            w_min_max_new = x.flatten().float().quantile(min_max_quantile).to(dtype)
            min_max = stat_smooth_factor * min_max + (1 - stat_smooth_factor) * w_min_max_new
    else:
        if min_max.isinf().any():
            min_max = torch.tensor([x.min().item(), x.max().item()], device=x.device, dtype=dtype)
        else:
            min_max_new = torch.tensor([x.min().item(), x.max().item()], device=x.device, dtype=dtype)
            min_max = stat_smooth_factor * min_max + (1 - stat_smooth_factor) * min_max_new
    return min_max


class OpticalTransformerLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        q_levels: int = 256,
        q_lut_min: float | None = 0.020040,
        q_quantiles: tuple[float, float] | None = None,
        q_smooth_factor: float = 0.9,
        q_init_seed: int = 0,
        q_bypass: bool = False,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.quant_levels = q_levels
        self.q_lut_min = q_lut_min
        if q_quantiles is None:
            self.q_min_max_quantile = None
        else:
            assert 0 < q_quantiles[0] < q_quantiles[1] < 1
            self.register_buffer("q_min_max_quantile", torch.tensor(q_quantiles, device=device, dtype=torch.float))
        self.register_buffer("x_min_max", torch.tensor([float("inf"), float("-inf")], device=device, dtype=dtype))
        self.register_buffer("w_min_max", torch.tensor([float("inf"), float("-inf")], device=device, dtype=dtype))
        self.register_buffer("out_min_max", torch.tensor([float("inf"), float("-inf")], device=device, dtype=dtype))
        self.register_buffer("seed", torch.tensor(q_init_seed, device=device, dtype=torch.int64))
        self.x_min_max: Tensor
        self.w_min_max: Tensor
        self.out_min_max: Tensor
        self.seed: Tensor
        self.stat_smooth_factor = q_smooth_factor
        self.bypass = q_bypass

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return super().forward(x)

        if self.training:
            with torch.no_grad():
                x_min_max = optical_transformer_update_qstats(
                    x, self.x_min_max, self.q_min_max_quantile, self.stat_smooth_factor
                )
                self.x_min_max.copy_(x_min_max)
                w_min_max = optical_transformer_update_qstats(
                    self.weight, self.w_min_max, self.q_min_max_quantile, self.stat_smooth_factor
                )
                self.w_min_max.copy_(w_min_max)
                if self.out_min_max.isinf().any():
                    o_min_max = optical_transformer_update_qstats(
                        x @ self.weight.T, self.out_min_max, self.q_min_max_quantile, self.stat_smooth_factor
                    )
                    self.out_min_max.copy_(o_min_max)

        out_q, q_seed = OTFunctions.quantized_linear_fn(
            x,
            self.weight,
            self.bias,
            x_min=self.x_min_max[0].item(),
            x_max=self.x_min_max[1].item(),
            w_min=self.w_min_max[0].item(),
            w_max=self.w_min_max[1].item(),
            w_lut_min=self.q_lut_min,
            o_min=self.out_min_max[0].item(),
            o_max=self.out_min_max[1].item(),
            q_levels=self.quant_levels,
            q_seed=self.seed.item(),
            skip_quantize=False,
        )

        with torch.no_grad():
            self.seed.copy_(q_seed)
            if self.training:
                out_min_max = optical_transformer_update_qstats(
                    out_q, self.out_min_max, self.q_min_max_quantile, self.stat_smooth_factor
                )
                self.out_min_max.copy_(out_min_max)

        return out_q

    def extra_repr(self) -> str:
        q_quantiles = self.q_min_max_quantile.tolist() if self.q_min_max_quantile is not None else None
        return (
            f"q_bypass={self.bypass}, q_levels={self.quant_levels}, q_lut_min={self.q_lut_min}, "
            f"q_quantiles={q_quantiles}, x_min_max={self.x_min_max}, "
            f"w_min_max={self.w_min_max}, out_min_max={self.out_min_max}, seed={self.seed}"
        )

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        q_levels: int = 256,
        q_lut_min: float | None = 0.020040,
        q_quantiles: tuple[float, float] = (0.001, 0.999),
        q_smooth_factor: float = 0.9,
        q_init_seed: int = 0,
        q_bypass: bool = False,
    ) -> "OpticalTransformerLinear":
        new_fc = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            q_levels=q_levels,
            q_lut_min=q_lut_min,
            q_quantiles=q_quantiles,
            q_smooth_factor=q_smooth_factor,
            q_init_seed=q_init_seed,
            q_bypass=q_bypass,
        )
        with torch.no_grad():
            if linear.weight.device != torch.device("meta"):
                new_fc.weight.copy_(linear.weight)
                if linear.bias is not None:
                    new_fc.bias.copy_(linear.bias)
        return new_fc
