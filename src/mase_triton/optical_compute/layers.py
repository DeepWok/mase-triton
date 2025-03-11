import torch
from torch import Tensor

from .core import optical_compute_quantized_forward_fn


class OpticalComputeQLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        device=None,
        dtype=None,
        q_levels: int = 256,
        q_lut_min: float | None = 0.020040,
        q_samples_to_track: int | None = None,
        q_smooth_factor: float = 0.9,
        q_min_quantile: float = 0.001,
        q_max_quantile: float = 0.999,
        q_init_seed: int = 0,
        q_bypass: bool = False,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.quant_levels = q_levels
        self.q_lut_min = q_lut_min
        self.num_samples_to_track = q_samples_to_track
        self.stat_smooth_factor = q_smooth_factor
        self.bypass = q_bypass

        # fmt: off
        self.register_buffer("q_min_max_quantile", torch.tensor([q_min_quantile, q_max_quantile], device=device, dtype=torch.float, requires_grad=False))
        self.register_buffer("x_running_min_max", torch.tensor([float("inf"), float("-inf")], device=device, dtype=dtype, requires_grad=False))
        self.register_buffer("w_min_max", torch.tensor([float("inf"), float("-inf")], device=device, dtype=dtype, requires_grad=False))
        self.register_buffer("out_running_min_max", torch.tensor([float("inf"), float("-inf")], device=device, dtype=dtype, requires_grad=False))
        self.register_buffer("q_num_tracked_samples", torch.tensor(0, device=device, dtype=torch.int64, requires_grad=False))
        # fmt: on
        self.q_min_max_quantile: Tensor
        self.x_running_min_max: Tensor
        self.w_min_max: Tensor
        self.out_running_min_max: Tensor
        self.q_num_tracked_samples: Tensor

        self.seed = q_init_seed

    @torch.no_grad()
    def _update_running_stats(self, x: Tensor | None, w: Tensor | None, out: Tensor | None):
        dtype = self.x_running_min_max.dtype
        if self.training:
            # act
            # fmt: off
            if x is not None:
                if self.num_samples_to_track is None or self.q_num_tracked_samples.item() < self.num_samples_to_track:
                    if torch.isinf(self.x_running_min_max).any():
                        x_running_min_max = x.flatten().float().quantile(self.q_min_max_quantile).to(dtype)
                        self.x_running_min_max = x_running_min_max
                    else:
                        x_running_min_max = x.flatten().float().quantile(self.q_min_max_quantile).to(dtype)
                        self.x_running_min_max = self.stat_smooth_factor * self.x_running_min_max + (1 - self.stat_smooth_factor) * x_running_min_max
                    self.q_num_tracked_samples += x.size(0)
            # fmt: on
            # weight
            if w is not None:
                self.w_min_max[0] = w.min()
                self.w_min_max[1] = w.max()
            # out
            # fmt: off
            if out is not None:
                if self.num_samples_to_track is None or self.q_num_tracked_samples.item() < self.num_samples_to_track:
                    if torch.isinf(self.out_running_min_max).any():
                        self.out_running_min_max = out.flatten().float().quantile(self.q_min_max_quantile).to(dtype)
                    else:
                        out_running_min_max = out.flatten().float().quantile(self.q_min_max_quantile).to(dtype)
                        self.out_running_min = self.stat_smooth_factor * self.out_running_min_max + (1 - self.stat_smooth_factor) * out_running_min_max
            # fmt: on

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return super().forward(x)
        else:
            self._update_running_stats(x, self.weight, None)

            x_q, _ = optical_compute_quantized_forward_fn(
                x,
                quant_levels=self.quant_levels,
                min_val=self.x_running_min_max[0].item(),
                max_val=self.x_running_min_max[1].item(),
                lut_min=None,
                quant_mode="det",
                seed=0,
            )
            w_q, _ = optical_compute_quantized_forward_fn(
                self.weight,
                quant_levels=self.quant_levels,
                min_val=self.w_min_max[0].item(),
                max_val=self.w_min_max[1].item(),
                lut_min=self.q_lut_min,
                quant_mode="det",
                seed=0,
            )
            out = torch.nn.functional.linear(x_q, w_q, self.bias)
            self._update_running_stats(None, None, out)
            out_q, new_seed = optical_compute_quantized_forward_fn(
                out,
                quant_levels=self.quant_levels,
                min_val=self.out_running_min_max[0].item(),
                max_val=self.out_running_min_max[1].item(),
                lut_min=None,
                quant_mode="rand",
                seed=self.seed,
            )
            self.seed = new_seed
            return out_q

    def extra_repr(self) -> str:
        return (
            f"q_bypass={self.bypass}, q_levels={self.quant_levels}, q_lut_min={self.q_lut_min}, "
            f"q_samples_to_track={self.num_samples_to_track}, q_smooth_factor={self.stat_smooth_factor}, "
            f"q_min_max_quantile={self.q_min_max_quantile}, x_running_min_max={self.x_running_min_max}, "
            f"w_min_max={self.w_min_max}, out_running_min_max={self.out_running_min_max}, "
            f"q_num_tracked_samples={self.q_num_tracked_samples}"
        )

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        q_levels: int = 256,
        q_lut_min: float | None = 0.020040,
        q_samples_to_track: int | None = None,
        q_smooth_factor: float = 0.9,
        q_min_quantile: float = 0.001,
        q_max_quantile: float = 0.999,
        q_init_seed: int = 0,
        q_bypass: bool = False,
    ) -> "OpticalComputeQLinear":
        new_fc = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            linear.weight.dtype,
            q_levels,
            q_lut_min,
            q_samples_to_track,
            q_smooth_factor,
            q_min_quantile,
            q_max_quantile,
            q_init_seed,
            q_bypass,
        )
        with torch.no_grad():
            new_fc.weight.copy_(linear.weight)
            if linear.bias is not None:
                new_fc.bias.copy_(linear.bias)
        return new_fc
