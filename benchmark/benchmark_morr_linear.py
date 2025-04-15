#%%
import random

import torch
import triton

import sys

sys.path.append("/home/jw3621/Projects/mase-triton")
from src.mase_triton.optical_compute.core.optical_morr import (
    morr_linear_fn,
    AllPassMORRCirculantLinear,
)

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def get_morr_linear_benchmark_configs():
    configs = []
    # Varying input sizes to benchmark
    batch_sizes = [16, 32, 64, 128, 256]
    in_features = [512]
    out_features = [512]

    x_vals = []
    for B in batch_sizes:
        for D_in in in_features:
            for D_out in out_features:
                x_vals.append([B, 1, D_in, D_out])

    configs.append(
        triton.testing.Benchmark(
            x_names=["B", "N", "D_in", "D_out"],
            x_vals=x_vals,
            line_arg="provider",
            line_vals=["triton"],
            line_names=["Triton MORR Linear"],
            styles=[("orange", "-")],
            ylabel="time (ms)",
            plot_name=f"morr_linear_performance",
            args={},
        )
    )
    return configs


@triton.testing.perf_report(get_morr_linear_benchmark_configs())
def benchmark_morr_linear(B, N, D_in, D_out, provider, miniblock=4):
    torch_dtype = torch.float

    x = torch.randn(B, N, D_in, device=DEVICE, dtype=torch_dtype)

    # Create the PyTorch module
    module = AllPassMORRCirculantLinear(
        in_features=D_in, out_features=D_out, bias=False, config={miniblock: miniblock,}
    ).to(DEVICE)
    grid_dim_y = module.grid_dim_y
    grid_dim_x = module.grid_dim_x
    morr_output_scale = module.morr_output_scale

    weight = torch.ones(
        grid_dim_y, grid_dim_x, miniblock, device=DEVICE, dtype=torch_dtype,
    )
    module.weight.data = weight

    if provider == "pytorch":
        # Run benchmark
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: module(x), quantiles=quantiles
        )
    if provider == "triton":
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: morr_linear_fn(
                x,
                weight,
                bias = None,
                grid_dim_x = module.grid_dim_x,
                grid_dim_y = module.grid_dim_y,
                miniblock = miniblock,
                enable_thermal_crosstalk=module.enable_thermal_crosstalk,
                crosstalk_factor=None if not module.enable_thermal_crosstalk else module.crosstalk_factor,
                enable_phase_noise=module.enable_phase_noise,
                phase_noise_std=None if not module.enable_phase_noise else module.phase_noise_std,
                trainable_morr_bias=module.trainable_morr_bias,
                mrr_a=module.mrr_a,
                mrr_r=module.mrr_r,
                finegrain_drop_mask=None,
                morr_output_scale = module.morr_output_scale,
                in_features = module.in_features,
                in_features_pad = module.in_features_pad,
                out_features = module.out_features,
                out_features_pad = module.out_features_pad,
                in_bit = module.in_bit,
                w_bit = module.w_bit,
            ),
            quantiles=quantiles,
        )

    # Calculate TFLOPS

    # ops = 2 * batch_size * in_features * out_features
    # tflops = lambda ms: ops * 1e-12 / (ms / 1000)

    print(ms, min_ms, max_ms)
    return ms, min_ms, max_ms


#%%
if __name__ == "__main__":
    df = benchmark_morr_linear.run(
        show_plots=True, 
        print_data=True,
    )