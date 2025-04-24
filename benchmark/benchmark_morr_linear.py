#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TRITON_INTERPRET"] = "1"

import random
import torch
import triton

import sys

sys.path.append("/home/jw3621/Projects/mase-triton")
from src.mase_triton.optical_compute.core.optical_morr import (
    morr_linear_fn,
    AllPassMORRCirculantLinear,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    raise RuntimeError("This benchmark requires a GPU")

def get_morr_linear_benchmark_configs():
    configs = []
    # Varying input sizes to benchmark
    # batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # in_features = [2048]
    # out_features = [2048]
    batch_sizes = [256]
    in_features = [1024]
    out_features = [1024]

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
            line_vals=["pytorch", "triton"],
            line_names=["Pytorch MORR Linear", "Triton MORR Linear"],
            styles=[("blue", "-"), ("orange", "-")],
            ylabel="time (ms)",
            plot_name=f"morr_linear_performance",
            args={},
        )
    )
    return configs


def morr_accuracy_test(B, N, D_in, D_out, miniblock=4):
    torch_dtype = torch.float32

    x = torch.randn(B, N, D_in, device=DEVICE, dtype=torch_dtype)

    # Create the PyTorch module
    module = AllPassMORRCirculantLinear(
        in_features=D_in, 
        out_features=D_out, 
        bias=False, 
        config={
            miniblock: miniblock,
        }
    ).to(DEVICE)
    grid_dim_y = module.grid_dim_y
    grid_dim_x = module.grid_dim_x
    morr_output_scale = module.morr_output_scale

    weight = torch.randn(
        grid_dim_y, grid_dim_x, miniblock, device=DEVICE, dtype=torch_dtype,
    )
    module.weight.data = weight

    torch_output = module(x)
    
    triton_output = morr_linear_fn(
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
        trainable_morr_bias=None,
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
    )


    print(f"torch_output shape: {torch_output.shape}, dtype: {torch_output.dtype}, device: {torch_output.device}")
    print(f"triton_output shape: {triton_output.shape}, dtype: {triton_output.dtype}, device: {triton_output.device}")

    # --- Comparison ---
    if torch_output.shape != triton_output.shape:
        print("\nError: Shapes do not match!")
        print(f"Torch shape: {torch_output.shape}")
        print(f"Triton shape: {triton_output.shape}")
    else:
        # Calculate the absolute difference
        abs_diff = torch.abs(torch_output - triton_output)

    # Calculate metrics
    max_abs_diff = torch.max(abs_diff)
    mean_abs_diff = torch.mean(abs_diff)

    are_close = torch.allclose(torch_output, triton_output, rtol=1e-5, atol=1e-8)

    print("\n--- Comparison Results ---")
    print(f"Are the outputs close (torch.allclose)? {are_close}")
    print(f"Maximum Absolute Difference: {max_abs_diff.item():.6e}") # .item() gets scalar value
    print(f"Mean Absolute Difference (MAE): {mean_abs_diff.item():.6e}")


@triton.testing.perf_report(get_morr_linear_benchmark_configs())
def benchmark_morr_linear(B, N, D_in, D_out, provider, miniblock=4):
    torch_dtype = torch.float32

    x = torch.randn(B, N, D_in, device=DEVICE, dtype=torch_dtype)

    # Create the PyTorch module
    module = AllPassMORRCirculantLinear(
        in_features=D_in, 
        out_features=D_out, 
        bias=False, 
        config={
            miniblock: miniblock,
        }
    ).to(DEVICE)
    grid_dim_y = module.grid_dim_y
    grid_dim_x = module.grid_dim_x
    morr_output_scale = module.morr_output_scale

    weight = torch.randn(
        grid_dim_y, grid_dim_x, miniblock, device=DEVICE, dtype=torch_dtype,
    )
    module.weight.data = weight

    if provider == "pytorch":
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
                trainable_morr_bias=None,
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

    return ms, min_ms, max_ms


#%%
df = benchmark_morr_linear.run(
    show_plots=True, 
    print_data=True,
)
# %%
# morr_accuracy_test(B=1, N=1, D_in=64, D_out=64, miniblock=4)
# %%
