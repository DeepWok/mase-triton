# %%
import random

import torch
import triton

from mase_triton.random_bitflip.core import random_bitflip_fn


def get_random_bitflip_fn_forward_benchmark_configs():
    configs = []
    n_elements_options = [1024 * 1024, 2048 * 1024, 1024 * 4096]
    n_halves_options = [1, 10, 20]
    zero_out_options = [True, False]

    for dtype in ["float32", "float16", "bfloat16"]:
        for n_halves in n_halves_options:
            for zero_out in zero_out_options:
                configs.append(
                    triton.testing.Benchmark(
                        x_names=["n_elements"],
                        x_vals=n_elements_options,
                        line_arg="provider",
                        line_vals=["random_bitflip_fn", "Tensor.clone()"],
                        line_names=["random_bitflip_fn", "Tensor.clone()"],
                        styles=[("blue", "-"), ("orange", "-")],
                        ylabel="Bandwidth (GB/s)",
                        plot_name=f"random_bitflip_fn (forward, {dtype}, n_halves: {n_halves}, enable_zero_out: {zero_out})",
                        args={
                            "dtype": dtype,
                            "n_halves": n_halves,
                            "zero_out": zero_out,
                        },
                    )
                )
    return configs


@triton.testing.perf_report(get_random_bitflip_fn_forward_benchmark_configs())
def benchmark_random_bitflip_fn_forward(
    n_elements: int, dtype: str, n_halves: int, zero_out: bool, provider: str
):
    x = torch.randn(n_elements, device="cuda", dtype=getattr(torch, dtype))
    quantiles = [0.5, 0.2, 0.8]
    if provider == "Tensor.clone()":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x.clone(), quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: random_bitflip_fn(
                x,
                exp_halves=n_halves,
                frac_halves=n_halves,
                seed_exp=random.randint(0, 100000),
                seed_frac=random.randint(0, 100000),
                zero_out_threshold=10 if zero_out else None,
            ),
            quantiles=quantiles,
        )
    gb_per_sec = lambda ms: (2 * n_elements * x.element_size()) / 2 ** 30 / (ms / 1000)
    return gb_per_sec(ms), gb_per_sec(min_ms), gb_per_sec(max_ms)


# %%
df = benchmark_random_bitflip_fn_forward.run(
    show_plots=True, print_data=True, return_df=True
)
