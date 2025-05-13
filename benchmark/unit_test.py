#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRITON_INTERPRET"] = "0"

import random
import torch
import triton

import sys

sys.path.append("/home/jw3621/Projects/mase-triton")
from src.mase_triton.optical_compute.core.optical_morr.linear import (
    morr_linear_fn,
)
from src.mase_triton.optical_compute.core.optical_morr.linear_mem import (
    morr_linear_fn_mem,
)
from src.mase_triton.optical_compute.core.optical_morr.optical_original.modules import (
    AllPassMORRCirculantLinear,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    raise RuntimeError("This benchmark requires a GPU")


def morr_accuracy_test(B, N, D_in, D_out, miniblock=4):
    torch_dtype = torch.float32
    torch.manual_seed(42)

    # Create the PyTorch module
    module = AllPassMORRCirculantLinear(
        in_features=D_in, 
        out_features=D_out, 
        bias=False, 
        config={
            "miniblock": miniblock,
            "trainable_morr_bias": True,
            "trainable_morr_scale": True,
        }
    ).to(DEVICE)
    # module.enable_crosstalk()
    # module.enable_phase_variation()
    # module.set_phase_variation(phase_noise_std=0.04)
    # module.set_crosstalk_coupling_matrix(coupling_factor=0.04)

    grid_dim_y = module.grid_dim_y
    grid_dim_x = module.grid_dim_x
    morr_output_scale = module.morr_output_scale

    x = torch.randn(B, N, D_in, device=DEVICE, dtype=torch_dtype)
    weight = torch.randn(grid_dim_y, grid_dim_x, miniblock, device=DEVICE, dtype=torch_dtype)
    morr_input_bias = torch.randn(
        grid_dim_y,
        grid_dim_x,
        device=DEVICE,
        dtype=torch_dtype,
    )
    # x = (torch.arange(1, B*N*D_in + 1, 1, device=DEVICE, dtype=torch_dtype).reshape(B, N, D_in)) * 0.1 
    # weight = (torch.arange(1, grid_dim_y*grid_dim_x*miniblock + 1, 1, device=DEVICE, dtype=torch_dtype).reshape(grid_dim_y, grid_dim_x, miniblock)) * 0.1

    module.weight.data = weight
    if module.morr_input_bias != None:
        module.morr_input_bias.data = morr_input_bias

    # pytorch inference
    torch_output = module(x)
    
    # cuda kernel inference
    triton_output, seed,  *_ = morr_linear_fn_mem( 
        x,
        weight,
        morr_input_bias = module.morr_input_bias,
        morr_output_scale = module.morr_output_scale,
        bias = module.bias,
        morr_bias = module.morr_bias.detach(),
        grid_dim_x = module.grid_dim_x,
        grid_dim_y = module.grid_dim_y,
        miniblock = miniblock,
        enable_thermal_crosstalk=module.enable_thermal_crosstalk,
        crosstalk_factor=None if not module.enable_thermal_crosstalk else module.crosstalk_factor,
        enable_phase_noise=module.enable_phase_noise,
        phase_noise_std=None if not module.enable_phase_noise else module.phase_noise_std,
        trainable_morr_bias=module.trainable_morr_bias, # bool
        mrr_a=module.mrr_a,
        mrr_r=module.mrr_r,
        finegrain_drop_mask=None,
        in_features = module.in_features,
        in_features_pad = module.in_features_pad,
        out_features = module.out_features,
        out_features_pad = module.out_features_pad,
        in_bit = module.in_bit,
        w_bit = module.w_bit,
        morr_fwhm = module.morr_fwhm,
        seed = 42,
    )

    # --- Comparison ---
    print(f"torch_output shape: {torch_output.shape}, dtype: {torch_output.dtype}, device: {torch_output.device}")
    print(f"triton_output shape: {triton_output.shape}, dtype: {triton_output.dtype}, device: {triton_output.device}")

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

    are_close = torch.allclose(torch_output, triton_output, rtol=1e-3, atol=1e-5)

    print("\n--- Comparison Results ---")
    print(f"Are the outputs close (torch.allclose)? {are_close}")
    print(f"Maximum Absolute Difference: {max_abs_diff.item():.6e}") # .item() gets scalar value
    print(f"Mean Absolute Difference (MAE): {mean_abs_diff.item():.6e}")

    return torch_output, triton_output, are_close




# %%
# torch_output, triton_output, _ = morr_accuracy_test(B=1, N=1, D_in=8, D_out=8, miniblock=2)
# torch_output, triton_output, _ = morr_accuracy_test(B=1, N=1, D_in=16, D_out=16, miniblock=4)
# torch_output, triton_output, _ = morr_accuracy_test(B=10, N=10, D_in=512, D_out=512, miniblock=4)
# print(torch_output)
# print(triton_output)

# %%

miniblock_vals = [2, 4, 8]
B_vals = [2, 4, 8]
N_vals = [2, 4, 8]
Din_vals = [128, 256, 512, 1024]
Dout_vals = [128, 256, 512, 1024]
for miniblock in miniblock_vals:
        for B in B_vals:
            for N in N_vals:
                for D in Din_vals:
                    for D_out in Dout_vals:
                        torch_output, triton_output, are_close = morr_accuracy_test(B=B, N=N, D_in=D, D_out=D_out, miniblock=miniblock)
                        assert are_close == True, f"Test Fail with B={B}, N={N}, D_in={D}, D_out={D_out}, miniblock={miniblock}"
# %%
