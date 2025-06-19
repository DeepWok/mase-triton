#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TRITON_INTERPRET"] = "0"

import random
import torch
import triton
import pandas as pd
import sys

sys.path.append("/home/jw3621/Projects/mase-triton")
from src.mase_triton.optical_compute.core.optical_morr import (
    morr_linear_fn,
    morr_linear_fn_mem,
    AllPassMORRCirculantLinear,
)

from src.mase_triton.optical_compute.core.optical_morr.optical_original.utils import toeplitz

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    raise RuntimeError("This benchmark requires a GPU")

DTYPE  = torch.float32

def morr_backward_accuracy_test(
        B=1, N=1, D_in=4, D_out=4, miniblock=4,
        dtype=DTYPE, device=DEVICE,
        rtol=1e-2, atol=1e-3):

    torch.manual_seed(42)

    # 1. Build identical inputs & weights
    x_ref = torch.randn(B, N, D_in, device=device, dtype=dtype, requires_grad=True)
    x_kern = x_ref.detach().clone().requires_grad_(True)

    module = AllPassMORRCirculantLinear(
        in_features=D_in,
        out_features=D_out,
        bias=False,
        config={
            "miniblock": miniblock,
        }
    ).to(device=device, dtype=dtype)
    module.enable_crosstalk()
    module.enable_phase_variation()
    module.set_phase_variation(phase_noise_std=0.04)
    module.set_crosstalk_coupling_matrix(coupling_factor=0.04)
    module.set_input_bitwidth(4)
    module.set_weight_bitwidth(4)
    module.disable_trainable_morr_bias()
    module.disable_trainable_morr_scale()

    # use exactly the same weight tensor for both paths
    w_init = torch.randn(module.grid_dim_y,
                         module.grid_dim_x,
                         miniblock,
                         device=device, dtype=dtype)

    with torch.no_grad():
        module.weight.copy_(w_init)
    w_ref = module.weight                      # ref path uses parameter tensor
    w_kern = w_init.clone().detach().requires_grad_(True)  # kernel path

    scale_kern = module.morr_output_scale.detach().clone().requires_grad_(True)
    bias_kern = None
    # bias_kern = module.morr_input_bias.detach().clone().requires_grad_(True)

    # 2. Forward + backward – pytorch
    out_ref  = module(x_ref)              # [B, N, D_out]
    loss_ref = out_ref.sum()
    loss_ref.backward()

    for name, p in module.named_parameters():
        if p.grad != None:
            shape = str(p.grad.shape) 
        else:
            shape = "None"
        print(f"{name:25s}: {shape}")

    grad_x_ref = x_ref.grad.detach().clone()
    grad_w_ref = module.weight.grad.detach().clone()
    grad_s_ref = module.morr_output_scale.grad.detach().clone()
    # grad_b_ref = module.morr_input_bias.grad.detach().clone()
    
    # 3. Forward + backward – triton kernel
    if module.morr_bias == None:
        kernel_morr_bias = torch.zeros(
            1,
            module.grid_dim_y,
            module.grid_dim_x,
            1,
            device=DEVICE,
            dtype=torch.float,
        )
    else:
        kernel_morr_bias = module.morr_bias.detach()
    
    out_kern, *_ = morr_linear_fn_mem(
        x_kern,
        w_kern,
        morr_input_bias = bias_kern if bias_kern else None,
        morr_output_scale = scale_kern,
        bias = module.bias,
        morr_input_scale=module.morr_input_scale,
        morr_bias = kernel_morr_bias,
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
        sigma_weight=module.sigma_weight,
        trainable_morr_scale=module.trainable_morr_scale, # bool
        morr_scale=module.morr_scale.detach() if module.trainable_morr_scale else None,
        weight_quant_gain=module.weight_quant_gain,
        seed = 42,
    )                                       # [B, N, D_out]

    loss_kern = out_kern.sum()
    loss_kern.backward()

    grad_x_kern = x_kern.grad.detach()
    grad_w_kern = w_kern.grad.detach()
    grad_s_kern = scale_kern.grad.detach()
    # grad_b_kern = bias_kern.grad.detach()

    # 4. Compare gradients
    def _report(name, g_ref, g_kern):
        diff = (g_ref - g_kern).abs()
        print(f"[{name}]  L_inf={diff.max():.3e}   L1_mean={diff.mean():.3e}")
        passed = torch.allclose(g_ref, g_kern, rtol=rtol, atol=atol)
        if passed:
            print("✅ gradients match within tolerance!")
        else:
            print("❌ gradient mismatch")
        return passed, diff.max(), diff.mean()

    print("---- Gradient comparison ---------------------------------------")
    # diff_max, diff_mean = [], []
    t1, maxx, meanx = _report("dL/dx", grad_x_ref, grad_x_kern)
    t2, maxw, meanw = _report("dL/dW", grad_w_ref, grad_w_kern)
    t3, maxs, means = _report("dL/dS", grad_s_ref, grad_s_kern)
    # t4 = _report("dL/dB", grad_b_ref, grad_b_kern)

    return [maxx, maxw, maxs], [meanx, meanw, means]
    

#%%

def unit_test():
    p, q, k = 2, 2, 2
    grad_w = torch.randn([1, p, q, k, k]) # [1, p, q, k, k]
    grad_w2 = grad_w.clone()

    # 2. weight = toeplitz(weight)
    k = grad_w.size(-1)
    row = torch.arange(k)[:, None]        # (k,1)
    col = torch.arange(k)[None, :]        # (1,k)
    idx = (row - col) & (k - 1) if (k & (k-1)) == 0 else (row - col + k) % k

    idx = idx.expand(grad_w.shape) # [1, p, q, k, k]
    buffer = torch.zeros_like(grad_w) # [1, p, q, k, k]
    buffer.scatter_add_(-2, idx, grad_w) # [1, p, q, k, k]

    grad_w = buffer.sum(dim=-1, keepdim=True).squeeze(0).squeeze(-1)
    print(grad_w)
    print(grad_w.shape)

    # autograd
    weight = torch.randn([p, q, k], requires_grad=True)
    out = toeplitz(weight).unsqueeze(0)  # [1, p, q, k, k]
    out.backward(grad_w2)

    # 5) inspect the gradient w.r.t. the input column
    print("col.grad:\n", weight.grad)
    print(weight.shape)

    if torch.allclose(weight.grad, grad_w, rtol=1e-3, atol=1e-5):
        print("✅ gradients match within tolerance!")
    else:
        print("❌ gradient mismatch")
# %%
max_rows, mean_rows = [], []

# miniblock_vals = [4]
# B_vals = [1, 2, 4, 8, 16]
# N_vals = [1, 2, 4, 8, 16]
# Din_vals = [64, 128, 256, 512, 768]
# Dout_vals = [64, 128, 256, 512, 768]
# for miniblock in miniblock_vals:
#         for B in B_vals:
#             for N in N_vals:
#                 for D_in in Din_vals:
#                     for D_out in Dout_vals:
config_list = [(1, 1, 8, 8, 4), (16, 16, 256, 256, 4), (32, 32, 768, 768, 4)]
for B, N, D_in, D_out, miniblock in config_list:
                        diff_max, diff_mean = morr_backward_accuracy_test(B=B, N=N, D_in=D_in, D_out=D_in, miniblock=miniblock)
                        
                        # --- make sure we have plain Python scalars ----------------
                        diff_max  = [d.item() if isinstance(d, torch.Tensor) else float(d)
                                    for d in diff_max]
                        diff_mean = [d.item() if isinstance(d, torch.Tensor) else float(d)
                                    for d in diff_mean]

                        # --- build one row for each table ---------------------------
                        meta = dict(miniblock=miniblock, B=B, N=N,
                                    D_in=D_in, D_out=D_out)           # shared columns

                        max_rows.append( {**meta, "x": diff_max[0],
                                                "weight": diff_max[1],
                                                "scale": diff_max[2]} )

                        mean_rows.append({**meta, "x": diff_mean[0],
                                                "weight": diff_mean[1],
                                                "scale": diff_mean[2]} )

# -------------------------------------------------------------------------
df_max  = pd.DataFrame(max_rows)
df_mean = pd.DataFrame(mean_rows)

# Optional: sort rows for consistent LaTeX output
sort_cols = ["miniblock", "B", "N", "D_in", "D_out"]
df_max  = df_max.sort_values(sort_cols).reset_index(drop=True)
df_mean = df_mean.sort_values(sort_cols).reset_index(drop=True)

# -------------------------------------------------------------------------
# 1. Quick on-screen preview
print("\n--- L∞ max abs-error table --------------------------------")
print(df_max.to_string(index=False, float_format="%.3e"))
print("\n--- Mean abs-error table ---------------------------------")
print(df_mean.to_string(index=False, float_format="%.3e"))

# 2. Produce LaTeX (caption/label are optional)
latex_max = df_max.to_latex(index=False,
                            float_format="%.2e",
                            column_format="rrrrr|rrr",
                            caption="Maximum absolute gradient differences ($L_\\infty$).",
                            label="tab:max_grad")

latex_mean = df_mean.to_latex(index=False,
                              float_format="%.2e",
                              column_format="rrrrr|rrr",
                              caption="Mean absolute gradient differences ($L_1$).",
                              label="tab:mean_grad")

print("\n--- copy-paste LaTeX for max table ------------------------")
print(latex_max)
print("\n--- copy-paste LaTeX for mean table -----------------------")
print(latex_mean)

# 3. Save .tex files if you want \input{} them later
with open("grad_max_table.tex",  "w") as f: f.write(latex_max)
with open("grad_mean_table.tex", "w") as f: f.write(latex_mean)
# %%
