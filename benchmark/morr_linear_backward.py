#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRITON_INTERPRET"] = "1"

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


DTYPE  = torch.float32

def morr_backward_accuracy_test(
        B=1, N=1, D_in=4, D_out=4, miniblock=4,
        dtype=DTYPE, device=DEVICE,
        rtol=1e-3, atol=1e-5):

    torch.manual_seed(42)

    # 1. Build identical inputs & weights
    x_ref = torch.randn(B, N, D_in, device=device, dtype=dtype, requires_grad=True)
    x_kern = x_ref.detach().clone().requires_grad_(True)

    module = AllPassMORRCirculantLinear(
        in_features=D_in,
        out_features=D_out,
        bias=False,
        config={"miniblock": miniblock},
    ).to(device=device, dtype=dtype)

    # use exactly the same weight tensor for both paths
    w_init = torch.randn(module.grid_dim_y,
                         module.grid_dim_x,
                         miniblock,
                         device=device, dtype=dtype)

    with torch.no_grad():
        module.weight.copy_(w_init)
    w_ref = module.weight                      # ref path uses parameter tensor
    w_kern = w_init.clone().detach().requires_grad_(True)  # kernel path

    # 2. Forward + backward – pytorch
    out_ref  = module(x_ref)              # [B, N, D_out]
    loss_ref = out_ref.sum()
    loss_ref.backward()

    grad_x_ref = x_ref.grad.detach().clone()
    grad_w_ref = w_ref.grad.detach().clone()

    # 3. Forward + backward – triton kernel
    out_kern, *_ = morr_linear_fn(
        x_kern,
        w_kern,
        bias=None,
        grid_dim_x=module.grid_dim_x,
        grid_dim_y=module.grid_dim_y,
        miniblock=miniblock,
        enable_thermal_crosstalk=module.enable_thermal_crosstalk,
        crosstalk_factor=None if not module.enable_thermal_crosstalk else module.crosstalk_factor,
        enable_phase_noise=module.enable_phase_noise,
        phase_noise_std=None if not module.enable_phase_noise
                        else module.phase_noise_std,
        trainable_morr_bias=False,          # we disabled bias in this test
        mrr_a=module.mrr_a,
        mrr_r=module.mrr_r,
        finegrain_drop_mask=None,
        morr_output_scale=module.morr_output_scale,
        in_features=module.in_features,
        in_features_pad=module.in_features_pad,
        out_features=module.out_features,
        out_features_pad=module.out_features_pad,
        in_bit=module.in_bit,
        w_bit=module.w_bit,
    )                                       # [B, N, D_out]

    loss_kern = out_kern.sum()
    loss_kern.backward()

    grad_x_kern = x_kern.grad.detach()
    grad_w_kern = w_kern.grad.detach()

    # 4. Compare gradients
    def _report(name, g_ref, g_kern):
        diff = (g_ref - g_kern).abs()
        print(f"[{name}]  L_inf={diff.max():.3e}   L1_mean={diff.mean():.3e}")
        if torch.allclose(g_ref, g_kern, rtol=rtol, atol=atol):
            print("✅ gradients match within tolerance!")
        else:
            print("❌ gradient mismatch")

    print("---- Gradient comparison ---------------------------------------")
    _report("dL/dx", grad_x_ref, grad_x_kern)
    _report("dL/dW", grad_w_ref, grad_w_kern)

if __name__ == "__main__":
    morr_backward_accuracy_test()

#%%

from src.mase_triton.optical_compute.core.optical_morr.optical_original.utils import toeplitz

p, q, k = 1, 1, 4
grad_w = torch.randn([1, p, q, k, k]) # [1, p, q, k, k]
grad_w2 = grad_w.clone()

# 2. weight = toeplitz(weight)
k = grad_w.size(-1)
row = torch.arange(k)[:, None]        # (k,1)
col = torch.arange(k)[None, :]        # (1,k)
idx = (row - col) & (k - 1) if (k & (k-1)) == 0 else (row - col + k) % k

idx = idx.unsqueeze(0).unsqueeze(0).unsqueeze(0)# [1, p, q, k, k]
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
# %%
