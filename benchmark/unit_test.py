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



def uniform_quantize(k, gradient_clip=False):
    class qfn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2 ** k - 1)
                out = torch.round(input * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            if gradient_clip:
                grad_input.clamp_(-1, 1)
            return grad_input

    return qfn.apply

def quantize(x: torch.Tensor, bit: int = 8):
    uniform_q = uniform_quantize(k=bit, gradient_clip=True)

    weight = torch.tanh(x)  # [-1, 1]
    r = torch.max(torch.abs(weight.data))
    weight = weight + r
    # weight = weight / 2 + 0.5
    weight_q = uniform_q(weight / (2 * r)) * 2 * r

    return weight_q

def custom_backward(x: torch.Tensor,
                       bit: int = 8,
                       gradient_clip: bool = True,
                       grad_output: torch.Tensor | None = None) -> torch.Tensor:

    if grad_output is None:
        grad_output = torch.ones_like(x)

    weight = torch.tanh(x)
    r      = torch.max(weight.abs()).detach() + 1e-12      # ε avoids /0

    # dL/dq
    g = grad_output * (2 * r)

    # straight-through estimator + optional clipping
    if gradient_clip:
        g = g.clamp(-1.0, 1.0)          # this is dL/dy
    # divide by 2r (dy/d(weight + r))
    g = g / (2 * r)

    # dL/dweight  (addition of r is constant wrt weight)
    # multiply by d(tanh)/dx
    grad_x = g * (1.0 - weight.pow(2))

    return grad_x
    


if __name__ == "__main__":
    torch.manual_seed(0)

    bit = 8                        # try 1, 4, 8, 32 for fun
    x   = torch.randn(16, 16, requires_grad=True)  # random tensor

    # forward & autograd backward
    y    = quantize(x, bit)
    loss = y.sum()
    loss.backward()
    grad_autograd = x.grad.clone()

    # analytic gradient
    grad_custom = custom_backward(x.detach(), bit)

    # compare
    max_err = (grad_autograd - grad_custom).abs().max().item()
    print(f"Max |grad_autograd - grad_custom| = {max_err:.3e}")

    tol = 1e-6
    assert max_err < tol, "Mismatch between custom and autograd gradients!"
    print("✓ custom backward matches autograd\n")



# %%
# torch_output, triton_output, _ = morr_accuracy_test(B=1, N=1, D_in=8, D_out=8, miniblock=2)
# torch_output, triton_output, _ = morr_accuracy_test(B=1, N=1, D_in=16, D_out=16, miniblock=4)
# torch_output, triton_output, _ = morr_accuracy_test(B=10, N=10, D_in=512, D_out=512, miniblock=4)
# print(torch_output)
# print(triton_output)

# %%

