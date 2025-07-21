# MASE-Triton

Software-emulation & acceleration triton kernels for [MASE](https://github.com/DeepWok/mase).

## Install

Please ensure you are using Python 3.11 or later, and run MASE-Triton on **CUDA-enabled GPU**.

### PyPI

```bash
pip install mase-triton
```

### Build from Source

1. Install [uv](https://docs.astral.sh/uv/)
2. Build the package

    ```bash
    uv build
    ```

    The wheel file can be found in `dist/` folder.
    You can install it by `pip install path/to/wheel/file.whl`


## Functionality
- Random Bitflip
    - [`functional APIs`](/src/mase_triton/random_bitflip/functional.py): random bitflip function with backward support.
    - [`layers.py`](/src/mase_triton/random_bitflip/layers.py): subclasses of `torch.nn.Module` that can be used in neural networks.
        - `RandomBitflipDropout`
        - `RandomBitflipLinear`
- [Optical Transformer](https://arxiv.org/abs/2302.10360)
    - [`functional APIs`](/src/mase_triton/optical_compute/functional.py): optical transformer function with backward support.
        - `ot_quantize`
        - `ot_linear`
        - `ot_matmul`
    - [`layers.py`](/src/mase_triton/optical_compute/layers.py): subclasses of `torch.nn.Module` that can be used in neural networks.
        - `OpticalTransformerLinear`
- [MXFP](https://arxiv.org/abs/2310.10537): Simulate MXFP formats on CPU & GPU using PyTorch & Triton.
    - [`functional`](/src/mase_triton/mxfp/functional/__init__.py)
        - `extract_mxfp_tensor`: Cast a tensor to MXFP format (extracting the shared exponent and Minifloat elements).
        - `compose_mxfp_tensor`: Cast an MXFP tensor to FP format (composing MXFP components).
        - `mxfp_linear`: functional linear operation with MXFP support.
        - `mxfp_matmul`: functional matrix multiplication with MXFP support.
    - [`layers`](/src/mase_triton/mxfp/layers.py)
        - `MXFPLinearPTQ`: Linear layer with MXFP support for post-training quantization (no back propagation support).
- Minifloat: Simulate minifloat formats on CPU & GPU using PyTorch & Triton.
    - [`functional`](/src/mase_triton/minifloat/functional/__init__.py)
        - `extract_minifloat_component`: Extract minifloat components from a tensor.
        - `compose_minifloat_component`: Compose minifloat components back to a tensor.
        - `quantize_dequantize`: Quantize and dequantize tensors using minifloat format.
        - `minifloat_linear`: functional linear operation with minifloat support.
        - `minifloat_matmul`: functional matrix multiplication with minifloat support.
    - [`layers`](/src/mase_triton/minifloat/layers.py)
        - `MinifloatLinearPTQ`: Linear layer with minifloat support for post-training quantization (no back propagation support).


## Dev

1. Install [uv](https://docs.astral.sh/uv/)
2. Install dependencies for development

    ```bash
    uv sync
    ```