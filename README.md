# MASE-Triton

Software-emulation & acceleration triton kernels for [MASE](https://github.com/DeepWok/mase).

## Install

Please ensure you are using Python 3.11 or later, and run MASE-Triton on **CUDA-enabled GPU**.

### PyPI

```bash
pip install mase-triton
```

### Build from Source

1. Install tox

    ```bash
    pip install tox
    ```

2. Build & Install

    ```bash
    tox -e build
    ```

    Then the wheel file will be generated in `dist/` folder.
    You can install it by `pip install path/to/wheel/file.whl`


## Functionality
- Random Bitflip
    - [`random_bitflip_fn`](/src/mase_triton/random_bitflip/core.py): random bitflip function with backward support.
    - [`layers.py`](/src/mase_triton/random_bitflip/layers.py): subclasses of `torch.nn.Module` that can be used in neural networks.
        - `RandomBitflipDropout`
        - `RandomBitflipLinear`
- MXFP: Simulate MXFP formats (Note that subnormal numbers are flushed to zero)
    - [`functional`](/src/mase_triton/mxfp/functional/__init__.py)
        - `extract_mxfp_tensor`: Cast a tensor to MXFP format (extracting the shared exponent and Minifloat elements).
        - `compose_mxfp_tensor`: Cast an MXFP tensor to FP format (composing MXFP components).
        - `mxfp_linear`: functional linear operation with MXFP support.
        - `mxfp_matmul`: functional matrix multiplication with MXFP support.
    - [`layers`](/src/mase_triton/mxfp/layers.py)
        - `MXFPLinearPTQ`: Linear layer with MXFP support for post-training quantization (no back propagation support).


## Dev

1. Install [tox](https://tox.wiki/en/latest/index.html)
    ```
    pip install tox
    ```

2. Create Dev Environment
    ```
    tox -e dev
    ```