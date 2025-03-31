# MASE-Triton

Software-emulation & acceleration triton kernels for [MASE](https://github.com/DeepWok/mase).

## Install

### PyPI

`🚧 TODO`

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


## Dev

1. Install [tox](https://tox.wiki/en/latest/index.html)
    ```
    pip install tox
    ```

2. Create Dev Environment
    ```
    tox -e dev
    ```