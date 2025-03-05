# MASE-Triton

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
- [Random BitFlip](/src/mase_triton/random_bitflip.py)
    - [random_bitflip_fn]: random bitflip function with backward support.
    - [RandomBitFLip]: subclass of `torch.nn.Module`

## Dev

1. Install [tox](https://tox.wiki/en/latest/index.html)
    ```
    pip install tox
    ```

2. Create Dev Environment
    ```
    tox -e dev
    ```