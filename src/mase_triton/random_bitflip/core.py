import math

import torch
import triton
import triton.language as tl
from torch import Tensor

from ..about import PACKAGE_NAME
from ..dtype import TORCH_DTYPE_TO_TRITON


def calculate_flip_probability(prob_halves: int | None) -> float | None:
    if prob_halves is None:
        return None
    else:
        assert prob_halves > 0
        return 0.5**prob_halves


def find_nearest_prob_n_halves(prob: float | None) -> int | None:
    if prob is None:
        return None
    else:
        assert 0 < prob < 1
        return math.ceil(math.log2(1 / prob))


@triton.jit
def _get_four_randints(seed, offsets, BIN_DTYPE: tl.constexpr, N_ROUNDS: tl.constexpr):
    rint1, rint2, rint3, rint4 = tl.randint4x(seed, offsets, n_rounds=N_ROUNDS)
    rint1 = rint1.to(tl.uint32, bitcast=True).to(BIN_DTYPE)
    rint2 = rint2.to(tl.uint32, bitcast=True).to(BIN_DTYPE)
    rint3 = rint3.to(tl.uint32, bitcast=True).to(BIN_DTYPE)
    rint4 = rint4.to(tl.uint32, bitcast=True).to(BIN_DTYPE)
    return rint1, rint2, rint3, rint4


@triton.jit
def _cta_random_flip(
    set_bits,
    offsets,
    prob_halves: int,
    seed: int,
    BIN_DTYPE: tl.constexpr,
    PHILOX_N_ROUNDS: tl.constexpr,
):
    q = prob_halves // 4
    r = prob_halves % 4
    for i in range(q):
        rint1, rint2, rint3, rint4 = _get_four_randints(
            seed + i, offsets, BIN_DTYPE, PHILOX_N_ROUNDS
        )
        set_bits = set_bits & rint1 & rint2 & rint3 & rint4
    rint1, rint2, rint3, _ = _get_four_randints(
        seed + q, offsets, BIN_DTYPE, PHILOX_N_ROUNDS
    )
    if r >= 1:
        set_bits = set_bits & rint1
    if r >= 2:
        set_bits = set_bits & rint2
    if r >= 3:
        set_bits = set_bits & rint3
    return set_bits


@triton.jit
def _create_sign_exp_mask(INPUT_DTYPE: tl.constexpr):
    if INPUT_DTYPE == tl.float16:
        exp_mask = 0xFC00  # bin = 1111_1100_0000_0000
        exp_mask = tl.full((1,), exp_mask, dtype=tl.uint16)
    elif INPUT_DTYPE == tl.bfloat16:
        exp_mask = 0xFF80  # bin = 1111_1111_1000_0000
        exp_mask = tl.full((1,), exp_mask, dtype=tl.uint16)
    else:
        # tl.float32
        exp_mask = 0xFF800000  # bin = 1111_1111_1000_0000_0000_0000_0000_0000
        exp_mask = tl.full((1,), exp_mask, dtype=tl.uint32)
    exp_mask = tl.constexpr(exp_mask)
    return exp_mask


@triton.jit
def _create_frac_mask(INPUT_DTYPE: tl.constexpr):
    if INPUT_DTYPE == tl.float16:
        frac_mask = 0x3FF  # bin = 0000_0011_1111_1111
        frac_mask = tl.full((1,), frac_mask, dtype=tl.uint16)
    elif INPUT_DTYPE == tl.bfloat16:
        frac_mask = 0x7F  # bin = 0000_0000_0111_1111
        frac_mask = tl.full((1,), frac_mask, dtype=tl.uint16)
    else:
        # tl.float32
        frac_mask = 0x7FFFFF  # bin = 0000_0000_0111_1111_1111_1111_1111_1111
        frac_mask = tl.full((1,), frac_mask, dtype=tl.uint32)
    frac_mask = tl.constexpr(frac_mask)
    return frac_mask


def _get_autotune_configs_forward():
    # small batch, not sure what is the right default cnnfig here.
    block_sizes = [128, 256, 512, 1024]
    stages = [1, 2, 3, 4]

    configs = []
    for bs in block_sizes:
        for s in stages:
            configs.append(triton.Config({"BLOCK_SIZE": bs}, num_stages=s))
    return configs


@triton.autotune(
    configs=_get_autotune_configs_forward(),
    key=["n_elements"],
    use_cuda_graph=False,
)
@triton.jit
def _random_bitflip_forward_kernel(
    x_ptr,
    output_ptr,
    n_elements: int,
    exp_halves: int,  # 0.5 ** exp_halves for exponent bits,
    frac_halves: int,  # 0.5 ** frac_halves for fraction bits
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float,
    SKIP_EXP_FLIP: tl.constexpr,
    SKIP_FRAC_FLIP: tl.constexpr,
    ENABLE_ZERO_OUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    BIN_DTYPE: tl.constexpr,
    EXP_PHILOX_N_ROUNDS: tl.constexpr,
    FRAC_PHILOX_N_ROUNDS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # load x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(BIN_DTYPE, bitcast=True)

    # flip exp bits
    # random flip using mask: https://stackoverflow.com/a/35796081
    if not SKIP_EXP_FLIP:
        bits_to_flip = ~tl.zeros(x.shape, dtype=BIN_DTYPE)  # all bits set to 1
        bits_to_flip = _cta_random_flip(
            bits_to_flip, offsets, exp_halves, seed_exp, BIN_DTYPE, EXP_PHILOX_N_ROUNDS
        )
        exp_mask = _create_sign_exp_mask(INPUT_DTYPE)
        x = x ^ (bits_to_flip & exp_mask)

    # flip frac bits
    if not SKIP_FRAC_FLIP:
        bits_to_flip = ~tl.zeros(x.shape, dtype=BIN_DTYPE)  # all bits set to 1
        bits_to_flip = _cta_random_flip(
            bits_to_flip,
            offsets,
            frac_halves,
            seed_frac,
            BIN_DTYPE,
            FRAC_PHILOX_N_ROUNDS,
        )
        frac_mask = _create_frac_mask(INPUT_DTYPE)
        x = x ^ (bits_to_flip & frac_mask)

    x = x.to(INPUT_DTYPE, bitcast=True)

    if ENABLE_ZERO_OUT:
        activated = x.abs() < zero_out_threshold
        x = tl.where(activated, x, 0.0)

    # store x
    tl.store(output_ptr + offsets, x, mask=mask)


BIT_FLIP_DTYPE_MAP = {
    torch.float32: tl.uint32,
    torch.float16: tl.uint16,
    torch.bfloat16: tl.uint16,
}


def _get_philox_n_rounds(n_halves: int):
    """
    numel = 2048 * 2048 = 4194304
    +---------------+--------------+------------------------+----------------+------------------------+---------------+------------------------+------------------+------------------------+
    |  input_dtype  | exp_n_halves |         exp_p          | exp_p*exp_bits |       exp_ratio        | frac_n_halves |         frac_p         | frac_p*frac_bits |       frac_ratio       |
    +---------------+--------------+------------------------+----------------+------------------------+---------------+------------------------+------------------+------------------------+
    | torch.float16 |      1       |          0.5           |    12582912    |  0.49987002213795984   |       1       |          0.5           |     20971520     |   0.5001420736312866   |
    | torch.float16 |      2       |          0.25          |    6291456     |  0.24993399779001868   |       2       |          0.25          |     10485760     |   0.2500120162963867   |
    | torch.float16 |      3       |         0.125          |    3145728     |  0.12495628992716468   |       3       |         0.125          |     5242880      |  0.12501926422119136   |
    | torch.float16 |      4       |         0.0625         |    1572864     |  0.06251192092895508   |       4       |         0.0625         |     2621440      |  0.06253018379211428   |
    | torch.float16 |      5       |        0.03125         |     786432     |  0.030786554018656376  |       5       |        0.03125         |     1310720      |  0.03129191398620601   |
    | torch.float16 |      6       |        0.015625        |     393216     |  0.015468835830688477  |       6       |        0.015625        |      655360      |  0.016313576698303245  |
    | torch.float16 |      7       |       0.0078125        |     196608     |  0.00779004891713464   |       7       |       0.0078125        |      327680      |  0.007802462577819802  |
    | torch.float16 |      8       |       0.00390625       |     98304      |  0.003905018170674679  |       8       |       0.00390625       |      163840      | 0.0038976669311523438  |
    | torch.float16 |      9       |      0.001953125       |     49152      |  0.001950820287068722  |       9       |      0.001953125       |      81920       | 0.0019538164138793723  |
    | torch.float16 |      10      |      0.0009765625      |     24576      | 0.0009798606236776086  |      10       |      0.0009765625      |      40960       | 0.0009843349456787331  |
    | torch.float16 |      11      |     0.00048828125      |     12288      | 0.0004904270172119141  |      11       |     0.00048828125      |      20480       | 0.0004933357238769975  |
    | torch.float16 |      12      |     0.000244140625     |      6144      | 0.0002469221750894812  |      12       |     0.000244140625     |      10240       | 0.00025000572204592064 |
    | torch.float16 |      13      |    0.0001220703125     |      3072      |  0.000119169553120968  |      13       |    0.0001220703125     |       5120       | 0.0001203775405883567  |
    | torch.float16 |      14      |    6.103515625e-05     |      1536      | 5.9604644775390625e-05 |      14       |    6.103515625e-05     |       2560       | 6.0367584228560034e-05 |
    | torch.float16 |      15      |    3.0517578125e-05    |      768       | 2.9206275939941406e-05 |      15       |    3.0517578125e-05    |       1280       | 2.9420852661110608e-05 |
    | torch.float16 |      16      |   1.52587890625e-05    |      384       | 1.398722330725466e-05  |      16       |   1.52587890625e-05    |       640        | 1.5544891357466284e-05 |
    | torch.float16 |      17      |   7.62939453125e-06    |      192       | 7.311503092410909e-06  |      17       |   7.62939453125e-06    |       320        |   7.62939453125e-06    |
    | torch.float16 |      18      |   3.814697265625e-06   |       96       | 4.0133794149133095e-06 |      18       |   3.814697265625e-06   |       160        | 3.4332275390402955e-06 |
    | torch.float16 |      19      |  1.9073486328125e-06   |       48       | 1.947085062625753e-06  |      19       |  1.9073486328125e-06   |        80        | 2.050399780295642e-06  |
    | torch.float16 |      20      |  9.5367431640625e-07   |       24       | 8.344650268554688e-07  |      20       |  9.5367431640625e-07   |        40        | 1.1205673218217527e-06 |
    | torch.float16 |      21      |  4.76837158203125e-07  |       12       | 2.781550089148155e-07  |      21       |  4.76837158203125e-07  |        20        | 4.529953002707643e-07  |
    | torch.float16 |      22      | 2.384185791015625e-07  |       6        | 7.947285973752827e-08  |      22       | 2.384185791015625e-07  |        10        | 1.9073486323684108e-07 |
    | torch.float16 |      23      | 1.1920928955078125e-07 |       3        | 7.947285973752827e-08  |      23       | 1.1920928955078125e-07 |        5         | 1.1920928955078125e-07 |
    | torch.float16 |      24      | 5.960464477539063e-08  |       2        | 3.973642981325298e-08  |      24       | 5.960464477539063e-08  |        2         | 4.768371586472142e-08  |

    When n_halves >= 24, the error_ratio does not match the expected value.

    | torch.float16 |      25      | 2.9802322387695312e-08 |       1        |          0.0           |      25       | 2.9802322387695312e-08 |        1         | 2.384185793236071e-08  |
    | torch.float16 |      26      | 1.4901161193847656e-08 |       0        |          0.0           |      26       | 1.4901161193847656e-08 |        1         | 2.384185793236071e-08  |
    | torch.float16 |      27      | 7.450580596923828e-09  |       0        |          0.0           |      27       | 7.450580596923828e-09  |        0         | 2.384185793236071e-08  |
    | torch.float16 |      28      | 3.725290298461914e-09  |       0        |          0.0           |      28       | 3.725290298461914e-09  |        0         | 2.384185793236071e-08  |
    | torch.float16 |      29      | 1.862645149230957e-09  |       0        | 3.973642981325298e-08  |      29       | 1.862645149230957e-09  |        0         |          0.0           |
    | torch.float16 |      30      | 9.313225746154785e-10  |       0        |          0.0           |      30       | 9.313225746154785e-10  |        0         | 2.384185793236071e-08  |
    | torch.float16 |      31      | 4.656612873077393e-10  |       0        |          0.0           |      31       | 4.656612873077393e-10  |        0         | 2.384185793236071e-08  |
    +---------------+--------------+------------------------+----------------+------------------------+---------------+------------------------+------------------+------------------------+
    """
    if n_halves is None:
        return 0
    if n_halves < 13:
        return 10
    elif n_halves < 19:
        return 12
    if n_halves < 25:
        return 16
    else:
        return 30


@torch.library.custom_op(
    f"{PACKAGE_NAME}::random_bitflip_random_bitflip_forward",
    mutates_args={},
)
def random_bitflip_fn(
    x: Tensor,
    exp_halves: int | None,
    frac_halves: int | None,
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float | None,
) -> tuple[Tensor, int, int]:
    """Forward pass of random bit flip operation.

    Parameters
    ----------
    x : Tensor
        input tensor
    exp_halves : int | None
        the random bit flip probability for sign-exponent bits = 0.5^exp_halves.
        If None, no random bit flip is performed for sign-exponent bits.
    frac_halves : int | None
        the random bit flip probability for fraction bits = 0.5^frac_halves.
        If None, no random bit flip is performed for fraction bits.
    seed_exp : int
        the random seed for sign-exp bits. Note the same seed generates the same random bits,
        thus the seed needs to be updated after each call.
    seed_frac : int
        the random seed for sign-exp bits. Note the same seed generates the same random bits,
        thus the seed needs to be updated after each call.
    zero_out_threshold : float | None
        if not None, zero out the bits whose absolute value is less than this threshold (including NaN).
        if None, no zero out operation is performed.

    Returns
    -------
    tuple[Tensor, int, int]
        the output tensor, the updated seed_exp, and the updated seed_frac
    """
    assert x.dtype in BIT_FLIP_DTYPE_MAP
    assert zero_out_threshold is None or zero_out_threshold >= 0.0
    assert exp_halves is None or (exp_halves > 0 and exp_halves <= 24), (
        "pseudo RNG works for 24 halves at most"
    )
    assert frac_halves is None or (frac_halves > 0 and frac_halves <= 24), (
        "pseudo RNG works for 24 halves at most"
    )
    skip_exp_flip = exp_halves is None
    skip_frac_flip = frac_halves is None
    enable_zero_out = zero_out_threshold is not None
    if skip_exp_flip and skip_frac_flip:
        if enable_zero_out:
            output = torch.where(x.abs() < zero_out_threshold, x, 0.0)
        else:
            output = x.clone()
        return output, seed_exp, seed_frac
    else:
        x = x.contiguous()
        output = torch.empty_like(x)
        num_elements = x.numel()

        def grid(meta):
            return (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        with torch.cuda.device(x.device.index):
            _random_bitflip_forward_kernel[grid](
                x,
                output,
                n_elements=num_elements,
                exp_halves=exp_halves,
                frac_halves=frac_halves,
                seed_exp=seed_exp,
                seed_frac=seed_frac,
                zero_out_threshold=zero_out_threshold if enable_zero_out else 0.0,
                SKIP_EXP_FLIP=skip_exp_flip,
                SKIP_FRAC_FLIP=skip_frac_flip,
                ENABLE_ZERO_OUT=enable_zero_out,
                INPUT_DTYPE=TORCH_DTYPE_TO_TRITON[x.dtype],
                BIN_DTYPE=BIT_FLIP_DTYPE_MAP[x.dtype],
                EXP_PHILOX_N_ROUNDS=_get_philox_n_rounds(exp_halves),
                FRAC_PHILOX_N_ROUNDS=_get_philox_n_rounds(frac_halves),
            )
        if not skip_exp_flip:
            seed_exp = seed_exp + math.ceil(exp_halves / 4)
        if not skip_frac_flip:
            seed_frac = seed_frac + math.ceil(frac_halves / 4)

        return output, seed_exp, seed_frac


@random_bitflip_fn.register_fake
def _random_bitflip_forward_fake(
    x: Tensor,
    exp_halves: int | None,
    frac_halves: int | None,
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float | None,
) -> tuple[Tensor, int, int]:
    output = torch.empty_like(x, dtype=x.dtype)
    seed_exp = seed_exp + 1
    seed_frac = seed_frac + 1
    return output, seed_exp, seed_frac


def _get_autotune_configs_backward():
    block_sizes = [128, 256, 512, 1024]
    stages = [1, 2, 3, 4]

    configs = []
    for bs in block_sizes:
        for s in stages:
            configs.append(triton.Config({"BLOCK_SIZE": bs}, num_stages=s))
    return configs


@triton.autotune(
    configs=_get_autotune_configs_backward(),
    key=["n_elements"],
    use_cuda_graph=False,
)
@triton.jit
def _random_bitflip_zero_outed_backward_kernel(
    x_ptr,
    grad_y_ptr,
    grad_x_ptr,
    n_elements: int,
    exp_halves: int,  # 0.5 ** exp_halves for exponent bits,
    frac_halves: int,  # 0.5 ** frac_halves for fraction bits
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float,
    SKIP_EXP_FLIP: tl.constexpr,
    SKIP_FRAC_FLIP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    BIN_DTYPE: tl.constexpr,
    GRAD_DTYPE: tl.constexpr,
    EXP_PHILOX_N_ROUNDS: tl.constexpr,
    FRAC_PHILOX_N_ROUNDS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # load x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(BIN_DTYPE, bitcast=True)

    # flip exp bits
    # random flip using mask: https://stackoverflow.com/a/35796081
    if not SKIP_EXP_FLIP:
        bits_to_flip = ~tl.zeros(x.shape, dtype=BIN_DTYPE)  # all bits set to 1
        bits_to_flip = _cta_random_flip(
            bits_to_flip, offsets, exp_halves, seed_exp, BIN_DTYPE, EXP_PHILOX_N_ROUNDS
        )
        exp_mask = _create_sign_exp_mask(INPUT_DTYPE)
        x = x ^ (bits_to_flip & exp_mask)

    # flip frac bits
    if not SKIP_FRAC_FLIP:
        bits_to_flip = ~tl.zeros(x.shape, dtype=BIN_DTYPE)  # all bits set to 1
        bits_to_flip = _cta_random_flip(
            bits_to_flip,
            offsets,
            frac_halves,
            seed_frac,
            BIN_DTYPE,
            FRAC_PHILOX_N_ROUNDS,
        )
        frac_mask = _create_frac_mask(INPUT_DTYPE)
        x = x ^ (bits_to_flip & frac_mask)

    x = x.to(INPUT_DTYPE, bitcast=True)

    # zero out mask
    activated = x.abs() < zero_out_threshold

    grad_y = tl.load(grad_y_ptr + offsets, mask=mask)
    grad_x = tl.where(activated, grad_y, 0.0).to(GRAD_DTYPE)

    # store grad_x
    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)


@torch.library.custom_op(
    f"{PACKAGE_NAME}::_random_bitflip_backward",
    mutates_args={},
)
def _random_bitflip_backward(
    x: Tensor,
    grad_y: Tensor,
    exp_halves: int | None,
    frac_halves: int | None,
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float | None,
) -> Tensor:
    assert x.dtype in BIT_FLIP_DTYPE_MAP
    skip_exp_flip = exp_halves is None
    skip_frac_flip = frac_halves is None
    enable_zero_out = zero_out_threshold is not None

    if skip_exp_flip and skip_frac_flip:
        if enable_zero_out:
            grad_x = torch.where(x.abs() < zero_out_threshold, grad_y, 0.0)
        return grad_x
    else:
        if enable_zero_out:
            x = x.contiguous()
            grad_y = grad_y.contiguous()
            grad_x = torch.empty_like(x)
            num_elements = x.numel()

            def grid(meta):
                return (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

            with torch.cuda.device(x.device.index):
                _random_bitflip_zero_outed_backward_kernel[grid](
                    x,
                    grad_y,
                    grad_x,
                    n_elements=num_elements,
                    exp_halves=exp_halves,
                    frac_halves=frac_halves,
                    seed_exp=seed_exp,
                    seed_frac=seed_frac,
                    zero_out_threshold=zero_out_threshold,
                    SKIP_EXP_FLIP=skip_exp_flip,
                    SKIP_FRAC_FLIP=skip_frac_flip,
                    INPUT_DTYPE=TORCH_DTYPE_TO_TRITON[x.dtype],
                    BIN_DTYPE=BIT_FLIP_DTYPE_MAP[x.dtype],
                    GRAD_DTYPE=TORCH_DTYPE_TO_TRITON[grad_y.dtype],
                    EXP_PHILOX_N_ROUNDS=_get_philox_n_rounds(exp_halves),
                    FRAC_PHILOX_N_ROUNDS=_get_philox_n_rounds(frac_halves),
                )
        else:
            grad_x = grad_y.clone()
        return grad_x


@_random_bitflip_backward.register_fake
def _random_bitflip_backward_fake(
    x: Tensor,
    grad_y: Tensor,
    exp_halves: int | None,
    frac_halves: int | None,
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float | None,
) -> Tensor:
    grad_x = torch.empty_like(grad_y)
    return grad_x


def _random_bitflip_backward_wrapper(ctx, *grad_outputs):
    exp_halves = ctx.exp_halves
    frac_halves = ctx.frac_halves
    seed_exp = ctx.seed_exp
    seed_frac = ctx.seed_frac
    zero_out_threshold = ctx.zero_out_threshold

    x = ctx.saved_tensors[0]
    grad_input = _random_bitflip_backward(
        x=x,
        grad_y=grad_outputs[0],
        exp_halves=exp_halves,
        frac_halves=frac_halves,
        seed_exp=seed_exp,
        seed_frac=seed_frac,
        zero_out_threshold=zero_out_threshold,
    )
    return grad_input, None, None, None, None, None, None


def _random_bitflip_setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])  # x
    ctx.exp_halves = inputs[1]
    ctx.frac_halves = inputs[2]
    ctx.seed_exp = inputs[3]
    ctx.seed_frac = inputs[4]
    ctx.zero_out_threshold = inputs[5]


random_bitflip_fn.register_autograd(
    _random_bitflip_backward_wrapper, setup_context=_random_bitflip_setup_context
)


@random_bitflip_fn.register_kernel("cpu")
def _random_bitflip_forward_cpu(
    x: Tensor,
    exp_halves: int | None,
    frac_halves: int | None,
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float | None,
) -> tuple[Tensor, int, int]:
    # TODO: implement the CPU version of random bit flip using numpy
    raise NotImplementedError("CPU version of random bit flip is not implemented yet.")


@_random_bitflip_backward.register_kernel("cpu")
def _random_bitflip_backward_cpu(
    x: Tensor,
    grad_y: Tensor,
    exp_halves: int | None,
    frac_halves: int | None,
    seed_exp: int,
    seed_frac: int,
    zero_out_threshold: float | None,
) -> Tensor:
    # TODO: implement the CPU version of random bit flip backward using numpy
    raise NotImplementedError(
        "CPU version of random bit flip backward is not implemented yet."
    )
