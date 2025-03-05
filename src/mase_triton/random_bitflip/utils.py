import torch
from torch import Tensor
from ..utils.bit_repr import get_binary_repr

DTYPE_TO_WIDTH = {
    torch.float32: 32,
    torch.float16: 16,
    torch.bfloat16: 16,
}


def count_matched_bits(a: Tensor, b: Tensor):

    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert a.dtype in DTYPE_TO_WIDTH
    bitwidth = DTYPE_TO_WIDTH[a.dtype]

    a = get_binary_repr(a, split_every=None).flatten().tolist()
    b = get_binary_repr(b, split_every=None).flatten().tolist()
    bit_match_counter = {i: 0 for i in range(bitwidth)}
    for a_el, b_el in zip(a, b):
        assert len(a_el) == len(b_el) == bitwidth
        for i in range(bitwidth):
            if a_el[i] == b_el[i]:
                bit_match_counter[i] += 1

    bit_match_counter["total"] = len(a)
    return bit_match_counter


def calculate_bit_mismatch_rate(a: Tensor, b: Tensor, group: dict[str, tuple[int]] | None = None):
    default_group_map = {
        torch.float32: {"sign_exp": tuple(range(0, 9)), "frac": tuple(range(9, 32))},
        torch.float16: {"sign_exp": tuple(range(0, 6)), "frac": tuple(range(6, 16))},
        torch.bfloat16: {"sign_exp": tuple(range(0, 8)), "frac": tuple(range(8, 16))},
    }
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert a.dtype in DTYPE_TO_WIDTH
    if group is None:
        group = default_group_map[a.dtype]

    bit_match_counter = count_matched_bits(a, b)
    results = {}
    for g_name, g_ids in group.items():
        g_matched_bits = sum(bit_match_counter[i] for i in g_ids)
        g_total_bits = len(g_ids) * bit_match_counter["total"]
        g_mismatch_rate = 1 - g_matched_bits / g_total_bits
        results[g_name] = g_mismatch_rate

    return results
