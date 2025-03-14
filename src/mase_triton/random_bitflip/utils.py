import concurrent.futures
import math

import torch
from torch import Tensor
from ..utils.bit_repr import get_binary_repr

DTYPE_TO_WIDTH = {
    torch.float32: 32,
    torch.float16: 16,
    torch.bfloat16: 16,
}


def _count_matched_bits_worker(a: list[str], b: list[str], bitwidth: int):
    bit_match_counter = {i: 0 for i in range(bitwidth)}
    for a_el, b_el in zip(a, b):
        assert len(a_el) == len(b_el) == bitwidth
        for i in range(bitwidth):
            if a_el[i] == b_el[i]:
                bit_match_counter[i] += 1
    return bit_match_counter


def count_matched_bits(a: Tensor, b: Tensor, num_workers: int | None = None):

    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert a.dtype in DTYPE_TO_WIDTH
    bitwidth = DTYPE_TO_WIDTH[a.dtype]

    a = get_binary_repr(a, split_every=None).flatten().tolist()
    b = get_binary_repr(b, split_every=None).flatten().tolist()

    if num_workers is None:
        bit_match_counter = {i: 0 for i in range(bitwidth)}
        for a_el, b_el in zip(a, b):
            assert len(a_el) == len(b_el) == bitwidth
            for i in range(bitwidth):
                if a_el[i] == b_el[i]:
                    bit_match_counter[i] += 1

        bit_match_counter["total"] = len(a)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            chunk_size = math.ceil(len(a) / num_workers)
            # submit a chunk of data to each worker
            for i in range(0, len(a), chunk_size):
                futures.append(
                    executor.submit(_count_matched_bits_worker, a[i : i + chunk_size], b[i : i + chunk_size], bitwidth)
                )

            bit_match_counter = {i: 0 for i in range(bitwidth)}
            for future in futures:
                bit_match_counter_worker = future.result()
                for i in range(bitwidth):
                    bit_match_counter[i] += bit_match_counter_worker[i]
        bit_match_counter["total"] = len(a)
    return bit_match_counter


def calculate_bit_mismatch_rate(
    a: Tensor, b: Tensor, group: dict[str, tuple[int]] | None = None, num_workers: int | None = None
):
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

    bit_match_counter = count_matched_bits(a, b, num_workers=num_workers)
    results = {}
    for g_name, g_ids in group.items():
        g_matched_bits = sum(bit_match_counter[i] for i in g_ids)
        g_total_bits = len(g_ids) * bit_match_counter["total"]
        g_mismatch_rate = 1 - g_matched_bits / g_total_bits
        results[g_name] = g_mismatch_rate

    return results
