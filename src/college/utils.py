import random
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def combine_layers(hidden_states: Sequence[torch.Tensor], layers: Iterable[int]) -> torch.Tensor:
    return torch.stack([hidden_states[i] for i in layers]).sum(0).squeeze()


def get_matching_indices(original: Sequence[int], modified: Sequence[int]) -> Tuple[List[int], List[int]]:
    corresponding_indices = []
    i = j = 0
    while i < len(original) and j < len(modified):
        if original[i] == modified[j]:
            corresponding_indices.append((i, j))
            i += 1
            j += 1
        else:
            i += 1
            if i == len(original):
                i = 0
                j += 1

    indices_in_original = [item[0] for item in corresponding_indices]
    indices_in_modified = [item[1] for item in corresponding_indices]
    if len(indices_in_original) != len(indices_in_modified):
        raise ValueError("Matching index lists must have the same length.")
    return indices_in_original, indices_in_modified


def order_and_select_indices(index_sequence: Sequence[Optional[int]]) -> List[int]:
    ordered = []
    for item in index_sequence:
        if item is None:
            continue
        if ordered and item < ordered[-1]:
            ordered.pop()
        ordered.append(item)
    return ordered
