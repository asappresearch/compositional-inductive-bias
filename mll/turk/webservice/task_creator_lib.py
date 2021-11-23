import random
import uuid
import time
from typing import Tuple

import numpy as np
import torch

from ulfs import tensor_utils

from mll import mem_common


def get_unique_string():
    return uuid.uuid4().hex


def get_grammar(grammar: str, seed: int, num_meaning_types=1, tokens_per_meaning=2, meanings_per_type=5, vocab_size=26):
    start_time = time.time()
    corruption = None
    grammar_family = grammar
    if grammar not in ['Compositional', 'Holistic']:
        corruption = grammar
        grammar_family = 'Compositional'
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    Grammar = getattr(mem_common, f'{grammar_family}Grammar')
    grammar_object = Grammar(
        num_meaning_types=num_meaning_types,
        tokens_per_meaning=tokens_per_meaning,
        meanings_per_type=meanings_per_type,
        vocab_size=vocab_size,
        corruptions=corruption
    )
    elapsed = int(time.time() - start_time)
    print(f'created grammar in {elapsed}s')
    return grammar_object


def utts_to_texts(utts: torch.Tensor):
    texts = []
    N = utts.size(0)
    for n in range(N):
        text = tensor_utils.tensor_to_str(utts[n] + 1)
        texts.append(text)
    return texts


def perturb_color(color: Tuple[int, int, int], amount: int):
    d_color = (random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10))
    color = tuple(
        max(0, min(255, c + d))
        for c, d in zip(color, d_color)
    )
    return color
