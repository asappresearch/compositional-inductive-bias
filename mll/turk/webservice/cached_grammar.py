import json
import os
from os.path import join

import numpy as np

from mll.turk.webservice import task_creator_lib


class CachedGrammar:
    def __init__(
        self, seed: int, grammar: str, num_examples: int,
        num_meaning_types: int, meanings_per_type: int, vocab_size: int
    ):
        language_dir = (
            f'data/languages/g{grammar}_s{seed}_n{num_examples}_'
            f'{num_meaning_types}x{meanings_per_type}_v{vocab_size}')
        if not os.path.isdir(language_dir):
            os.makedirs(language_dir)
        if not os.path.isfile(join(language_dir, 'utts.txt')):
            language = task_creator_lib.get_grammar(
                grammar=grammar,
                seed=seed,
                num_meaning_types=num_meaning_types,
                meanings_per_type=meanings_per_type,
                vocab_size=vocab_size
            )
            utts = task_creator_lib.utts_to_texts(language.utterances_by_meaning)
            meanings = language.meanings
            N = len(utts)
            if num_examples < N:
                print(f'sampling {num_examples} examples from {N} examples')
                order = np.random.choice(N, num_examples, replace=False)
                meanings = meanings[order]
                utts = [utts[idx] for idx in order]
            else:
                print('num_examples < N, so no need for sampling')
            with open(join(language_dir, 'utts.txt'), 'w') as f:
                for utt in utts:
                    f.write(utt + '\n')
            with open(join(language_dir, 'meanings.json'), 'w') as f:
                f.write(json.dumps(meanings.tolist()))
        with open(join(language_dir, 'utts.txt'), 'r') as f:
            self.utts = [line[:-1] for line in f]
        with open(join(language_dir, 'meanings.json'), 'r') as f:
            self.meanings = json.load(f)
        self.num_pairs = len(self.utts)

    def get_meaning_utt(self, idx):
        assert idx >= 0 and idx < self.num_pairs
        return {'meaning': self.meanings[idx], 'utt': self.utts[idx]}
