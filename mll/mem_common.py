import torch
import numpy as np
import math
import random
import argparse

from ulfs import metrics

from mll import corruptions_lib


# from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adjust_render_time(render_every_seconds, elapsed_time):
    thresholds = {
        0.01: 0.1,
        0.03: 0.3,
        0.1: 1,
        0.3: 3,
        1: 10,
        3: 30,
        10: 100,
        30: None
    }
    threshold = thresholds[render_every_seconds]
    if threshold is None:
        return render_every_seconds
    if elapsed_time >= threshold:
        render_list = sorted(thresholds.keys())
        render_every_seconds = render_list[render_list.index(render_every_seconds) + 1]
        print('render every seconds', render_every_seconds)
    return render_every_seconds


def inc_meaning(meaning, base):
    j = meaning.size(0) - 1
    while True:
        meaning[j] += 1
        if meaning[j] < base:
            return True
        meaning[j] = 0
        j -= 1
        if j < 0:
            return False


def get_corruption(
        corruption_name: str, vocab_size: int, meanings_per_type: int, num_meaning_types: int, tokens_per_meaning: int,
        r: np.random.RandomState = None):
    Corruption = getattr(corruptions_lib, f'{corruption_name}Corruption')
    print('r', r)
    corruption_params = {
        'num_meaning_types': num_meaning_types,
        'tokens_per_meaning': tokens_per_meaning,
        'vocab_size': vocab_size,
        'r': r
    }
    if corruption_name in ['ShuffleWordsDet']:
        corruption_params['meanings_per_type'] = meanings_per_type
    # print('utterances_by_meaning', self.utterances_by_meaning)
    corruption = Corruption(**corruption_params)
    return corruption


class CompositionalGrammar(object):
    def __init__(self, num_meaning_types, tokens_per_meaning, meanings_per_type, vocab_size, corruptions):
        """
        so, we have to generate the grammar
        for each meaning type, we'll pick tokens_per_meaning
        random tokens, drawn from vocab_size possible tokens
        """
        self.id = random.randint(0, 100000)  # for debugging...
        self.num_meaning_types = num_meaning_types
        self.tokens_per_meaning = tokens_per_meaning
        self.utt_len = num_meaning_types * tokens_per_meaning
        self.meanings_per_type = meanings_per_type
        self.subuts_by_meaning_by_type = torch.zeros(
            num_meaning_types, meanings_per_type, tokens_per_meaning, dtype=torch.int64)
        num_meaning_combos = int(math.pow(vocab_size, tokens_per_meaning))
        all_meaning_idxes = torch.from_numpy(
            np.random.choice(num_meaning_combos, num_meaning_types * meanings_per_type, replace=False))
        for meaning_type in range(num_meaning_types):
            meaning_idxes = all_meaning_idxes[meaning_type * meanings_per_type: (meaning_type + 1) * meanings_per_type]
            for j in range(tokens_per_meaning):
                token = meaning_idxes % vocab_size
                meaning_idxes = meaning_idxes // vocab_size
                self.subuts_by_meaning_by_type[meaning_type, :, j] = token

        # lets flatten them now ...
        self.tot_examples = int(math.pow(meanings_per_type, num_meaning_types))
        self.utterances_by_meaning = torch.zeros(
            self.tot_examples,
            self.utt_len,
            dtype=torch.int64
        )
        # just do this stupidly, unless it turns out to be slow...

        def meaning_to_utterance(meaning):
            utt = torch.zeros(self.utt_len, dtype=torch.int64)
            for meaning_type in range(num_meaning_types):
                m_start = meaning_type * self.tokens_per_meaning
                m_end = m_start + self.tokens_per_meaning
                utt[m_start:m_end] = self.subuts_by_meaning_by_type[meaning_type, meaning[meaning_type]]
            return utt

        j = 0
        self.meanings = torch.zeros(self.utterances_by_meaning.size(0), num_meaning_types, dtype=torch.int64)
        meaning = torch.zeros(num_meaning_types, dtype=torch.int64)
        while True:
            self.meanings[j] = meaning
            self.utterances_by_meaning[j] = meaning_to_utterance(meaning)
            if not inc_meaning(meaning=meaning, base=meanings_per_type):
                break
            j += 1
        if corruptions is not None:
            for corruption_name in corruptions.split(','):
                corruption = get_corruption(
                    corruption_name=corruption_name, vocab_size=vocab_size,
                    meanings_per_type=meanings_per_type,
                    num_meaning_types=self.num_meaning_types,
                    tokens_per_meaning=self.tokens_per_meaning
                )
                self.utterances_by_meaning = corruption(self.utterances_by_meaning)
                print(f'ran corruption {corruption_name}')
                assert self.utterances_by_meaning is not None
        print('created compositional grammar')
        self.rho = metrics.topographic_similarity(
            utts=self.utterances_by_meaning, labels=self.meanings)
        # print('rho %.3f' % self.rho)

    def meanings_to_utterances(self, meanings):
        """
        meanings is [batch-size][num-meaning-types]
                    [N][T]
        eg batch size might be 32 or 128, and num meaning types might be: 5

        output will be [M][N]
        where N is batch size, and
              M is tokens_per_meaning * num_meaning_types
        """
        # print('instance', self.id)
        N, T = meanings.size()
        assert T == self.num_meaning_types
        meaning_idxes = torch.zeros(N, dtype=torch.int64)
        for t in range(T):
            meaning_idxes *= self.meanings_per_type
            meaning_idxes += meanings[:, t]
        res = self.utterances_by_meaning[meaning_idxes].transpose(0, 1)
        return res


class HolisticGrammar(object):
    def __init__(self, num_meaning_types, tokens_per_meaning, meanings_per_type, vocab_size, corruptions):
        """
        corruptions is ignored
        """
        self.id = random.randint(0, 100000)  # for debugging...
        self.meanings_per_type = meanings_per_type
        self.num_meaning_types = num_meaning_types
        self.utt_len = num_meaning_types * tokens_per_meaning
        num_utterances = int(math.pow(vocab_size, self.utt_len))
        num_meanings = int(math.pow(meanings_per_type, num_meaning_types))
        self.utterances_by_meaning = torch.zeros(
            num_meanings,
            self.utt_len,
            dtype=torch.int64
        )
        # note: putting replace=False causes massive memory allocation :P dont do that :P
        replace = True
        if num_utterances / num_meanings < 1000:
            print('using replace False')
            replace = False
        meaning_idxes = torch.from_numpy(np.random.choice(num_utterances, num_meanings, replace=replace))
        for t in range(self.utt_len):
            token = meaning_idxes % vocab_size
            meaning_idxes = meaning_idxes // vocab_size
            self.utterances_by_meaning[:, t] = token
        print('created holistic grammar')
        j = 0
        self.meanings = torch.zeros(self.utterances_by_meaning.size(0), num_meaning_types, dtype=torch.int64)
        meaning = torch.zeros(num_meaning_types, dtype=torch.int64)
        while True:
            self.meanings[j] = meaning
            if not inc_meaning(meaning=meaning, base=meanings_per_type):
                break
            j += 1
        self.rho = metrics.topographic_similarity(
            utts=self.utterances_by_meaning, labels=self.meanings)
        # print('rho %.3f' % self.rho)

    def meanings_to_utterances(self, meanings):
        return CompositionalGrammar.meanings_to_utterances(self=self, meanings=meanings)


def run(args):
    Grammar = globals()[f'{args.grammar}Grammar']
    grammar = Grammar(
        num_meaning_types=args.num_meaning_types,
        tokens_per_meaning=args.tokens_per_meaning,
        meanings_per_type=args.meanings_per_type,
        vocab_size=args.vocab_size,
        corruptions=args.corruptions
    )
    print('utts:')
    print(grammar.utterances_by_meaning[:20])
    print('meanings:')
    print(grammar.meanings[:20])
    # it = 0
    # while True:
    #     meanings = torch.from_numpy(
    #         np.random.choice(args.meanings_per_type, (args.batch_size, args.num_meaning_types), replace=True))
    #     grammar.meanings_to_utterances(meanings)
    # print('it done', it)


if __name__ == '__main__':
    # test grammar somehow a bit
    parser = argparse.ArgumentParser()
    parser.add_argument('--grammar', type=str, choices=['Compositional', 'Holistic'], default='Compositional')
    parser.add_argument('--meanings', type=str, default='5x10')
    parser.add_argument('--corruptions', type=str)
    parser.add_argument('--vocab-size', type=int, default=4)
    parser.add_argument('--tokens-per-meaning', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    args.num_meaning_types, args.meanings_per_type = [
        int(v) for v in args.meanings.split('x')]
    del args.__dict__['meanings']
    run(args)
