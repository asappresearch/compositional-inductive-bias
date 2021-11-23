import torch
import numpy as np


class RandomProjCorruption(object):
    def __init__(self, num_meaning_types, tokens_per_meaning, vocab_size, r: np.random.RandomState = None):
        # DONE
        if r is None:
            r = np.random.RandomState()
        self.r = r
        self.num_meaning_types = num_meaning_types
        self.tokens_per_meaning = tokens_per_meaning
        self.vocab_size = vocab_size

    def __call__(self, utterances_by_meaning):
        tot_examples, utt_len = utterances_by_meaning.size()
        utts_onehot = torch.zeros(tot_examples, utt_len, self.vocab_size, dtype=torch.float32)
        utts_onehot.scatter_(-1, utterances_by_meaning.unsqueeze(-1), 1)
        print('r', self.r)
        proj = torch.from_numpy(self.r.rand(utt_len * self.vocab_size, utt_len * self.vocab_size)).float()
        utts_onehot_v = utts_onehot.view(tot_examples, -1) @ proj
        utts_onehot = utts_onehot_v.view(tot_examples, utt_len, self.vocab_size)
        _, utterances_by_meaning = utts_onehot.max(dim=-1)
        return utterances_by_meaning


class WordPairSumsCorruption(object):
    def __init__(self, num_meaning_types, tokens_per_meaning, vocab_size, r: np.random.RandomState = None):
        # determinstic
        if r is None:
            r = np.random.RandomState()
        self.r = r
        self.num_meaning_types = num_meaning_types
        self.tokens_per_meaning = tokens_per_meaning
        self.vocab_size = vocab_size

    def __call__(self, utterances_by_meaning):
        for t in range(self.num_meaning_types):
            w1_idx = t
            w2_idx = (t + 1) % self.num_meaning_types
            w1_start = w1_idx * self.tokens_per_meaning
            w1_end_excl = w1_start + self.tokens_per_meaning
            w2_start = w2_idx * self.tokens_per_meaning
            w2_end_excl = w2_start + self.tokens_per_meaning
            utterances_by_meaning[:, w1_start:w1_end_excl] = utterances_by_meaning[
                :, w1_start:w1_end_excl] + utterances_by_meaning[:, w2_start:w2_end_excl]
            utterances_by_meaning = utterances_by_meaning % self.vocab_size
        return utterances_by_meaning


class ZeroCorruption(object):
    def __init__(self, num_meaning_types, tokens_per_meaning, vocab_size):
        pass

    def __call__(self, utterances_by_meaning):
        utterances_by_meaning.fill_(0)
        return utterances_by_meaning


class OneCorruption(object):
    def __init__(self, num_meaning_types, tokens_per_meaning, vocab_size):
        pass

    def __call__(self, utterances_by_meaning):
        utterances_by_meaning.fill_(1)
        return utterances_by_meaning


def verify_bijective(self, utterances_by_meaning):
    print('verifying is bijective map')
    tot_examples, utt_len = utterances_by_meaning.size()
    utterances_set = set()
    utterances_int = torch.zeros(tot_examples, dtype=torch.int64)
    for t in range(utt_len):
        assert self.tokens_per_meaning < 10
        utterances_int = utterances_int * 10
        utterances_int = utterances_int + utterances_by_meaning[:, t]
    duplicates = 0
    for n in range(tot_examples):
        i = utterances_int[n].item()
        # if n < 5:
        #     print(n, i)
        if i not in utterances_set:
            utterances_set.add(i)
        else:
            # print(i)
            duplicates += 1
    if duplicates > 0:
        print('duplicates', duplicates, 'out of', tot_examples)
    print('... verification complete')


class ShuffleWordsCorruption(object):
    def __init__(self, num_meaning_types, tokens_per_meaning, vocab_size, r: np.random.RandomState = None):
        # DONE
        if r is None:
            r = np.random.RandomState()
        self.r = r
        self.num_meaning_types = num_meaning_types
        self.tokens_per_meaning = tokens_per_meaning

    def __call__(self, utterances_by_meaning):
        tot_examples, utt_len = utterances_by_meaning.size()
        view = utterances_by_meaning.view(tot_examples, self.num_meaning_types, self.tokens_per_meaning)
        for n in range(tot_examples):
            idxes = torch.from_numpy(self.r.choice(self.num_meaning_types, self.num_meaning_types, replace=False))
            view[n] = view[n, idxes]
        verify_bijective(self, utterances_by_meaning)
        return utterances_by_meaning


class ShuffleWordsDetCorruption(object):
    def __init__(
            self, num_meaning_types, tokens_per_meaning, vocab_size, meanings_per_type,
            r: np.random.RandomState = None):
        # DONE
        if r is None:
            r = np.random.RandomState()
        self.r = r
        self.num_meaning_types = num_meaning_types
        self.tokens_per_meaning = tokens_per_meaning
        self.meanings_per_type = meanings_per_type

        # for each meaning of the first meaning type, there is an ordering over all the meaning types:
        self.ordering_by_last_meaning = torch.zeros(meanings_per_type, num_meaning_types, dtype=torch.int64)
        for i in range(meanings_per_type):
            self.ordering_by_last_meaning[i] = torch.from_numpy(
                r.choice(self.num_meaning_types, self.num_meaning_types, replace=False))

    def __call__(self, utterances_by_meaning):
        tot_examples, utt_len = utterances_by_meaning.size()
        view = utterances_by_meaning.view(tot_examples, self.num_meaning_types, self.tokens_per_meaning)
        last_meanings = torch.from_numpy(np.arange(0, tot_examples)) % self.meanings_per_type
        orderings = self.ordering_by_last_meaning[last_meanings]
        for n in range(tot_examples):
            view[n] = view[n, orderings[n]]

        verify_bijective(self, utterances_by_meaning)
        return utterances_by_meaning


class PermuteCorruption(object):
    def __init__(self, num_meaning_types, tokens_per_meaning, vocab_size, r: np.random.RandomState = None):
        # DONE
        if r is None:
            r = np.random.RandomState()
        self.r = r
        pass

    def __call__(self, utterances_by_meaning):
        tot_examples, utt_len = utterances_by_meaning.size()
        self.idxes = torch.from_numpy(self.r.choice(utt_len, utt_len, replace=False))
        print('permute', self.idxes)
        utterances_by_meaning = utterances_by_meaning[:, self.idxes]
        return utterances_by_meaning


class CumrotCorruption(object):
    def __init__(self, num_meaning_types, tokens_per_meaning, vocab_size, r: np.random.RandomState = None):
        # DETERMIISTC
        if r is None:
            r = np.random.RandomState()
        self.r = r
        self.num_meaning_types = num_meaning_types
        self.tokens_per_meaning = tokens_per_meaning
        self.vocab_size = vocab_size

    def __call__(self, utterances_by_meaning):
        tot_examples, utt_len = utterances_by_meaning.size()
        for t in range(utt_len - 1, -1, -1):
            if t == 0:
                continue
            utterances_by_meaning[:, t] = utterances_by_meaning[:, :t + 1].sum(dim=1) % self.vocab_size
        return utterances_by_meaning
