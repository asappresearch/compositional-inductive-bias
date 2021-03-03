import torch
from torch import nn


class HashtableModel(nn.Module):
    supports_dropout = False

    """
    this converts utterances into an int, so we can store into a hashtable
    """
    def __init__(self, embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type):
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type
        self.vocab_size = vocab_size

        super().__init__()
        self.meaning_by_utt = {}

    def _utts_t_to_l(self, utts_t):
        """
        assumes utts_t is [M][N]
        """
        seq_len, batch_size = utts_t.size()
        utts_cat = torch.zeros(batch_size, dtype=torch.int64)
        for t in range(seq_len):
            utts_cat = utts_cat * self.vocab_size
            utts_cat = utts_cat + utts_t[t]
        utts_l = utts_cat.tolist()
        return utts_l

    def forward(self, utts):
        seq_len, batch_size = utts.size()
        utts_l = self._utts_t_to_l(utts_t=utts)
        meanings = torch.zeros(batch_size, self.num_meaning_types, dtype=torch.int64)
        for n, utt in enumerate(utts_l):
            meaning = self.meaning_by_utt.get(utt, None)
            if meaning is not None:
                meanings[n] = meaning

        N, T = meanings.size()
        meanings_onehot = torch.zeros(N, T, self.meanings_per_type, dtype=torch.float32)
        meanings_onehot.scatter_(2, meanings.unsqueeze(-1), 1)
        return meanings_onehot

    def run_training(self, utts, meanings):
        utts_l = self._utts_t_to_l(utts_t=utts)
        for i, utt in enumerate(utts_l):
            if utt not in self.meaning_by_utt:
                self.meaning_by_utt[utt] = meanings[i]
