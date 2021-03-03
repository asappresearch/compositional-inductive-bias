import math
import torch
from torch import nn


class KNNModel(nn.Module):
    supports_dropout = False

    def __init__(self, embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type):
        """
        ignores rnn_type
        """
        super().__init__()
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type
        self.vocab_size = vocab_size
        self.utt_len = utt_len

        self.K = 1
        self.meaning_space_size = int(math.pow(meanings_per_type, num_meaning_types))
        print('meaning space size', self.meaning_space_size)
        self.seen_utts = set()
        self.utts_t = torch.zeros(self.meaning_space_size, utt_len * vocab_size, dtype=torch.float32)
        self.meanings_t = torch.zeros(self.meaning_space_size, num_meaning_types, dtype=torch.int64)

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
        M, N = utts.size()
        support_size = len(self.seen_utts)
        if support_size > 0:
            batch = utts.transpose(0, 1).view(N, -1)
            batch_fluffy = torch.zeros(N, M, self.vocab_size, dtype=torch.float32)
            batch_fluffy.scatter_(2, batch.unsqueeze(-1), 1)
            batch_fluffy = batch_fluffy.view(N, M * self.vocab_size)

            batch_squared = (batch_fluffy * batch_fluffy).sum(dim=1)
            support = self.utts_t[:len(self.seen_utts)]
            support_squared = (support * support).sum(dim=1)
            transpose = batch_fluffy @ support.transpose(0, 1)
            squared_dist = batch_squared.unsqueeze(1) + support_squared.unsqueeze(0) - 2 * transpose
            _, top_k = squared_dist.topk(k=self.K, largest=False)
            # lets just use K = 1 for now....
            meanings = self.meanings_t[top_k].squeeze(1)
        else:
            meanings = torch.zeros(batch_size, self.num_meaning_types, dtype=torch.int64)

        N, T = meanings.size()
        meanings_onehot = torch.zeros(N, T, self.meanings_per_type, dtype=torch.float32)
        meanings_onehot.scatter_(2, meanings.unsqueeze(-1), 1)
        return meanings_onehot

    def run_training(self, utts, meanings):
        M, N = utts.size()
        N_, _ = meanings.size()
        assert N_ == N
        utts_l = self._utts_t_to_l(utts_t=utts)
        for n, utt_int in enumerate(utts_l):
            if utt_int in self.seen_utts:
                continue
            support_n = len(self.seen_utts)
            self.utts_t[support_n].view(self.utt_len, self.vocab_size).scatter_(1, utts[:, n].unsqueeze(-1), 1)
            self.meanings_t[support_n] = meanings[n]
            self.seen_utts.add(utt_int)
