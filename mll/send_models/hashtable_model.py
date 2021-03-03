import torch
from torch import nn


class HashtableModel(nn.Module):
    """
    this converts utterances into an int, so we can store into a hashtable
    """
    supports_gumbel = False
    supports_dropout = False

    def __init__(self, embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type):
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type
        self.vocab_size = vocab_size
        self.utt_len = utt_len

        super().__init__()
        self.utt_by_meaning = {}

    def _meanings_t_to_l(self, meanings_t):
        """
        assumes utts_t is [M][N]
        """
        N, T = meanings_t.size()
        meanings_cat = torch.zeros(N, dtype=torch.int64)
        for t in range(T):
            meanings_cat = meanings_cat * self.meanings_per_type
            meanings_cat = meanings_cat + meanings_t[:, t]
        meanings_l = meanings_cat.tolist()
        return meanings_l

    def forward(self, meanings):
        N, T = meanings.size()
        meanings_l = self._meanings_t_to_l(meanings_t=meanings)
        utts = torch.zeros(self.utt_len, N, dtype=torch.int64, device=meanings.device)
        for n, meaning in enumerate(meanings_l):
            utt = self.utt_by_meaning.get(meaning, None)
            if utt is not None:
                utts[:, n] = utt

        utts_onehot = torch.zeros(self.utt_len, N, self.vocab_size + 1, dtype=torch.float32, device=meanings.device)
        utts_onehot.scatter_(2, utts.unsqueeze(-1), 1)
        return utts_onehot

    def train(self, utts, meanings):
        meanings_l = self._meanings_t_to_l(meanings_t=meanings)
        for i, meaning in enumerate(meanings_l):
            if meaning not in self.utt_by_meaning:
                self.utt_by_meaning[meaning] = utts[:, i]
