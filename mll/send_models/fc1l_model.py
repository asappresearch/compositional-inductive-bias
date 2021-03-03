import torch
from torch import nn


class FC1LModel(nn.Module):
    supports_gumbel = False
    supports_dropout = False

    def __init__(self, embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type):
        self.embedding_size = embedding_size
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type
        self.utt_len = utt_len
        self.vocab_size = vocab_size

        super().__init__()
        self.h1 = nn.Linear(num_meaning_types * meanings_per_type, utt_len * (vocab_size + 1))

    def forward(self, meanings):
        """
        meanings are [N][T], index-encoded
        we'll fluff up to one-hot, then pass through a linear
        """
        N, T = meanings.size()
        meanings_onehot = torch.zeros(N, T, self.meanings_per_type, dtype=torch.float32, device=meanings.device)
        meanings_onehot.scatter_(2, meanings.unsqueeze(-1), 1)
        meanings_onehot = meanings_onehot.view(N, -1)

        utts = self.h1(meanings_onehot)
        utts = utts.view(N, self.utt_len, self.vocab_size + 1)
        utts = utts.transpose(0, 1)
        return utts
