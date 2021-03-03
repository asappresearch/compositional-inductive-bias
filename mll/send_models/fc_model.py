import torch
from torch import nn


class FC2LModel(nn.Module):
    supports_gumbel = False
    supports_dropout = True

    def __init__(self, embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type, dropout):
        self.embedding_size = embedding_size
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type
        self.utt_len = utt_len
        self.vocab_size = vocab_size

        super().__init__()
        self.h_in = nn.Linear(num_meaning_types * meanings_per_type, embedding_size)
        self.h_out = nn.Linear(embedding_size, utt_len * (vocab_size + 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, meanings):
        """
        meanings are [N][T], index-encoded
        we'll fluff up to one-hot, then pass through a linear
        """
        N, T = meanings.size()
        meanings_onehot = torch.zeros(N, T, self.meanings_per_type, dtype=torch.float32, device=meanings.device)
        meanings_onehot.scatter_(2, meanings.unsqueeze(-1), 1)
        meanings_onehot = meanings_onehot.view(N, -1)

        embs = self.h_in(meanings_onehot)
        embs = self.drop(embs)
        embs = torch.tanh(embs)
        utts = self.h_out(embs)
        utts = utts.view(N, self.utt_len, self.vocab_size + 1)
        utts = utts.transpose(0, 1)
        return utts
