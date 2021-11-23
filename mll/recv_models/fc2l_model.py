import torch
from torch import nn


class FC2LModel(nn.Module):
    supports_dropout = True

    def __init__(self, embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type, dropout):
        """
        ignores rnn_type
        """
        self.embedding_size = embedding_size
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type

        super().__init__()

        self.utts_offset = ((torch.ones(utt_len, dtype=torch.int64).cumsum(dim=-1) - 1) * (vocab_size + 1)).unsqueeze(1)
        self.embedding = nn.Embedding(utt_len * (vocab_size + 1), embedding_size)
        self.h2 = nn.Linear(embedding_size, num_meaning_types * meanings_per_type)
        self.drop = nn.Dropout(dropout)

    def forward(self, utts):
        """
        architecture:
        - input: utterances [M][N], discrete up to V + 1
        - embedding (V + 1, embedding_size)  (output is: [M][N][E])
        - drop
        - tanh
        - reshape to [N][c_len * embedding_size]
        - linear(c_len * embedding_size, n_att * n_val)
        - rehsape to [N][n_att][n_val]
        """
        batch_size = utts.size(1)

        utts_discrete_offset = utts + self.utts_offset
        embs = self.embedding(utts_discrete_offset).sum(dim=0)
        x = self.drop(embs)
        x = torch.tanh(x)
        x = self.h2(x)
        view_list = [self.num_meaning_types, self.meanings_per_type]
        x = x.view(batch_size, *view_list)
        return x
