import torch
from torch import nn


class FC1LModel(nn.Module):
    supports_dropout = False

    def __init__(self, embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type):
        """
        ignores rnn_type
        """
        self.embedding_size = embedding_size
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type

        super().__init__()

        self.utts_offset = ((torch.ones(utt_len, dtype=torch.int64).cumsum(dim=-1) - 1) * (vocab_size + 1)).unsqueeze(1)
        self.embedding = nn.Embedding(utt_len * (vocab_size + 1), num_meaning_types * meanings_per_type)

    def forward(self, utts):
        """
        architecture:
        - input: utterances [M][N]
        - embedding (utt_len * (V + 1), n_att * n_val)  (output: [M][N][n_att * n_val])
        - rehsape to [N][n_att][n_val]
        """
        batch_size = utts.size(1)

        utts_discrete_offset = utts + self.utts_offset
        embs = self.embedding(utts_discrete_offset).sum(dim=0)
        view_list = [self.num_meaning_types, self.meanings_per_type]
        x = embs.view(batch_size, *view_list)
        return x
