import torch
from torch import nn
from ulfs import nn_modules


class FC3LModel(nn.Module):
    supports_dropout = True

    def __init__(self, embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type, dropout):
        """
        ignores rnn_type
        """
        self.embedding_size = embedding_size
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type

        super().__init__()
        self.embedding = nn_modules.EmbeddingAdapter(vocab_size + 1, embedding_size)
        self.translater = nn.Linear(utt_len * embedding_size, embedding_size)
        self.h_out = nn.Linear(embedding_size, num_meaning_types * meanings_per_type)
        self.drop = nn.Dropout(dropout)

    def forward(self, utts):
        """
        3 layers (embedding, linear, linear)

        architecture:
        - input: utterances
        - embedding(V + 1, embedding_size)
        - drop
        - tanh
        - reshape to [N][c_len * embedding_size]
        - linear(c_len * embedding_size, embedding_size)
        - drop
        - tanh
        - linear(embedding_size, n_att * n_val)
        - reshape to [N][n_att][n_val]
        """
        batch_size = utts.size(1)
        embs = self.embedding(utts)
        embs = self.drop(embs)
        embs = torch.tanh(embs)
        embs = embs.transpose(0, 1).contiguous()
        embs = embs.view(batch_size, -1)
        state = self.translater(embs)
        state = self.drop(state)
        state = torch.tanh(state)
        x = self.h_out(state)
        view_list = [self.num_meaning_types, self.meanings_per_type]
        x = x.view(batch_size, *view_list)
        return x
