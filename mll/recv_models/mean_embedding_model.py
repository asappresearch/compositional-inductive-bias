import torch
from torch import nn
from ulfs import nn_modules


class MeanEmbeddingsModel(nn.Module):
    supports_dropout = False

    def __init__(self, embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type):
        """
        ignores rnn_type
        """
        self.embedding_size = embedding_size
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type

        super().__init__()
        self.embedding = nn_modules.EmbeddingAdapter(vocab_size, embedding_size)
        self.h_out = nn.Linear(embedding_size, num_meaning_types * meanings_per_type)

    def forward(self, utts):
        batch_size = utts.size(1)
        embs = self.embedding(utts)
        embs_mean = embs.mean(dim=0)
        embs_mean = torch.tanh(embs_mean)
        x = self.h_out(embs_mean)
        view_list = [self.num_meaning_types, self.meanings_per_type]
        x = x.view(batch_size, *view_list)
        return x
