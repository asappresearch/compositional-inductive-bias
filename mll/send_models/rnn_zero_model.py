import torch
from torch import nn

from mll.darts_cell import DGReceiver, DGSender


class RNNZeroModel(nn.Module):
    supports_gumbel = False
    supports_dropout = True
    """
    generator model
    we'll use a differentiable teacher-forcing model, with fixed utterance length
    """
    def __init__(
            self, embedding_size: int, vocab_size: int, utt_len: int, num_meaning_types: int, meanings_per_type: int,
            rnn_type: str, dropout: float, num_layers: int = 1
    ):
        self.embedding_size = embedding_size
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type
        self.utt_len = utt_len
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        super().__init__()
        """
        so, we'll make the meanings one-hot, then pass through a Linear?

        """
        print('rnn dropout', dropout)
        # self.h_in = nn.Linear(num_meaning_types * meanings_per_type, embedding_size)
        if rnn_type == 'SRU':
            from sru import SRU
            RNN = SRU
        elif rnn_type == 'dgrecv':
            RNN = DGReceiver
        elif rnn_type == 'dgsend':
            RNN = DGSender
        else:
            RNN = getattr(nn, f'{rnn_type}')
        rnn_params = {
            'input_size': embedding_size,
            'hidden_size': embedding_size,
            'num_layers': num_layers,
            'dropout': dropout
        }
        if rnn_type == 'SRU':
            rnn_params['rescale'] = False
            rnn_params['use_tanh'] = True
        self.rnn = RNN(**rnn_params)
        self.h_out = nn.Linear(embedding_size, vocab_size + 1)
        self.drop = nn.Dropout(dropout)

        self.meanings_offset = (torch.ones(num_meaning_types, dtype=torch.int64).cumsum(dim=-1) - 1) * meanings_per_type
        self.embeddings = nn.Embedding(num_meaning_types * meanings_per_type, embedding_size)

    def forward(self, meanings):
        """
        meanings are [N][T], index-encoded
        """
        N = meanings.size(0)
        meanings_discrete_offset = meanings + self.meanings_offset
        embs = self.embeddings(meanings_discrete_offset).sum(dim=1)

        embs = self.drop(embs)
        # we'll use this as the state for a generator now

        if self.rnn_type in ['LSTM']:
            h = torch.zeros(self.num_layers, N, self.embedding_size, dtype=torch.float32, device=meanings.device)
            h[0] = embs
            c = torch.zeros(self.num_layers, N, self.embedding_size, dtype=torch.float32, device=meanings.device)
        elif self.rnn_type in ['SRU']:
            h = torch.zeros(self.num_layers, N, self.embedding_size, dtype=torch.float32, device=meanings.device)
            h[0] = embs
        elif self.rnn_type in ['GRU', 'RNN', 'dgsend', 'dgrecv']:
            h = torch.zeros(self.num_layers, N, self.embedding_size, dtype=torch.float32, device=meanings.device)
            h[0] = embs
        else:
            raise Exception(f'unrecognized rnn type {self.rnn_type}')

        if self.rnn_type in ['LSTM']:
            fake_input = torch.zeros(self.utt_len, N, self.embedding_size, dtype=torch.float32, device=meanings.device)
        else:
            fake_input = torch.zeros(self.utt_len, N, self.embedding_size, dtype=torch.float32, device=meanings.device)

        if self.rnn_type == 'SRU':
            output, state = self.rnn(fake_input, h)
        elif self.rnn_type == 'LSTM':
            h, c = self.rnn(fake_input, (h, c))
            output = h
        elif self.rnn_type in ['GRU', 'RNN', 'dgsend', 'dgrecv']:
            output, h = self.rnn(fake_input, h)
        else:
            raise Exception(f'rnn type {self.rnn_type} not recognized')
        utts = self.h_out(output)
        return utts  # [utt_len, N, vocab_size]


class RNNZero2LModel(RNNZeroModel):
    supports_gumbel = False
    supports_dropout = True
    """
    generator model
    we'll use a differentiable teacher-forcing model, with fixed utterance length
    """
    def __init__(
            self, embedding_size: int, vocab_size: int, utt_len: int, num_meaning_types: int, meanings_per_type: int,
            rnn_type: str, dropout: float
    ):
        super().__init__(
            embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type, rnn_type, num_layers=2,
            dropout=dropout)


class RNNZero3LModel(RNNZeroModel):
    supports_gumbel = False
    supports_dropout = True
    """
    generator model
    we'll use a differentiable teacher-forcing model, with fixed utterance length
    """
    def __init__(
            self, embedding_size: int, vocab_size: int, utt_len: int, num_meaning_types: int, meanings_per_type: int,
            rnn_type: str, dropout: float
    ):
        super().__init__(
            embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type, rnn_type, num_layers=3,
            dropout=dropout)
