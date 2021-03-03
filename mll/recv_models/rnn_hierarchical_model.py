import torch
from torch import nn
from ulfs import nn_modules

from mll.darts_cell import DGReceiverCell, DGSenderCell


class RNNHierEncoder(nn.Module):
    def __init__(
            self, embedding_size, rnn_type
    ):
        self.rnn_type = rnn_type
        self.embedding_size = embedding_size

        super().__init__()
        # assert rnn_type in ['GRU', 'LSTM', 'RNN', 'SRU']
        # assert rnn_type not in ['LSTM']
        if rnn_type in ['SRU']:
            from sru import SRUCell
            RNNCell = SRUCell
        elif rnn_type == 'dgrecv':
            RNNCell = DGReceiverCell
        elif rnn_type == 'dgsend':
            RNNCell = DGSenderCell
        else:
            RNNCell = getattr(nn, f'{rnn_type}Cell')
        self.rnn_upper = RNNCell(
            input_size=embedding_size,
            hidden_size=embedding_size
        )
        self.rnn_lower = RNNCell(
            input_size=embedding_size,
            hidden_size=embedding_size
        )

        self.linear_lower_stop = nn.Linear(embedding_size, 1)

    def forward(self, inputs, return_stopness=False):
        seq_len, batch_size, embedding_size = inputs.size()

        device = inputs.device
        N = batch_size
        if self.rnn_type in ['LSTM']:
            lower_state = [
                torch.zeros(N, self.embedding_size, dtype=torch.float32, device=device),
                torch.zeros(N, self.embedding_size, dtype=torch.float32, device=device)
            ]
            upper_state = [
                torch.zeros(N, self.embedding_size, dtype=torch.float32, device=device),
                torch.zeros(N, self.embedding_size, dtype=torch.float32, device=device)
            ]
        else:
            lower_state = torch.zeros(N, self.embedding_size, dtype=torch.float32, device=device)
            upper_state = torch.zeros(N, self.embedding_size, dtype=torch.float32, device=device)
        lower_stopness = torch.ones(N, 1, dtype=torch.float32, device=device)
        all_lower_stopness = torch.zeros(seq_len, N, dtype=torch.float32, device=device)
        for m in range(seq_len):
            if self.rnn_type in ['LSTM']:
                lower_state = list(lower_state)
                lower_state[0] = lower_state[0] * (1 - lower_stopness)
                lower_state[1] = lower_state[1] * (1 - lower_stopness)
            else:
                lower_state = lower_state * (1 - lower_stopness)
            in_token = inputs[m]
            lower_state = self.rnn_lower(in_token, lower_state)
            lower_stopness = torch.sigmoid(self.linear_lower_stop(lower_state[0]))
            all_lower_stopness[m] = lower_stopness.squeeze(-1)
            if self.rnn_type in ['LSTM']:
                input_to_upper = lower_state[0]
            else:
                input_to_upper = lower_state
            new_upper_state = self.rnn_upper(input_to_upper, upper_state)
            if self.rnn_type in ['LSTM']:
                new_upper_state = list(new_upper_state)
                upper_state[0] = lower_stopness * new_upper_state[0] + (1 - lower_stopness) * upper_state[0]
                upper_state[1] = lower_stopness * new_upper_state[1] + (1 - lower_stopness) * upper_state[1]
            else:
                upper_state = lower_stopness * new_upper_state + (1 - lower_stopness) * upper_state

        if self.rnn_type in ['LSTM']:
            state = upper_state[0]
        else:
            state = upper_state

        if return_stopness:
            return state, all_lower_stopness
        else:
            return state


class HierModel(nn.Module):
    supports_dropout = True

    def __init__(
            self, embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type,
            rnn_type, dropout
    ):
        self.rnn_type = rnn_type
        self.embedding_size = embedding_size
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type

        super().__init__()
        self.embedding = nn_modules.EmbeddingAdapter(vocab_size + 1, embedding_size)
        self.rnn_hierarchical = RNNHierEncoder(
            embedding_size=embedding_size,
            rnn_type=rnn_type
        )
        self.linear_out = nn.Linear(embedding_size, num_meaning_types * meanings_per_type)
        self.drop = nn.Dropout(dropout)

    def forward(self, utts):
        embs = self.embedding(utts)
        M, N, E = embs.size()
        embs = self.drop(embs)

        state, self.all_lower_stopness = self.rnn_hierarchical(inputs=embs, return_stopness=True)
        x = self.linear_out(self.drop(state))
        view_list = [self.num_meaning_types, self.meanings_per_type]
        x = x.view(N, *view_list)
        return x
