from torch import nn
from ulfs import nn_modules

from mll.darts_cell import DGReceiver, DGSender


class RNNModel(nn.Module):
    supports_dropout = True

    def __init__(
        self, embedding_size: int, vocab_size: int, utt_len: int, num_meaning_types: int, meanings_per_type: int,
        rnn_type: str, dropout: float, num_layers: int = 1
    ):
        self.rnn_type = rnn_type
        self.embedding_size = embedding_size
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type
        self.vocab_size = vocab_size

        super().__init__()
        self.embedding = nn_modules.EmbeddingAdapter(vocab_size + 1, embedding_size)
        if rnn_type == 'SRU':
            from sru import SRU
            RNN = SRU
        elif rnn_type == 'dgrecv':
            RNN = DGReceiver
        elif rnn_type == 'dgsend':
            RNN = DGSender
        else:
            RNN = getattr(nn, f'{rnn_type}')
        self.rnn = RNN(
            input_size=embedding_size,
            hidden_size=embedding_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.h_out = nn.Linear(embedding_size, num_meaning_types * meanings_per_type)
        self.h_predict_correct = nn.Linear(embedding_size, num_meaning_types * 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, utts, do_predict_correct=False):
        embs = self.embedding(utts)
        embs = self.drop(embs)
        seq_len, batch_size, embedding_size = embs.size()
        output, state = self.rnn(embs)

        if self.rnn_type == 'LSTM':
            h, c = state
            x = h[-1]
        elif self.rnn_type == 'SRU':
            h = state
            x = h[-1]
        else:
            h = state
            x = h[-1]
        x = self.h_out(self.drop(x))
        view_list = [self.num_meaning_types, self.meanings_per_type]
        x = x.view(batch_size, *view_list)
        if do_predict_correct:
            pred_correct = self.h_predict_correct(state)
            pred_correct = pred_correct.view(batch_size, self.num_meaning_types, 2)
            return pred_correct, x
        return x


class RNN2LModel(RNNModel):
    def __init__(
        self, embedding_size: int, vocab_size: int, utt_len: int, num_meaning_types: int, meanings_per_type: int,
        rnn_type: str, dropout: float
    ):
        super().__init__(
            embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type,
            rnn_type, dropout, num_layers=2)
