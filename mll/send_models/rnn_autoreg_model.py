import torch
from torch import nn
import torch.nn.functional as F

from ulfs import rl_common
from ulfs.stochastic_trajectory import StochasticTrajectory

from mll.darts_cell import DGReceiver, DGSender


class RNNSamplingModel(nn.Module):
    supports_gumbel = False
    supports_dropout = True

    def __init__(
            self, embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type, rnn_type,
            dropout, num_layers=1
    ):
        super().__init__()
        self.utt_len = utt_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.rnn_type = rnn_type
        self.start_token = vocab_size
        self.num_layers = num_layers
        print('rnn dropout', dropout)

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
        self.e2v = nn.Linear(embedding_size, vocab_size + 1)
        self.v2e = nn.Embedding(vocab_size + 1, embedding_size)
        self.drop = nn.Dropout(dropout)

        self.meanings_offset = (torch.ones(num_meaning_types, dtype=torch.int64).cumsum(dim=-1) - 1) * meanings_per_type
        self.embeddings = nn.Embedding(num_meaning_types * meanings_per_type, embedding_size)

    def forward(self, meanings):
        """
        Input is meanings
        output is discrete sampled utterance and stochastic trajectory
        """
        N, T = meanings.size()
        batch_size = meanings.size(0)
        meanings_discrete_offset = meanings + self.meanings_offset
        embs = self.embeddings(meanings_discrete_offset).sum(dim=1)
        embs = self.drop(embs)

        if self.rnn_type in ['LSTM']:
            h = torch.zeros(self.num_layers, N, self.embedding_size, dtype=torch.float32, device=meanings.device)
            h[0] = embs
            c = torch.zeros(self.num_layers, N, self.embedding_size, dtype=torch.float32, device=meanings.device)
        elif self.rnn_type in ['SRU']:
            c = torch.zeros(self.num_layers, N, self.embedding_size, dtype=torch.float32, device=meanings.device)
            c[0] = embs
        else:
            h = torch.zeros(self.num_layers, N, self.embedding_size, dtype=torch.float32, device=meanings.device)
            h[0] = embs

        # state = embs.unsqueeze(0)
        last_token = torch.full((batch_size, ), self.start_token, device=embs.device, dtype=torch.int64)
        utterances = torch.zeros(self.utt_len, batch_size, dtype=torch.int64, device=embs.device)
        stochastic_trajectory = StochasticTrajectory()
        for t in range(self.utt_len):
            input_emb = self.v2e(last_token)
            input_emb = input_emb.unsqueeze(0)

            if self.rnn_type == 'SRU':
                h, c = self.rnn(input_emb, c)
                output = h
            elif self.rnn_type == 'LSTM':
                output, (h, c) = self.rnn(input_emb, (h, c))
            else:
                output, h = self.rnn(input_emb, h)
            output = output.squeeze(0)

            token_logits = self.e2v(output)
            if self.training:
                token_probs = F.softmax(token_logits, dim=-1)
                s = rl_common.draw_categorical_sample(
                    action_probs=token_probs, batch_idxes=None)
                stochastic_trajectory.append_stochastic_sample(s=s)
                token = s.actions.view(-1)
            else:
                _, token = token_logits.max(dim=-1)
            utterances[t] = token
            last_token = token
        res = {
            'stochastic_trajectory': stochastic_trajectory,
            'utterances': utterances,  # [utt_len, N]
        }
        return res


class RNNSampling2LModel(RNNSamplingModel):
    supports_gumbel = False
    supports_dropout = True

    def __init__(
            self, embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type, rnn_type,
            dropout
    ):
        super().__init__(
            embedding_size=embedding_size, vocab_size=vocab_size, num_meaning_types=num_meaning_types,
            meanings_per_type=meanings_per_type, rnn_type=rnn_type, dropout=dropout,
            utt_len=utt_len,
            num_layers=2
        )


class RNNAutoRegModel(nn.Module):
    supports_gumbel = True
    supports_dropout = True

    """
    feedback the last token as the input token each time (cf RNNModel, which uses
    dummy input each decoder timestep)
    """
    def __init__(
            self, embedding_size: int, vocab_size: int, utt_len: int, num_meaning_types: int,
            meanings_per_type: int, rnn_type: str, dropout: float,
            gumbel: bool, gumbel_tau: float,
            num_layers: int = 1):
        self.embedding_size = embedding_size
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type
        self.utt_len = utt_len
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau
        self.start_token = vocab_size

        super().__init__()
        """
        so, we'll make the meanings one-hot, then pass through a Linear?

        """
        self.embeddings = nn.Embedding(num_meaning_types * meanings_per_type, embedding_size)
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
            embedding_size,
            embedding_size,
            num_layers=self.num_layers,
            dropout=dropout
        )
        self.drop = nn.Dropout(dropout)
        self.e2v = nn.Linear(embedding_size, vocab_size + 1)
        self.v2e = nn.Linear(vocab_size + 1, embedding_size)

        self.meanings_offset = (torch.ones(num_meaning_types, dtype=torch.int64).cumsum(
          dim=-1) - 1) * meanings_per_type

    def forward(self, meanings):
        """
        meanings are [N][T], index-encoded
        we'll fluff up to one-hot, then pass through a linear
        """
        N, T = meanings.size()
        meanings_discrete_offset = meanings + self.meanings_offset

        embs = self.embeddings(meanings_discrete_offset).sum(dim=1)
        embs = self.drop(embs)
        # we'll use this as the state for a generator now

        if self.rnn_type in ['LSTM']:
            h = torch.zeros(self.num_layers, N, self.embedding_size, dtype=torch.float32, device=meanings.device)
            h[0] = embs
            c = torch.zeros(self.num_layers, N, self.embedding_size, dtype=torch.float32, device=meanings.device)
        elif self.rnn_type in ['SRU']:
            c = torch.zeros(self.num_layers, N, self.embedding_size, dtype=torch.float32, device=meanings.device)
            c[0] = embs
        else:
            h = torch.zeros(self.num_layers, N, self.embedding_size, dtype=torch.float32, device=meanings.device)
            h[0] = embs

        last_token = torch.zeros(N, self.vocab_size + 1, dtype=torch.float32, device=meanings.device)
        last_token[:, self.start_token] = 1
        utts = torch.zeros(self.utt_len, N, self.vocab_size + 1, device=meanings.device)
        for t in range(self.utt_len):
            input_emb = self.v2e(last_token)
            input_emb = input_emb.unsqueeze(0)
            if self.rnn_type == 'SRU':
                h, c = self.rnn(input_emb, c)
                output = h
            elif self.rnn_type == 'LSTM':
                output, (h, c) = self.rnn(input_emb, (h, c))
            else:
                output, h = self.rnn(input_emb, h)
            output = output.squeeze(0)

            token_logits = self.e2v(output)

            if self.gumbel:
                if self.training:
                    token_probs = F.gumbel_softmax(token_logits, tau=self.gumbel_tau, hard=True)
                else:
                    _, token = token_logits.max(dim=-1)
                    token_probs = torch.zeros(token_logits.size(), device=token_logits.device, dtype=torch.float32)
                    token_probs[torch.arange(N), token] = 1
                utts[t, :] = token_probs
            else:
                utts[t, :] = token_logits
                token_probs = F.softmax(token_logits, dim=-1)
            last_token = token_probs
        return utts


class RNNAutoReg2LModel(RNNAutoRegModel):
    supports_gumbel = True
    supports_dropout = True
    """
    feedback the last token as the input token each time (cf RNNModel, which uses
    dummy input each decoder timestep)
    """
    def __init__(
            self, embedding_size: int, vocab_size: int, utt_len: int, num_meaning_types: int,
            meanings_per_type: int, rnn_type: str, dropout: float,
            gumbel: bool, gumbel_tau: float):
        super().__init__(
            embedding_size=embedding_size, vocab_size=vocab_size, num_meaning_types=num_meaning_types,
            meanings_per_type=meanings_per_type, rnn_type=rnn_type, dropout=dropout,
            gumbel=gumbel, gumbel_tau=gumbel_tau, utt_len=utt_len,
            num_layers=2)
