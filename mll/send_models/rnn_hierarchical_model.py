"""
    concept is that embedding goes to upper layer
    lower layer generates a word or so, then sets lower stopness
    then upper layer takes a step, and we go down to lower
    except all is soft
    so:
    - when lower_stopness is close to 1, top layer changes
       - otherwise conserves existing state
    - for the lower layer:
        - when lower stopness close to 1, lower layer copies state
          mostly from upper state
        - otherwise from new lower state
"""
import torch
from torch import nn
import torch.nn.functional as F

from ulfs.stochastic_trajectory import StochasticTrajectory
from ulfs import rl_common

from mll.darts_cell import DGReceiverCell, DGSenderCell


class HierZeroModel(nn.Module):
    supports_gumbel = False
    supports_dropout = True
    """
    decoder only; uses fake zeros inputs, so just need to provide input state, and utt_len
    """
    def __init__(
            self, embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type, rnn_type,
            dropout: float = 0
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type
        self.utt_len = utt_len
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type

        # assert rnn_type in ['RNN', 'GRU', 'LSTM']  # temporary dev guard
        if rnn_type == 'dgrecv':
            RNNCell = DGReceiverCell
        elif rnn_type == 'dgsend':
            RNNCell = DGSenderCell  # type: ignore
        else:
            RNNCell = getattr(nn, f'{rnn_type}Cell')
        self.rnn_upper = RNNCell(
            input_size=1,
            hidden_size=embedding_size
        )
        self.rnn_lower = RNNCell(
            input_size=1,
            hidden_size=embedding_size
        )

        self.linear_lower_stop = nn.Linear(embedding_size, 1)

        self.linear_out = nn.Linear(embedding_size, vocab_size + 1)
        self.drop = nn.Dropout(dropout)

        self.meanings_offset = (torch.ones(num_meaning_types, dtype=torch.int64).cumsum(dim=-1) - 1) * meanings_per_type
        self.embeddings = nn.Embedding(num_meaning_types * meanings_per_type, embedding_size)

    def forward(self, meanings):
        """
        meanings are [N][T], index-encoded
        """
        N, T = meanings.size()
        device = meanings.device

        meanings_discrete_offset = meanings + self.meanings_offset
        embs = self.embeddings(meanings_discrete_offset).sum(dim=1)
        embs = self.drop(embs)

        lower_stopness = torch.ones(N, 1, dtype=torch.float32, device=device)

        if self.rnn_type in ['LSTM']:
            upper_state = [
                embs,
                torch.zeros(N, self.embedding_size, dtype=torch.float32, device=device)
            ]
            lower_state = [
                torch.zeros(N, self.embedding_size, dtype=torch.float32, device=device),
                torch.zeros(N, self.embedding_size, dtype=torch.float32, device=device)
            ]
        else:
            upper_state = embs
            lower_state = torch.zeros(N, self.embedding_size, dtype=torch.float32, device=device)

        fake_input = torch.zeros(N, 1, dtype=torch.float32, device=device)
        all_lower_stopness = torch.zeros(self.utt_len, N, dtype=torch.float32, device=device)
        outputs = torch.zeros(self.utt_len, N, self.embedding_size, dtype=torch.float32, device=device)
        for m in range(self.utt_len):
            new_upper_state = self.rnn_upper(fake_input, upper_state)

            if self.rnn_type in ['LSTM']:
                upper_state[0] = lower_stopness * new_upper_state[0] + (1 - lower_stopness) * upper_state[0]
                upper_state[1] = lower_stopness * new_upper_state[1] + (1 - lower_stopness) * upper_state[1]

                lower_state = list(lower_state)
                lower_state[0] = lower_stopness * upper_state[0] + (1 - lower_stopness) * lower_state[0]
                lower_state[1] = lower_stopness * upper_state[1] + (1 - lower_stopness) * lower_state[1]
            else:
                upper_state = lower_stopness * new_upper_state + (1 - lower_stopness) * upper_state
                lower_state = lower_stopness * upper_state + (1 - lower_stopness) * lower_state
            lower_state = self.rnn_lower(fake_input, lower_state)
            if self.rnn_type in ['LSTM']:
                stopness_input = lower_state[0]
            else:
                stopness_input = lower_state
            lower_stopness = torch.sigmoid(self.linear_lower_stop(stopness_input))
            all_lower_stopness[m] = lower_stopness.squeeze(-1)
            lower_stopness = lower_stopness
            if self.rnn_type in ['LSTM']:
                outputs[m] = lower_state[0]
            else:
                outputs[m] = lower_state

        self.all_lower_stopness = all_lower_stopness
        utts = self.linear_out(outputs)
        return utts


class HierAutoRegModel(nn.Module):
    supports_gumbel = True
    supports_dropout = True
    """
    decoder only

    the autoreg will be for the lower network only (?)
    """
    def __init__(
        self, embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type, rnn_type,
        gumbel: bool, gumbel_tau: float,
        dropout: float = 0
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type
        self.utt_len = utt_len
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.start_token = vocab_size  # put at end
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau

        self.linear_in = nn.Linear(num_meaning_types * meanings_per_type, embedding_size)

        self.v2e = nn.Linear(vocab_size + 1, embedding_size)
        self.e2v = nn.Linear(embedding_size, vocab_size + 1)
        self.drop = nn.Dropout(dropout)

        # assert rnn_type in ['RNN', 'GRU', 'LSTM']  # temporary dev guard
        if rnn_type == 'dgrecv':
            RNNCell = DGReceiverCell
        elif rnn_type == 'dgsend':
            RNNCell = DGSenderCell  # type: ignore
        else:
            RNNCell = getattr(nn, f'{rnn_type}Cell')
        self.rnn_upper = RNNCell(
            input_size=1,
            hidden_size=embedding_size
        )
        self.rnn_lower = RNNCell(
            input_size=embedding_size,
            hidden_size=embedding_size
        )

        self.linear_lower_stop = nn.Linear(embedding_size, 1)

        self.meanings_offset = (torch.ones(num_meaning_types, dtype=torch.int64).cumsum(dim=-1) - 1) * meanings_per_type
        self.embeddings = nn.Embedding(num_meaning_types * meanings_per_type, embedding_size)

    def forward(self, meanings):
        """
        meanings are [N][T], index-encoded
        """
        N, T = meanings.size()
        device = meanings.device

        meanings_discrete_offset = meanings + self.meanings_offset
        embs = self.embeddings(meanings_discrete_offset).sum(dim=1)
        embs = self.drop(embs)

        lower_stopness = torch.ones(N, 1, dtype=torch.float32, device=device)

        if self.rnn_type in ['LSTM']:
            upper_state = [
                embs,
                torch.zeros(N, self.embedding_size, dtype=torch.float32, device=device)
            ]
            lower_state = [
                torch.zeros(N, self.embedding_size, dtype=torch.float32, device=device),
                torch.zeros(N, self.embedding_size, dtype=torch.float32, device=device)
            ]
        else:
            upper_state = embs
            lower_state = torch.zeros(N, self.embedding_size, dtype=torch.float32, device=device)

        upper_fake_input = torch.zeros(N, 1, dtype=torch.float32, device=device)
        last_lower_token = torch.zeros(N, self.vocab_size + 1, dtype=torch.float32, device=device)
        last_lower_token[:, self.start_token] = 1
        all_lower_stopness = torch.zeros(self.utt_len, N, dtype=torch.float32, device=device)
        utts = torch.zeros(self.utt_len, N, self.vocab_size + 1, dtype=torch.float32, device=device)
        for m in range(self.utt_len):
            new_upper_state = self.rnn_upper(upper_fake_input, upper_state)

            if self.rnn_type in ['LSTM']:
                upper_state[0] = lower_stopness * new_upper_state[0] + (1 - lower_stopness) * upper_state[0]
                upper_state[1] = lower_stopness * new_upper_state[1] + (1 - lower_stopness) * upper_state[1]

                lower_state = list(lower_state)
                lower_state[0] = lower_stopness * upper_state[0] + (1 - lower_stopness) * lower_state[0]
                lower_state[1] = lower_stopness * upper_state[1] + (1 - lower_stopness) * lower_state[1]
            else:
                upper_state = lower_stopness * new_upper_state + (1 - lower_stopness) * upper_state
                lower_state = lower_stopness * upper_state + (1 - lower_stopness) * lower_state

            embedded_last_lower_token = self.v2e(last_lower_token)
            lower_state = self.rnn_lower(embedded_last_lower_token, lower_state)
            if self.rnn_type in ['LSTM']:
                stopness_input = lower_state[0]
            else:
                stopness_input = lower_state
            lower_stopness = torch.sigmoid(self.linear_lower_stop(stopness_input))
            all_lower_stopness[m] = lower_stopness.squeeze(-1)
            lower_stopness = lower_stopness
            if self.rnn_type in ['LSTM']:
                lower_embs = lower_state[0]
            else:
                lower_embs = lower_state

            lower_logits = self.e2v(lower_embs)
            if self.gumbel:
                if self.training:
                    lower_token_probs = F.gumbel_softmax(lower_logits, hard=True, tau=self.gumbel_tau)
                else:
                    _, _lower_token = lower_logits.max(dim=-1)
                    lower_token_probs = torch.zeros(
                        lower_logits.size(), device=lower_logits.device, dtype=torch.float32)
                    lower_token_probs[torch.arange(N), _lower_token] = 1
                utts[m] = lower_token_probs
            else:
                lower_token_probs = torch.softmax(lower_logits, dim=-1)
                utts[m] = lower_logits
            last_lower_token = lower_token_probs

        self.all_lower_stopness = all_lower_stopness
        return utts


class HierAutoRegSamplingModel(nn.Module):
    supports_gumbel = False
    supports_dropout = True
    """
    decoder only

    the autoreg will be for the lower network only
    """
    def __init__(
        self, embedding_size, vocab_size, utt_len, num_meaning_types, meanings_per_type, rnn_type,
        dropout: float = 0
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type
        self.utt_len = utt_len
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.start_token = vocab_size  # put at end

        self.linear_in = nn.Linear(num_meaning_types * meanings_per_type, embedding_size)

        self.v2e = nn.Embedding(vocab_size + 1, embedding_size)
        self.e2v = nn.Linear(embedding_size, vocab_size + 1)
        self.drop = nn.Dropout(dropout)

        # assert rnn_type in ['RNN', 'GRU', 'LSTM']  # temporary dev guard
        if rnn_type == 'dgrecv':
            RNNCell = DGReceiverCell
        elif rnn_type == 'dgsend':
            RNNCell = DGSenderCell  # type: ignore
        else:
            RNNCell = getattr(nn, f'{rnn_type}Cell')
        self.rnn_upper = RNNCell(
            input_size=1,
            hidden_size=embedding_size
        )
        self.rnn_lower = RNNCell(
            input_size=embedding_size,
            hidden_size=embedding_size
        )

        self.linear_lower_stop = nn.Linear(embedding_size, 1)

        self.meanings_offset = (torch.ones(num_meaning_types, dtype=torch.int64).cumsum(dim=-1) - 1) * meanings_per_type
        self.embeddings = nn.Embedding(num_meaning_types * meanings_per_type, embedding_size)

    def forward(self, meanings):
        """
        meanings are [N][T], index-encoded
        """
        N, T = meanings.size()
        device = meanings.device

        meanings_discrete_offset = meanings + self.meanings_offset
        embs = self.embeddings(meanings_discrete_offset).sum(dim=1)
        embs = self.drop(embs)

        if self.rnn_type in ['LSTM']:
            upper_state = [
                embs,
                torch.zeros(N, self.embedding_size, dtype=torch.float32, device=device)
            ]
            lower_state = [
                torch.zeros(N, self.embedding_size, dtype=torch.float32, device=device),
                torch.zeros(N, self.embedding_size, dtype=torch.float32, device=device)
            ]
        else:
            upper_state = embs
            lower_state = torch.zeros(N, self.embedding_size, dtype=torch.float32, device=device)

        lower_stopness = torch.ones((N, 1), dtype=torch.int64, device=device)
        upper_fake_input = torch.zeros((N, 1), dtype=torch.float32, device=device)
        last_lower_token = torch.full((N, ), self.start_token, dtype=torch.int64, device=device)
        all_lower_stopness = torch.zeros(self.utt_len, N, dtype=torch.int64, device=device)
        utts = torch.zeros(self.utt_len, N, dtype=torch.int64, device=device)
        stochastic_trajectory = StochasticTrajectory()
        for m in range(self.utt_len):
            new_upper_state = self.rnn_upper(upper_fake_input, upper_state)

            if self.rnn_type in ['LSTM']:
                upper_state[0] = lower_stopness * new_upper_state[0] + (1 - lower_stopness) * upper_state[0]
                upper_state[1] = lower_stopness * new_upper_state[1] + (1 - lower_stopness) * upper_state[1]

                lower_state = list(lower_state)
                lower_state[0] = lower_stopness * upper_state[0] + (1 - lower_stopness) * lower_state[0]
                lower_state[1] = lower_stopness * upper_state[1] + (1 - lower_stopness) * lower_state[1]
            else:
                upper_state = lower_stopness * new_upper_state + (1 - lower_stopness) * upper_state
                lower_state = lower_stopness * upper_state + (1 - lower_stopness) * lower_state

            embedded_last_lower_token = self.v2e(last_lower_token)
            lower_state = self.rnn_lower(embedded_last_lower_token, lower_state)
            if self.rnn_type in ['LSTM']:
                stopness_input = lower_state[0]
            else:
                stopness_input = lower_state
            lower_stopness = torch.sigmoid(self.linear_lower_stop(stopness_input))
            all_lower_stopness[m] = lower_stopness.squeeze(-1)
            lower_stopness = lower_stopness
            if self.rnn_type in ['LSTM']:
                lower_embs = lower_state[0]
            else:
                lower_embs = lower_state

            lower_logits = self.e2v(lower_embs)
            if self.training:
                lower_token_probs = torch.softmax(lower_logits, dim=-1)
                s = rl_common.draw_categorical_sample(action_probs=lower_token_probs, batch_idxes=None)
                stochastic_trajectory.append_stochastic_sample(s)
                token = s.actions.view(-1)
            else:
                _, token = lower_logits.max(dim=-1)
            utts[m] = token
            last_lower_token = token

        self.all_lower_stopness = all_lower_stopness
        return {
            'stochastic_trajectory': stochastic_trajectory,
            'utterances': utts
        }
