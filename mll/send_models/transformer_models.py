from torch import nn
import torch
import torch.nn.functional as F

from ulfs import transformer_modules, rl_common
from ulfs.stochastic_trajectory import StochasticTrajectory


class TransDecSoftModel(nn.Module):
    supports_gumbel = True
    supports_dropout = True

    def __init__(
        self, embedding_size: int, vocab_size: int, utt_len: int, gumbel: bool,
        gumbel_tau: float, num_meaning_types: int, meanings_per_type: int,
        dropout: float, num_layers: int = 1
    ):
        super().__init__()
        hidden_size = embedding_size
        self.utt_len = utt_len
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type
        self.start_token = vocab_size  # put it at the end
        self.dropout = dropout

        self.v2e = nn.Linear(vocab_size + 1, hidden_size)
        self.e2v = nn.Linear(hidden_size, vocab_size + 1)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=1, dim_feedforward=hidden_size * 4, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.pe = transformer_modules.PositionEncoding(max_len=utt_len, embedding_size=hidden_size, dropout=dropout)

        self.generate_square_subsequent_mask = nn.Transformer().generate_square_subsequent_mask

        self.meanings_offset = (torch.ones(num_meaning_types, dtype=torch.int64).cumsum(dim=-1) - 1) * meanings_per_type
        self.embeddings = nn.Embedding(num_meaning_types * meanings_per_type, embedding_size)

    def forward(self, meanings):
        meanings_discrete_offset = meanings + self.meanings_offset
        src_embs = self.embeddings(meanings_discrete_offset).sum(dim=1)
        src_embs = src_embs.unsqueeze(0)

        batch_size = src_embs.size(1)
        decoded_token_probs = torch.zeros((1, batch_size, self.vocab_size + 1), dtype=torch.float32)
        decoded_token_probs[:, :, self.start_token] = 1
        if not self.gumbel:
            decoded_token_logits = torch.zeros((self.utt_len, batch_size, self.vocab_size + 1), dtype=torch.float32)
        for i in range(self.utt_len):
            mask_dec = self.generate_square_subsequent_mask(i + 1)
            decoded_embeddings = self.v2e(decoded_token_probs)
            decoded_embeddings = self.pe(decoded_embeddings)
            outputs = self.decoder(decoded_embeddings, src_embs, tgt_mask=mask_dec)
            logits = self.e2v(outputs)
            if self.gumbel:
                last_token_logits = logits[-1:]
                if self.training:
                    last_token_probs = F.gumbel_softmax(last_token_logits, tau=self.gumbel_tau, hard=True)
                else:
                    _, last_token_idxes = last_token_logits.max(dim=-1)
                    last_token_probs = torch.zeros(last_token_logits.size(), device=logits.device, dtype=torch.float32)
                    flat_probs = last_token_probs.view(-1, self.vocab_size + 1)
                    flat_probs[torch.arange(flat_probs.size(0)), last_token_idxes.view(-1)] = 1
            else:
                last_token_probs = torch.softmax(logits[-1:], dim=-1)
                decoded_token_logits[i] = logits[-1]
            last_token = last_token_probs
            decoded_token_probs = torch.cat([decoded_token_probs, last_token], dim=0)
        if self.gumbel:
            return decoded_token_probs[1:]
        else:
            return decoded_token_logits


class TransDecSamplingModel(nn.Module):
    supports_gumbel = False
    supports_dropout = True

    def __init__(
            self, embedding_size: int, vocab_size: int, utt_len: int,
            num_meaning_types: int, meanings_per_type: int, dropout: float, num_layers: int = 1
    ):
        super().__init__()
        hidden_size = embedding_size
        self.utt_len = utt_len
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.start_token = vocab_size  # put it at the end

        self.v2e = nn.Embedding(vocab_size + 1, hidden_size)
        self.e2v = nn.Linear(hidden_size, vocab_size + 1)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=1, dim_feedforward=hidden_size * 4, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.pe = transformer_modules.PositionEncoding(max_len=utt_len, embedding_size=hidden_size, dropout=dropout)

        self.generate_square_subsequent_mask = nn.Transformer().generate_square_subsequent_mask

        self.meanings_offset = (torch.ones(num_meaning_types, dtype=torch.int64).cumsum(dim=-1) - 1) * meanings_per_type
        self.embeddings = nn.Embedding(num_meaning_types * meanings_per_type, embedding_size)

    def forward(self, meanings):
        meanings_discrete_offset = meanings + self.meanings_offset
        src_embs = self.embeddings(meanings_discrete_offset).sum(dim=1)
        src_embs = src_embs.unsqueeze(0)

    # def forward(self, src_embs, seq_len: int, start_token: int):
        batch_size = src_embs.size(1)
        decoded_tokens = torch.full((1, batch_size), self.start_token, dtype=torch.int64)
        stochastic_trajectory = StochasticTrajectory()
        for i in range(self.utt_len):
            mask_dec = self.generate_square_subsequent_mask(i + 1)
            decoded_embeddings = self.v2e(decoded_tokens)
            decoded_embeddings = self.pe(decoded_embeddings)
            outputs = self.decoder(decoded_embeddings, src_embs, tgt_mask=mask_dec)
            logits = self.e2v(outputs)

            if self.training:
                last_token_prob = torch.softmax(logits[-1], dim=-1)
                s = rl_common.draw_categorical_sample(
                    action_probs=last_token_prob, batch_idxes=None)
                stochastic_trajectory.append_stochastic_sample(s)
                last_token = s.actions
            else:
                _, last_token = logits[-1].max(dim=-1)
            decoded_tokens = torch.cat([decoded_tokens, last_token.unsqueeze(0)], dim=0)
        return {'utterances': decoded_tokens[1:], 'stochastic_trajectory': stochastic_trajectory}


class TransDecSampling2LModel(TransDecSamplingModel):
    supports_gumbel = False
    supports_dropout = True

    def __init__(
            self, embedding_size: int, vocab_size: int, utt_len: int,
            num_meaning_types: int, meanings_per_type: int, dropout: float
    ):
        super().__init__(
            embedding_size=embedding_size, vocab_size=vocab_size, utt_len=utt_len,
            num_meaning_types=num_meaning_types, meanings_per_type=meanings_per_type,
            dropout=dropout, num_layers=2)


class TransDecSoft2LModel(TransDecSoftModel):
    supports_gumbel = True
    supports_dropout = True

    def __init__(
            self, embedding_size: int, vocab_size: int, utt_len: int,
            num_meaning_types: int, meanings_per_type: int, dropout: float,
            gumbel: bool, gumbel_tau: float
    ):
        super().__init__(
            embedding_size=embedding_size, vocab_size=vocab_size, utt_len=utt_len,
            num_meaning_types=num_meaning_types, meanings_per_type=meanings_per_type,
            dropout=dropout, num_layers=2,
            gumbel=gumbel, gumbel_tau=gumbel_tau)
