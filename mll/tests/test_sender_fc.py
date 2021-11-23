import torch
import numpy as np

from mll.send_models import fc1l_model, fc2l_model


def test_fc1l():
    N = 5
    utt_len = 20
    vocab_size = 4
    embedding_size = 11
    num_meaning_types = 5
    meanings_per_type = 10

    inputs = torch.from_numpy(np.random.choice(meanings_per_type, (N, num_meaning_types), replace=True))
    fc1l = fc1l_model.FC1LModel(
        embedding_size=embedding_size, vocab_size=vocab_size, utt_len=utt_len, num_meaning_types=num_meaning_types,
        meanings_per_type=meanings_per_type)
    output = fc1l(inputs)
    assert list(output.size()) == [utt_len, N, vocab_size + 1]


def test_fc2l():
    N = 5
    utt_len = 20
    vocab_size = 4
    embedding_size = 11
    num_meaning_types = 5
    meanings_per_type = 10

    inputs = torch.from_numpy(np.random.choice(meanings_per_type, (N, num_meaning_types), replace=True))
    fc1l = fc2l_model.FC2LModel(
        embedding_size=embedding_size, vocab_size=vocab_size, utt_len=utt_len, num_meaning_types=num_meaning_types,
        meanings_per_type=meanings_per_type, dropout=0.5)
    output = fc1l(inputs)
    assert list(output.size()) == [utt_len, N, vocab_size + 1]
