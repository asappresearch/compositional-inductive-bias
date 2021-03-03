import torch

from mll import corruptions_lib


def test_permute():
    num_utts = 100
    utt_len = 10
    vocab_size = 100

    permute = corruptions_lib.PermuteCorruption(num_meaning_types=0, tokens_per_meaning=0, vocab_size=0)
    utterances = (torch.rand(num_utts, utt_len) * vocab_size).long()
    permuted = permute(utterances)
    idxes = permute.idxes
    idxes_l = [(i, j) for i, j in enumerate(idxes)]
    idxes_l.sort(key=lambda x: x[1])
    rev_idx = torch.LongTensor([idx[0] for idx in idxes_l])
    assert (permuted != utterances).any()
    assert (permuted[:, rev_idx] == utterances).all()
