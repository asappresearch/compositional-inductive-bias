"""
calculate loss between hard gumbel output (one-hot) and indexed ground
we will fluff up indexed ground to one-hot, and calculate loss as the
mean of the squared difference (ie the absolute differnece)
"""
import torch


def gumbel_loss(preds, tgt):
    """
    preds are [seq_len, batch_size, vocab_size], onehot, float32
    tgt is [seq_len, batch_size], indices, int64

    preds is assumed to be only 0s and 1s. Violating this
    assumption will break this function
    """
    # everything where gnd is should be 1, and everything else should be 0
    # first flatten everything
    vocab_size = preds.size(-1)
    preds = preds.contiguous().view(-1, vocab_size)
    tgt = tgt.contiguous().view(-1)
    N = tgt.size(0)

    tgt_fluffy = torch.zeros(N, vocab_size, dtype=torch.float32)
    tgt_fluffy[torch.arange(N), tgt] = 1
    tgt_pred_diff = tgt_fluffy - preds
    loss = (tgt_pred_diff * tgt_pred_diff).mean()
    return loss
