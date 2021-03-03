import torch
import numpy as np

from mll import gumbel_loss


def test_gumbel_loss():
    N = 4
    V = 5
    preds = (torch.rand(N, V, requires_grad=True) + 0).round()
    print('preds', preds)
    assert preds.requires_grad

    tgt = torch.from_numpy(np.random.choice(V, N, replace=True))

    loss = gumbel_loss.gumbel_loss(preds, tgt)
    print('loss', loss)
    print('loss %.3f' % loss)
    assert loss.item() > 0
    assert loss.item() < 1
    assert loss.requires_grad
