"""
from https://github.com/jacobandreas/tre/blob/master/evals2.py
forked 2021, feb 7
this code was provided under apache 2 license
"""
import torch
from torch import nn
from torch import optim


def flatten(not_flat):
    if not isinstance(not_flat, tuple):
        return (not_flat,)

    out = ()
    for ll in not_flat:
        out = out + flatten(ll)
    return out


class L1Dist(nn.Module):
    def forward(self, pred, target):
        return torch.abs(pred - target).sum()


class CosDist(nn.Module):
    def forward(self, x, y):
        nx, ny = nn.functional.normalize(x), nn.functional.normalize(y)
        return 1 - (nx * ny).sum()


class Objective(nn.Module):
    def __init__(self, vocab, repr_size, comp_fn, err_fn, zero_init):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), repr_size)
        if zero_init:
            self.emb.weight.data.zero_()
        self.comp = comp_fn
        self.err = err_fn

    def compose(self, e):
        if isinstance(e, tuple):
            args = (self.compose(ee) for ee in e)
            return self.comp(*args)
        return self.emb(e)

    def forward(self, rep, expr):
        return self.err(self.compose(expr), rep)


def evaluate(reps, exprs, comp_fn, err_fn, quiet=False, steps=400, include_pred=False, zero_init=True):
    vocab = {}
    for expr in exprs:
        # print('expr', expr)
        toks = flatten(expr)
        # print('toks', toks)
        for tok in toks:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    def index(e):
        if isinstance(e, tuple):
            return tuple(index(ee) for ee in e)
        return torch.LongTensor([vocab[e]])

    treps = [torch.FloatTensor([r]) for r in reps]
    texprs = [index(e) for e in exprs]

    obj = Objective(vocab, reps[0].size, comp_fn, err_fn, zero_init)
    opt = optim.RMSprop(obj.parameters(), lr=0.01)

    for t in range(steps):
        opt.zero_grad()
        errs = [obj(r, e) for r, e in zip(treps, texprs)]
        loss = sum(errs)
        loss.backward()
        if not quiet and t % 100 == 0:
            print(' %.3f' % loss.item(), end='', flush=True)
        opt.step()
    if not quiet:
        print('')

    final_errs = [err.item() for err in errs]
    if include_pred:
        lexicon = {
            k: obj.emb(torch.LongTensor([v])).data.cpu().numpy()
            for k, v in vocab.items()
        }
        composed = [obj.compose(t) for t in texprs]
        return final_errs, lexicon, composed
    else:
        return final_errs
