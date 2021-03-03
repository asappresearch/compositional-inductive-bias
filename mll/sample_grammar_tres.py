"""
Code adapted from https://github.com/jacobandreas/tre/blob/master/comm.ipynb
forked 2021 feb 7
"""
import argparse
import torch
import time
import csv
import random
from collections import OrderedDict
from torch import nn
import numpy as np

from mll import mem_common, tre


class Compose(nn.Module):
    def __init__(self, num_terms, seq_len, vocab_size):
        super().__init__()
        self.num_terms = num_terms
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.proj_l = [nn.Linear(self.seq_len, self.seq_len) for i in range(num_terms)]

    def forward(self, *args):
        x_l = [arg.view(1, self.vocab_size + 2, self.seq_len) for arg in args]
        x_l = [proj(x) for x, proj in zip(x_l, self.proj_l)]
        x = torch.stack(x_l)
        x = x.sum(dim=0)
        return x.view(1, (self.vocab_size + 2) * self.seq_len)


def lift(seq_len, vocab_size, msg):
    data = np.zeros((vocab_size + 2, seq_len))
    for i, tok in enumerate(msg):
        data[tok, i] = 1
    return data.ravel()


def evaluate_tre(args, grammar_object):
    seq_len = args.num_meaning_types * args.tokens_per_meaning
    COMP_FN = Compose(
        num_terms=args.num_meaning_types,
        seq_len=seq_len,
        vocab_size=args.vocab_size)
    ERR_FN = tre.L1Dist()

    meanings = grammar_object.meanings
    utts = grammar_object.utterances_by_meaning

    print('sampling')
    N = meanings.size(0)
    if N > args.max_samples:
        idxes = np.random.choice(N, args.max_samples, replace=False)
        meanings = meanings[idxes]
        utts = utts[idxes]

    N = meanings.size(0)
    print('N after sampling', N)
    reps = []
    specs = []
    last_print = time.time()
    for i in range(N):
        meaning = meanings[i]
        if args.binary:
            spec = (f"0{meaning[0]}", f"1{meaning[1]}")
            for t in range(meaning.size(0)):
                if t >= 2:
                    spec = (spec, f"{t}{meaning[t]}")
        else:
            spec = tuple([f'{t}{meaning[t]}' for t in range(meaning.size(0))])
        specs.append(spec)
        if time.time() - last_print >= 3.0:
            print(i, '/', N)
            last_print = time.time()
    reps = [lift(seq_len=seq_len, vocab_size=args.vocab_size, msg=utts[i]) for i in range(utts.size(0))]
    print('prepared reps and specs')

    comp = tre.evaluate(reps, specs, COMP_FN, ERR_FN, quiet=False, steps=1000)
    comp = np.mean(comp)
    print('TRE %.3f' % comp)
    return comp


def write_results(out_csv, results, fieldnames):
    with open(out_csv, 'w') as f_out:
        dict_writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        dict_writer.writeheader()
        for row in results:
            dict_writer.writerow(row)


def run(args):
    results = []
    fieldnames = OrderedDict()
    for grammar in args.grammars:
        corruption = None
        grammar_family = grammar
        if grammar not in ['Compositional', 'Holistic']:
            corruption = grammar
            grammar_family = 'Compositional'
        result = {'grammar': grammar}
        fieldnames['grammar'] = None
        fieldnames['mean_str'] = None
        results.append(result)
        results_l = []
        for seed in args.seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            Grammar = getattr(mem_common, f'{grammar_family}Grammar')
            grammar_object = Grammar(
                num_meaning_types=args.num_meaning_types,
                tokens_per_meaning=args.tokens_per_meaning,
                meanings_per_type=args.meanings_per_type,
                vocab_size=args.vocab_size,
                corruptions=corruption
            )
            tre = evaluate_tre(args=args, grammar_object=grammar_object)
            results_l.append(tre)
            fieldnames[str(seed)] = None
            result[str(seed)] = '%.3f' % tre
            print(grammar, seed, '%.3f' % tre)
            write_results(out_csv=args.out_csv, results=results, fieldnames=fieldnames.keys())
        assert len(results_l) == len(args.seeds)
        mean = np.mean(results_l).item()
        stderr = np.std(results_l) / len(results_l)
        mean_str = f'{mean:.3f}+/-{stderr:.3f}'
        print(mean_str)
        result['mean_str'] = mean_str
        write_results(out_csv=args.out_csv, results=results, fieldnames=fieldnames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meanings', type=str, default='5x10')
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--max-samples', type=int, default=300)
    parser.add_argument('--grammar', type=str, default='Compositional')
    parser.add_argument('--tokens-per-meaning', type=int, default=4)
    parser.add_argument('--vocab-size', type=int, default=4, help='excludes any terminator')
    parser.add_argument('--out-csv', type=str, required=True)
    parser.add_argument('--seeds', type=str, default='123,124,125,126,127')
    parser.add_argument('--grammars', type=str,
                        default='Compositional,Holistic,RandomProj,WordPairSums,Permute,Cumrot,'
                                'ShuffleWords,ShuffleWordsDet')
    args = parser.parse_args()
    args.grammars = args.grammars.split(',')
    args.seeds = [int(v) for v in args.seeds.split(',')]
    args.num_meaning_types, args.meanings_per_type = [int(v) for v in args.meanings.split('x')]
    run(args)
