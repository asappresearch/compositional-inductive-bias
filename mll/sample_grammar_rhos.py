import argparse
import csv
import torch
from collections import OrderedDict
import numpy as np
import random
from mll import mem_common


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
            rho = grammar_object.rho
            results_l.append(rho)
            fieldnames[str(seed)] = None
            result[str(seed)] = '%.3f' % rho
            print(grammar, seed, '%.3f' % rho)
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
    parser.add_argument('--seeds', type=str, default='123,124,125,126,127')
    parser.add_argument('--grammars', type=str,
                        default='Compositional,Holistic,RandomProj,WordPairSums,Permute,Cumrot,'
                                'ShuffleWords,ShuffleWordsDet')
    parser.add_argument('--tokens-per-meaning', type=int, default=4)
    parser.add_argument('--vocab-size', type=int, default=4, help='excludes any terminator')
    parser.add_argument('--out-csv', type=str, required=True)
    args = parser.parse_args()

    args.grammars = args.grammars.split(',')
    args.seeds = [int(v) for v in args.seeds.split(',')]
    args.num_meaning_types, args.meanings_per_type = [int(v) for v in args.meanings.split('x')]
    run(args)
