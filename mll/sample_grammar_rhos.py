import argparse
import csv
import torch
import time
# from collections import OrderedDict
import numpy as np
import random
import pandas as pd

from ulfs import metrics

from mll import mem_common
from mll.analyse import reduce_common


def write_results(out_csv, results, fieldnames):
    with open(out_csv, 'w') as f_out:
        dict_writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        dict_writer.writeheader()
        for row in results:
            dict_writer.writerow(row)


def run(args):
    """
    we'll run for each seed, and put a dataframe in a list for that seed,
    then use existing tools to find the average over seeds
    for each seed, we'll create a dataframe with the metrics as columns,
    and the grammars as rows
    """
    dfs_l = []
    for seed in args.seeds:
        rows_l = []
        for grammar in args.grammars:
            corruption = None
            grammar_family = grammar
            if grammar not in ['Compositional', 'Holistic']:
                corruption = grammar
                grammar_family = 'Compositional'
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
            _utts = grammar_object.utterances_by_meaning
            _meanings = grammar_object.meanings
            _N = _meanings.size(0)
            row = {'grammar': grammar}
            for metric_name in args.metrics:
                p = None
                start_time = time.time()
                if metric_name == 'topsim':
                    metric_value = grammar_object.rho
                elif metric_name == 'resent_resnick':
                    assert not args.normalize
                    _adj = torch.full((
                        _N, args.num_meaning_types), args.meanings_per_type).cumsum(dim=-1) - args.meanings_per_type
                    meanings_for_resnick = _adj + _meanings
                    metric_value = metrics.get_residual_entropy_orig(
                        lang=_utts.numpy(),
                        target=meanings_for_resnick.numpy(),
                        num_att_vals=args.meanings_per_type,
                    )
                else:
                    child_args = {
                        'utts': _utts,
                        'meanings': _meanings
                    }
                    metric_fn = {
                        'bosdis': metrics.bos_dis,
                        'posdis': metrics.pos_dis,
                        'resent': metrics.res_ent,
                        'resent_g': metrics.res_ent_greedy,
                        'compent': metrics.compositional_entropy,
                    }[metric_name]
                    if metric_name in ['resent', 'resent_g', 'compent']:
                        child_args['normalize'] = args.normalize
                    else:  # bosdis, posdis
                        assert args.normalize
                    if metric_name == 'bosdis':
                        child_args['vocab_size'] = args.vocab_size
                    metric_value = metric_fn(**child_args)
                    if metric_name in ['resent', 'resent_g']:
                        metric_value, p = metric_value
                elapsed = time.time() - start_time
                print(metric_name, ' t=%.0fs' % elapsed, 'v=%.4f' % metric_value, 'p', p)
                row[metric_name] = metric_value
            print(row)
            rows_l.append(row)
        df = pd.DataFrame(rows_l)
        print(df.round(4))
        dfs_l.append(df)

        avg_str, avg, counts = reduce_common.average_of(dfs_l, show_stderr=False, show_mean=True)
        print('avg', avg)
        avg.to_csv(args.out_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meanings', type=str, default='5x10')
    parser.add_argument('--seeds', type=str, default='123,124,125,126,127')
    parser.add_argument('--grammars', type=str,
                        default='Compositional,Permute,RandomProj,ShuffleWordsDet,ShuffleWords,'
                                'WordPairSums,Cumrot,Holistic')
    parser.add_argument(
        '--metrics', type=str, required=True, nargs='+',
        default='posdis,bosdis,compent,topsim'.split(','),
        choices=['posdis', 'bosdis', 'compent', 'topsim', 'resent', 'resent_g', 'resent_resnick'],
        help='space separated')
    parser.add_argument('--tokens-per-meaning', type=int, default=4)
    parser.add_argument('--vocab-size', type=int, default=4, help='excludes any terminator')
    parser.add_argument('--out-csv', type=str, required=True)
    parser.add_argument('--no-normalize', action='store_true')
    args = parser.parse_args()

    args.grammars = args.grammars.split(',')
    args.seeds = [int(v) for v in args.seeds.split(',')]
    args.normalize = not args.no_normalize
    del args.__dict__['no_normalize']
    args.num_meaning_types, args.meanings_per_type = [int(v) for v in args.meanings.split('x')]
    # args.metrics = args.metrics.split(',')
    run(args)
