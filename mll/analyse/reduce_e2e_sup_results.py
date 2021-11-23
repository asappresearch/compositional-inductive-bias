import argparse
import csv
import os
from collections import defaultdict

import pandas as pd
import numpy as np

from mll.analyse import reduce_common


include_architectures = set([
    'RNNAutoReg:dgsend+RNN:dgrecv',
    'RNNAutoReg:GRU+RNN:GRU',
    'HierZero:GRU+Hier:GRU',
    'RNNZero2L:RNN+RNN2L:RNN',
    'RNNZero:GRU+RNN:GRU',
    'RNNZero:RNN+RNN:RNN',
    'RNNZero:GRU+RNN:GRU'
])


def do_for_ref(args, repr, ref):
    with open(f'pull/{ref}.csv', 'r') as f:
        dict_reader = csv.DictReader(f)
        rows = list(dict_reader)
    if args.filter_arch:
        new_rows = []
        for row in rows:
            key_str = '+'.join([row[key] for key in ['send_arch', 'recv_arch']])
            if key_str in include_architectures:
                new_rows.append(row)
        rows = new_rows
    df_src = pd.DataFrame(rows)

    archs = [row['send_arch'] + '+' + row['recv_arch'] for row in rows]
    print(archs)
    grammars = [field.replace('_e2e_acc', '') for field in rows[0].keys() if field.endswith('_e2e_acc')]
    print('grammars', grammars)

    e2e_send_acc_cols = ['send_arch', 'recv_arch'] + [f'{grammar}_e2e_{args.direction}_acc' for grammar in grammars]
    df_e2e_send_acc = df_src[e2e_send_acc_cols].copy()
    name_mapping = {f'{grammar}_e2e_{args.direction}_acc': reduce_common.name_mapping.get(
        grammar, grammar) for grammar in grammars}
    df_e2e_send_acc = df_e2e_send_acc.rename(columns=name_mapping)
    field_names = [reduce_common.name_mapping.get(grammar, grammar) for grammar in grammars]
    df_e2e_send_acc[field_names] = df_e2e_send_acc[field_names].astype(np.float32)
    df_e2e_send_acc['repr'] = repr
    df_e2e_send_acc = df_e2e_send_acc[['repr', 'send_arch', 'recv_arch'] + field_names].reindex()

    def create_sup_ok(df_src, direction):
        grammar_fields = [f'{grammar}_sup_{direction}_acc' for grammar in grammars]
        sup_ok = df_src[['send_arch', 'recv_arch'] + grammar_fields].copy()
        sup_ok[grammar_fields] = sup_ok[grammar_fields].astype(np.float32)
        name_mapping = {
            f'{grammar}_sup_{direction}_acc': reduce_common.name_mapping.get(grammar, grammar) for grammar in grammars}
        sup_ok = sup_ok.rename(columns=name_mapping)
        grammar_fields = [reduce_common.name_mapping.get(grammar, grammar) for grammar in grammars]
        sup_ok[grammar_fields] = sup_ok[grammar_fields] >= (args.sup_tgt_acc - 1e-4)
        return sup_ok

    send_sup_ok = create_sup_ok(df_src, 'send')
    recv_sup_ok = create_sup_ok(df_src, 'recv')
    send_recv_ok = send_sup_ok.copy()
    field_names = [reduce_common.name_mapping.get(grammar, grammar) for grammar in grammars]
    send_recv_ok[field_names] = send_sup_ok[field_names] & recv_sup_ok[field_names]

    df_e2e_send_acc[field_names] = df_e2e_send_acc[field_names].mask(~send_recv_ok[field_names])
    return df_e2e_send_acc


def run(args):
    df_by_ref_by_repr = defaultdict(dict)
    for repr, refs in args.refs_by_repr.items():
        for ref in refs:
            df_by_ref_by_repr[repr][ref] = do_for_ref(ref=ref, repr=repr, args=args)
            print(df_by_ref_by_repr[repr][ref])

    df_str_by_repr = {}
    df_mean_by_repr = {}
    df_counts_by_repr = {}
    for repr, df_by_ref in df_by_ref_by_repr.items():
        print(repr)
        df_str, df_mean, df_counts = reduce_common.average_of(
            list(df_by_ref.values()),
            show_stderr=args.show_stderr, show_mean=args.show_mean)
        df_str_by_repr[repr] = df_str
        df_mean_by_repr[repr] = df_mean
        df_counts_by_repr[repr] = df_counts
    df_str = pd.concat(df_str_by_repr, ignore_index=True)
    df_mean = pd.concat(df_mean_by_repr, ignore_index=True)
    df_counts = pd.concat(df_counts_by_repr, ignore_index=True)
    print('df_str', df_str)
    print('df_counts', df_counts)

    numeric_fields = [field for field, dtype in zip(df_mean.columns, df_mean.dtypes) if dtype in [np.float32, np.int64]]

    df_str.to_csv(args.out_csv)
    os.system(f'open {args.out_csv}')

    df_counts.to_csv('counts.csv')
    os.system('open counts.csv')

    titles = {'send arch': 'Send arch', 'recv arch': 'Recv arch', 'repr': 'Repr'}
    for field in numeric_fields:
        titles[field] = '\\textsc{' + field + '}'

    if args.show_mean:
        maximize = set(args.maximize) & set(numeric_fields)
        minimize = set(numeric_fields) - set(args.maximize)
    else:
        maximize, minimize = [], []

    reduce_common.write_tex(
        df_str=df_str, df_mean=df_mean, filepath=args.out_tex,
        keys=['send_arch', 'recv_arch'],
        maximize=maximize, minimize=minimize, titles=titles)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refs-by-repr', type=str, required=True,
                        help='eg soft[ib1,ib2,ib3],gumb[ib7,ib8,ib9],discr[ib4,ib5,ib6]')
    parser.add_argument('--direction', type=str, required=True, choices=['send', 'recv'],
                        help='which model to use to measure final sup accuracy')
    parser.add_argument('--sup-tgt-acc', type=float, default=0.99)
    parser.add_argument('--in-csv_templ', type=str, default='pull/{ref}.csv')
    parser.add_argument('--no-stderr', action='store_true')
    parser.add_argument('--no-mean', action='store_true')
    parser.add_argument('--out-csv', type=str, default='foo.csv')
    parser.add_argument('--out-tex', type=str, default='foo.tex')
    parser.add_argument('--filter-arch', action='store_true')
    parser.add_argument('--maximize', type=str, default='shufdet,shuf,comp')
    args = parser.parse_args()
    _refs_by_repr = {}
    for repr_refs in args.refs_by_repr.split('],'):
        _repr, _refs = repr_refs.split('[')
        _refs = _refs.replace(']', '').split(',')
        _refs_by_repr[_repr] = _refs
    args.refs_by_repr = _refs_by_repr
    print('args.refs_by_repr', args.refs_by_repr)
    # args.refs = args.refs.split(',')
    args.maximize = args.maximize.split(',')
    args.show_stderr = not args.no_stderr
    del args.__dict__['no_stderr']
    args.show_mean = not args.no_mean
    del args.__dict__['no_mean']
    run(args)
