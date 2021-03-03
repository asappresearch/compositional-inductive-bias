"""
take send or receive results, and reduce them into csv file
"""
import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, Set
import pandas as pd
import numpy as np

import reduce_common


include_architectures_by_dir: Dict[str, Set[str]] = {
    'send': set([
        'FC1L',
        'FC2L',
        'HierZero:RNN',
        'RNNZero2L:SRU',
        'HierAutoReg:RNN',
        'HierAutoReg:RNN',
        'HierZero:dgsend',
        'TransDecSoft',
        'TransDecSoft2L'
    ]),
    'recv': set([
        'RNN:LSTM',
        # 'Hier:RNN',
        'Hier:GRU',
        'Hier:dgrecv',
        'RNN:dgrecv',
        # 'RNN2L:RNN',
        'RNN2L:SRU',
        'RNN2L:GRU',
        # 'RNN2L:dgrecv',
        # 'FC',
        'FC2L',
        'CNN'
    ])
}

arch_family_rename = {
    'RNN': 'RNN1L'
}


def create_result_table_for_ref(ref, repr, args):
    with open(f'pull/{ref}.csv', 'r') as f_in:
        dict_reader = csv.DictReader(f_in)
        rows = list(dict_reader)
    new_rows = []
    for row in rows:
        if args.filter_arch and row[args.row_key] not in include_architectures_by_dir[args.dir]:
            continue
        row['repr'] = repr

        arch_family = row['arch'].split(':')[0]
        if arch_family in arch_family_rename:
            row['arch'] = arch_family_rename[arch_family] + ':' + row['arch'].split(':')[1]
        # row['arch'] = arch_rename.get(row['arch'], row['arch'])
        new_rows.append(row)
    rows = new_rows
    df = pd.DataFrame(rows)
    fieldnames = ['repr'] + list(rows[0].keys())[:-1]
    df = df[fieldnames].reindex()
    name_map = {field: reduce_common.name_mapping.get(field, field) for field in df.columns}
    df = df.rename(columns=name_map)
    fields = list(df.columns)
    if args.skip_max:
        fields = [field for field in fields if not field.endswith('_max')]
        df = df[fields].reindex()
    fields = [field for field in fields if field not in set(args.exclude_fields)]
    df = df[fields].reindex()
    numeric_fields = [field for field in fields if field not in ['repr', 'Model']]
    df[numeric_fields] = df[numeric_fields].astype(np.float32)
    df = df.sort_values(by='Model')
    return df


def run(args):
    df_by_ref_by_repr = defaultdict(dict)
    for repr, refs in args.refs_by_repr.items():
        for ref in refs:
            df_by_ref_by_repr[repr][ref] = create_result_table_for_ref(ref=ref, repr=repr, args=args)
            print(df_by_ref_by_repr[repr][ref])

    df_str_by_repr = {}
    df_mean_by_repr = {}
    df_counts_by_repr = {}
    for repr, df_by_ref in df_by_ref_by_repr.items():
        df_str, df_mean, df_counts = reduce_common.average_of(list(df_by_ref.values()),
                                                              show_stderr=args.show_stderr, show_mean=args.show_mean)
        df_str_by_repr[repr] = df_str
        df_mean_by_repr[repr] = df_mean
        df_counts_by_repr[repr] = df_counts
    df_str = pd.concat(df_str_by_repr, ignore_index=True)
    df_mean = pd.concat(df_mean_by_repr, ignore_index=True)
    df_counts = pd.concat(df_counts_by_repr, ignore_index=True)
    print('df_str', df_str)
    print('df_counts', df_counts)

    numeric_fields = [field for field in df_str.columns if field not in ['Model', 'repr']]
    minimize = set(numeric_fields) - set(['comp']) - set(args.maximize)

    df_str.to_csv(args.out_csv)
    os.system(f'open {args.out_csv}')

    df_counts.to_csv('counts.csv')
    os.system('open counts.csv')

    titles = {'repr': 'Repr'}
    for field in numeric_fields:
        titles[field] = '\\textsc{' + field + '}'

    if args.show_mean:
        maximize = args.maximize
        minimize = minimize
    else:
        maximize, minimize = [], []

    reduce_common.write_tex(
        filepath=args.out_tex, df_str=df_str, df_mean=df_mean,
        minimize=minimize, maximize=maximize,
        keys=['Model'], titles=titles, longtable=args.longtable)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refs-by-repr', type=str, required=True,
                        help='eg soft[ib1,ib2,ib3],gumb[ib7,ib8,ib9],discr[ib4,ib5,ib6]')
    parser.add_argument('--dir', type=str, required=True, choices=['send', 'recv'])
    parser.add_argument('--out-ref', type=str, required=True)
    parser.add_argument('--out-csv', type=str, default='{out_ref}.csv')
    parser.add_argument('--out-tex', type=str, default='{out_ref}.tex')
    parser.add_argument('--row-key', type=str, default='arch')
    parser.add_argument('--exclude-fields', type=str, default='sps,steps,params,Comp_time,rot+perm')
    parser.add_argument('--ints', type=str, default='')
    parser.add_argument('--skip-max', action='store_true')
    parser.add_argument('--no-mean', action='store_true')
    parser.add_argument('--no-stderr', action='store_true')
    parser.add_argument('--filter-arch', action='store_true')
    parser.add_argument('--maximize', type=str, default='shufdet,shuf')
    parser.add_argument('--exclude-best', type=str, default='model,comp,repr')
    parser.add_argument('--exclude-numeric', type=str, default='repr')
    parser.add_argument('--longtable', action='store_true', help='use longtable in latex')
    args = parser.parse_args()
    _refs_by_repr = {}
    for repr_refs in args.refs_by_repr.split('],'):
        _repr, _refs = repr_refs.split('[')
        _refs = _refs.replace(']', '').split(',')
        _refs_by_repr[_repr] = _refs
    args.refs_by_repr = _refs_by_repr
    print('args.refs_by_repr', args.refs_by_repr)
    args.ints = set(args.ints.split(','))
    args.out_csv = args.out_csv.format(**args.__dict__)
    args.out_tex = args.out_tex.format(**args.__dict__)
    args.exclude_fields = set(args.exclude_fields.split(','))
    args.maximize = set(args.maximize.split(','))
    args.exclude_best = set(args.exclude_best.split(','))
    args.exclude_numeric = set(args.exclude_numeric.split(','))
    args.show_stderr = not args.no_stderr
    del args.__dict__['no_stderr']
    args.show_mean = not args.no_mean
    del args.__dict__['no_mean']
    run(args)
