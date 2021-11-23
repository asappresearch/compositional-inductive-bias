"""
take send or receive results, and reduce them into csv file

this is for step counting

columns:
arch,params,perm_ratio,proj_ratio, ...
"""
import argparse
import csv
import os
import pandas as pd
import numpy as np
import itertools

from ulfs import string_utils, tex_utils, pd_utils


def remove_empty(target_list):
    return [v for v in target_list if v != '']


def create_result_table_for_ref(ref, args):
    with open(f'pull/{ref}.csv', 'r') as f_in:
        dict_reader = csv.DictReader(f_in)
        rows = list(dict_reader)
    new_rows = []
    for row in rows:
        new_rows.append(row)
    rows = new_rows
    if len(args.include_keys) > 0:
        if args.row_key not in row:
            print('available fields', row.keys())
            raise Exception(f'row key {args.row_key} not found')
        row_by_key = {row[args.row_key]: row for row in rows}
        for key in args.include_keys:
            if key not in row_by_key:
                print('available keys', row_by_key.keys())
                raise Exception(f'key row not found {key}')
        rows = [row_by_key[key] for key in args.include_keys]
    df = pd.DataFrame(rows)
    fieldnames = list(rows[0].keys())[:-1]
    df = df[fieldnames].reindex()
    fields = list(df.columns)
    for field in args.include_fields:
        if field not in fields:
            print('available fields', fields)
            raise Exception(f'include field {field} not found')
    for field in args.numeric_fields:
        if field not in fields:
            print('available fields', fields)
            raise Exception(f'numeric_field field {field} not found')
    df = df[args.include_fields].reindex()
    df[args.numeric_fields] = df[args.numeric_fields].astype(np.float32)
    return df


def run(args):
    df_by_ref = {}
    for ref in args.in_refs:
        df_by_ref[ref] = create_result_table_for_ref(ref=ref, args=args)

    df_str, df_mean = map(pd_utils.average_of(
        list(df_by_ref.values()),
        key_fields=[args.row_key],
        show_mean=True,
        show_ci95=args.show_ci95).__getitem__, ['averaged_str', 'mean'])
    print('df_str', df_str)

    df_str.to_csv(args.out_csv)
    os.system(f'open {args.out_csv}')

    tex_utils.write_tex(
        filepath=args.out_tex, df_str=df_str,
        longtable=args.longtable)
    os.system(f'open {args.out_tex}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-refs', required=True, type=str, help='comma-separated')
    parser.add_argument('--out-ref', type=str, required=True)
    parser.add_argument('--out-csv', type=str, default='{out_ref}.csv')
    parser.add_argument('--out-tex', type=str, default='{out_ref}.tex')
    parser.add_argument('--row-key', type=str)
    parser.add_argument('--include-fields', type=str, required=True, help='comma-separated')
    parser.add_argument('--ints', type=str, default='')
    parser.add_argument('--show-ci95', action='store_true')
    parser.add_argument('--numeric-fields', type=str, default='', help='comma-separated')
    parser.add_argument('--include-keys', type=str, default='', help='comma-separated, optional')
    parser.add_argument('--longtable', action='store_true', help='use longtable in latex')
    args = parser.parse_args()
    args.in_refs = args.in_refs.split(',')
    args.in_refs = list(itertools.chain(
        *[[ref] if '-' not in ref else string_utils.ref_range_to_refs(ref) for ref in args.in_refs]))
    print('args.in_refs', args.in_refs)
    args.ints = set(args.ints.split(','))
    args.out_csv = args.out_csv.format(**args.__dict__)
    args.out_tex = args.out_tex.format(**args.__dict__)
    args.include_fields = args.include_fields.split(',')
    args.include_keys = remove_empty(args.include_keys.split(','))
    if len(args.include_keys) > 0:
        assert args.row_key is not None
    args.numeric_fields = args.numeric_fields.split(',')
    print('include_keys', args.include_keys)
    run(args)
