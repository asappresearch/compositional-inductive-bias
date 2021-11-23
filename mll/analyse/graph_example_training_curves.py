#!/usr/bin/env python3
"""
forked from utils/permuted_grammar_batch_graphs.ipynb, 2021 apr 4th

this will create a csv file with the data in to plot, which we will then
attempt to feed to pgfplots, instead a tikzpicture, instead of
uploading a pgf etc directly

This will hopefully mean fonts etc match a bit better (?)
"""
import argparse
import csv
import glob
import itertools
import json
from typing import Iterable, List, Dict, Optional
from collections import OrderedDict, defaultdict

import pandas as pd

from ulfs.params import Params
from ulfs import graphing_common, string_utils


def get_values(
        log_filepath: str, step_key: str, value_keys: Iterable[str],
        units: str, maximum_step: Optional[int] = None,
        strip_value_underscore: bool = False,
        skip_record_types: Iterable = (),
        record_type: Optional[str] = None) -> List[Dict[str, str]]:
    result_rows: List[Dict[str, str]] = []
    count_by_record_types: Dict[str, int] = defaultdict(int)
    skip_record_types = set(skip_record_types)
    with open(log_filepath, 'r') as f:
        f.readline()
        for line in f:
            dline = json.loads(line)
            _record_type = dline.get('record_type', dline.get('type', None))
            count_by_record_types[_record_type] += 1
            if record_type is not None and _record_type != record_type:
                continue
            if _record_type in skip_record_types:
                continue

            if args.step_key not in dline:
                print(dline)
                print('keys', list(dline.keys()))
                print('record_type', dline.get('record_type', None))
                raise Exception('no step key ' + args.step_key)
            step = dline[args.step_key]
            if maximum_step is not None and step > maximum_step:
                continue
            if units == 'thousands':
                step = step / 1000
            result_row = {
                'step': '%.3f' % step,
            }
            for value_key in value_keys:
                if value_key not in dline:
                    print(dline)
                    print('keys', list(dline.keys()))
                    print('record_type', dline.get('record_type', None))
                    raise Exception('no value key ' + value_key)
                v = dline[value_key]
                if strip_value_underscore:
                    value_key = value_key.replace('_', '')
                result_row[value_key] = '%.3f' % v
            result_rows.append(result_row)
    return result_rows


def run(args):
    meta_by_grammar = {}
    grammars = []
    grammars_set = set()
    meta_by_grammar_by_rmlm = OrderedDict()

    result_csv_by_ref = {}
    for ref in args.refs:
        print(ref)
        with open(f'{args.csv_dir}/{ref}.csv', 'r') as f_in:
            dict_reader = csv.DictReader(f_in)
            result_csv_by_ref[ref] = list(dict_reader)
        df = pd.DataFrame(result_csv_by_ref[ref])
        print(df[['comp_sup_send_steps', 'comp_sup_recv_steps', 'comp_e2e_steps']])
        e2e_logfiles = glob.glob(f'{args.log_dir}/log_{args.script}_{ref}*.log')
        for i, log_filepath in enumerate(e2e_logfiles):
            num_lines = graphing_common.get_num_lines(log_filepath)
            if num_lines <= 3:
                continue
            meta = graphing_common.read_meta(log_filepath)
            meta['num_lines'] = num_lines
            params = Params(meta['params'])
            _ref = params.ref
            if ref not in _ref:
                print(ref, 'not in', _ref)
                continue
            if 'recv' in _ref or 'send' in _ref:
                continue
            meta['log_filepath'] = log_filepath
            meanings = params.meanings
            grammar = params.grammar
            if params.corruptions is not None and params.corruptions != '':
                grammar = params.corruptions
            if grammar not in grammars_set:
                grammars.append(grammar)
                grammars_set.add(grammar)
            if params.send_arch != args.send_arch:
                continue
            if params.recv_arch != args.recv_arch:
                continue
            if meanings != args.meanings:
                continue
            if params.link != args.link:
                continue
            rmlm = f'{ref}_{params.send_arch}_{params.recv_arch}_{params.link}_{meanings}'
            exclude = False
            for filter in args.filters:
                if filter not in rmlm:
                    exclude = True
            if exclude:
                continue
            # check for timeouts
            sup_lines = graphing_common.head(log_filepath, 3).split('\n')[1:3]
            for line in sup_lines:
                d = json.loads(line)
                if d['terminate_reason'] == 'timeout':
                    exclude = True
                    print('excluding for timeout', rmlm, grammar)
            if exclude:
                continue
            if rmlm not in meta_by_grammar_by_rmlm:
                meta_by_grammar_by_rmlm[rmlm] = {}
            meta_by_grammar_by_rmlm[rmlm][grammar] = meta

    # csv_row_by_episode = defaultdict(dict)
    value_keys_no_underscore = [k.replace('_', '') for k in args.value_keys]
    with open(args.out_csv, 'w') as f_out:
        dict_writer = csv.DictWriter(f_out, fieldnames=['grammar', 'step'] + value_keys_no_underscore)
        dict_writer.writeheader()
        for rmlm, meta_by_grammar in meta_by_grammar_by_rmlm.items():
            # for j, value_key in enumerate(args.value_keys):
            for grammar in grammars:
                if grammar not in meta_by_grammar:
                    continue
                meta = meta_by_grammar[grammar]
                log_filepath = meta['log_filepath']
                values = get_values(
                    strip_value_underscore=True, units=args.units,
                    log_filepath=log_filepath, step_key=args.step_key,
                    value_keys=args.value_keys, maximum_step=args.max_step,
                    record_type=args.record_type, skip_record_types=args.skip_record_types)
                for row in values:
                    row['grammar'] = grammar
                    dict_writer.writerow(row)
    print('wrote', args.out_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', type=str, default='e2e_fixpoint_grid')
    parser.add_argument('--refs', type=str, default='ibe099')
    parser.add_argument('--value-keys', type=str, default='e2e_loss,e2e_acc,send_acc,recv_acc')
    parser.add_argument('--filters', type=str, default='')
    parser.add_argument('--step-key', type=str, default='episode')
    parser.add_argument('--record-type', type=str)
    parser.add_argument('--skip-record-types', type=str, default='sup_train_res')
    parser.add_argument('--max-step', type=int, default=15000)
    parser.add_argument('--units', type=str, default='thousands')
    parser.add_argument('--csv-dir', type=str, default='out_csv', help='folder with ibe099.csv etc in')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--out-csv', type=str, required=True)
    parser.add_argument('--send-arch', type=str, default='RNNAutoReg:LSTM')
    parser.add_argument('--recv-arch', type=str, default='RNN:LSTM')
    parser.add_argument('--meanings', type=str, default='5x10')
    parser.add_argument('--link', type=str, default='RL')
    args = parser.parse_args()
    args.refs = args.refs.split(',')
    args.refs = list(itertools.chain(
        *[[ref] if '-' not in ref else string_utils.ref_range_to_refs(ref) for ref in args.refs]))
    args.skip_record_types = args.skip_record_types.split(',')
    args.value_keys = args.value_keys.split(',')
    args.filters = args.filters.split(',')
    run(args)
