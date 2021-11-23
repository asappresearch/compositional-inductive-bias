"""
written from scratch, not forked from graph_example_training_curves.py
input:
- many runs over different seeds, for different grammars, same send/recv architecture
output:
- tall csv, ready for pgfplots
"""
import argparse
import itertools
import glob
import seaborn as sns
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ulfs import graphing, string_utils, pd_utils


plt.rc('font', family='serif')


def aggregate_series(file: str, value_keys: List[str], num_bins: int) -> pd.DataFrame:
    """
    we assume we are given a single file, and have to get the values from that file,
    for each of hte value keys, and aggregate over the bins

    returns a dataframe with columns 'step', 'value_key', 'value'
    where there are multiple rows for  each step, one for each value_key
    """
    step_min, step_max = -1, -1
    steps, values_by_key = graphing.get_log_results_multi(
        logfile=file, step_key='episode', skip_record_types=['sup_train_res'], value_keys=value_keys,
    )
    _min, _max = min(steps), max(steps)
    if step_min is -1 or _min < step_min:
        step_min = _min
    if step_max is -1 or _max < step_max:
        step_max = _max
    bins = np.linspace(step_min + (step_max - step_min) / num_bins, step_max, num_bins)

    # steps_all = []
    # values_all = []
    # value_keys_all = []
    # bin_numbers_all = []
    df = None
    for value_key, values in values_by_key.items():
        means, edges, bin_numbers = stats.binned_statistic(steps, values, bins=bins)
        # steps_all += list(edges)[1:]
        # values_all += list(means)
        # value_keys_all += len(means) * [value_key]
        bin_numbers_uniq = list(dict.fromkeys(bin_numbers))
        if df is None:
            df = pd.DataFrame({'step': list(edges)[1:], 'bin_numbers': list(bin_numbers_uniq)[1:]})
        df[value_key] = list(means)
        # bin_numbers_all += list(bin_numbers_uniq)[1:]
        # print('bin_numbers_uniq', bin_numbers_uniq)
        # print('len(bin_numbers_uniq)', len(bin_numbers_uniq), 'len(means)', len(means), 'len(edges)', len(edges))
    # df = pd.DataFrame({
    #     'step': steps_all, 'value': values_all, 'value_key': value_keys_all, 'bin_numbers': bin_numbers_all})
    return df


def aggregate_data_for_seed(
        log_dir: str, max_files: int, num_bins: int, ref: str, send_arch: str, recv_arch: str):
    """
    we assume a single e2e file for each combination of send_arch/recv_arch/seed
    each ref is assumed to correpond to a different seed. this function only handles a single
    seed
    """
    print(send_arch, recv_arch)
    send_arch_filename = send_arch.replace(':', '')
    recv_arch_filename = recv_arch.replace(':', '')
    file_by_grammar = {}
    grammars_set = set()

    files = glob.glob(f'{log_dir}/log_*_{ref}_{send_arch_filename}_{recv_arch_filename}*.log')
    assert len(files) > 0
    files = [file for file in files if 'recv' not in file and 'send' not in file]
    assert len(files) > 0
    print('got list of files for', ref, len(files))
    if max_files is not None:
        files = files[:max_files]
    grammars = []
    for file in files:
        grammar = file.split(f'{send_arch_filename}_{recv_arch_filename}_')[1].split('_')[0]
        file_by_grammar[grammar] = file
        grammars_set.add(grammar)
    assert len(grammars_set) > 0
    grammars = sorted(list(grammars_set))

    grammar_display = {
        'Comp': 'comp',
        'Cumrot': 'rot',
        'Permute': 'perm',
        'RandomProj': 'proj',
        'ShuffleWordsDet': 'shufdet'
    }
    df_l = []
    for grammar in grammars:
        file = file_by_grammar[grammar]
        print(grammar)
        df = aggregate_series(
            num_bins=num_bins, file=file,
            value_keys=['e2e_loss', 'e2e_acc', 'send_acc', 'recv_acc'])
        df['grammar'] = grammar_display.get(grammar, grammar)
        df['ref'] = ref
        df_l.append(df)
    df = pd.concat(df_l, ignore_index=True)
    print(df)
    return df


def aggregate_data(
    log_dir: str, max_files: int, num_bins: int, refs: List[str], send_arch: str, recv_arch: str
):
    """
    we assume a single e2e file for each combination of send_arch/recv_arch/seed
    each ref is assumed to correpond to a different seed.
    """
    value_keys = ['e2e_loss', 'e2e_acc', 'send_acc', 'recv_acc']
    df_by_ref = {}
    for ref in refs:
        print('ref', ref)
        df = aggregate_data_for_seed(
            ref=ref, max_files=max_files, log_dir=log_dir,
            send_arch=send_arch, recv_arch=recv_arch, num_bins=num_bins)
        df_by_ref[ref] = df
    for ref, df in df_by_ref.items():
        print(ref)
        print(df)
    if args.out_png is not None:
        df_all = pd.concat(df_by_ref.values())
        print('df_all', df_all)
        print('df_all.index.names', df_all.index.names)
        df_all = df_all.reset_index()
        print('df_all.index.names', df_all.index.names)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i, value_key in enumerate(['e2e_loss', 'e2e_acc', 'send_acc', 'recv_acc']):
            sns.lineplot(ax=axes[i], data=df_all, x='bin_numbers', y=value_key, hue='grammar')
        plt.savefig(args.out_png)

    if args.out_csv is not None:
        key_fields = ['grammar', 'bin_numbers']
        mean, counts, ci95 = map(pd_utils.average_of(
            dfs=list(df_by_ref.values()),
            key_fields=key_fields,
            show_mean=True,
            show_ci95=False
        ).__getitem__, ['mean', 'counts', 'ci95'])
        print('mean', mean)
        print('counts', counts)
        print('ci95', ci95)
        print('mean key fields', mean.index.names)
        print('ci95 key fields', ci95.index.names)
        mean = mean.set_index(ci95.index.names)
        print('mean key fields', mean.index.names)

        for value_key in value_keys:
            mean[f'{value_key}_ci95'] = ci95[value_key]
        mean.drop('ref', 1, inplace=True)
        print(mean)
        mean.to_csv(args.out_csv)


def run(args):
    print(args.send_arch, args.recv_arch)
    aggregate_data(
        refs=args.in_refs, max_files=args.max_files, log_dir=args.log_dir,
        send_arch=args.send_arch, recv_arch=args.recv_arch, num_bins=args.num_bins)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--max-files', type=int, help='for dev/debug mostly')
    parser.add_argument('--in-refs', type=str, required=True, help='comma-separated, can by hyphenated range')
    parser.add_argument('--send-arch', type=str, default='RNNAutoReg:LSTM')
    parser.add_argument('--recv-arch', type=str, default='RNN:LSTM')
    parser.add_argument('--num-bins', type=int, default=50)
    parser.add_argument('--out-csv', type=str)
    parser.add_argument('--out-png', type=str)
    args = parser.parse_args()
    assert args.out_png or args.out_csv
    args.in_refs = args.in_refs.split(',')
    args.in_refs = list(itertools.chain(
        *[[ref] if '-' not in ref else string_utils.ref_range_to_refs(ref) for ref in args.in_refs]))
    run(args)
