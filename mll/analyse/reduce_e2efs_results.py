"""
Create results table for e2e from scratch experiments
"""
import os
import argparse
import datetime
from typing import Tuple, Optional

import pandas as pd
import mlflow
import numpy as np

import reduce_common


def get_metric_history(run_id, metric_name, step_divide=1, return_times: bool = False):
    client = mlflow.tracking.MlflowClient()
    history = client.get_metric_history(run_id, metric_name)
    history = [(m.step / step_divide, m.timestamp, m.value) for m in client.get_metric_history(run_id, metric_name)]
    steps, timestamps, values = list(zip(*history))
    if return_times:
        start_time = datetime.datetime.fromtimestamp(timestamps[0] / 1000)
        time_deltas = [datetime.datetime.fromtimestamp(t / 1000) - start_time for t in timestamps]
        times = [t.total_seconds() / 3600 for t in time_deltas]
    returns = [steps, values]
    if return_times:
        returns.append(times)
    return tuple(returns)


def get_step_value(run_id, metric, step: int) -> Tuple[Optional[int], Optional[int]]:
    client = mlflow.tracking.MlflowClient()
    history = client.get_metric_history(run_id, metric)
    for m in history:
        _step, value = m.step, m.value
        if _step >= step:
            return _step, value
    return None, None


def filter_ref_dupes(runs):
    """
    we may have run a ref several times, since first time crashed
    remove these based on start time
    """
    runs = runs.groupby(['send_arch', 'recv_arch', 'link']).head(1)
    return runs


def create_result_table_for_ref(ref, args):
    assert 'MLFLOW_TRACKING_URI' in os.environ

    exp = mlflow.get_experiment_by_name(args.exp_name)
    exp_id = exp.experiment_id

    runs = []
    for repr in args.reprs:
        repr = repr.replace('discr', 'rl')
        _runs = mlflow.search_runs([exp_id], filter_string=f'tags.mlflow.runName like "%{ref}%{repr}"')
        runs.append(_runs)
    runs = pd.concat(runs, ignore_index=True)
    name_mapper = {field: field.split('.')[-1] for field in runs.columns}
    runs = runs.rename(columns=name_mapper)

    runs = filter_ref_dupes(runs)

    runs = runs.set_index(['send_arch', 'recv_arch', 'link'])
    results_rows = []
    for idx, row in runs.iterrows():
        step, e2e_acc = get_step_value(run_id=row.run_id, metric='e2e_acc', step=args.k_steps * 1000)
        step_rho, rho = get_step_value(run_id=row.run_id, metric='rho', step=args.k_steps * 1000)
        result_row = {}
        result_row['send_arch'] = idx[0]
        result_row['recv_arch'] = idx[1]
        result_row['repr'] = idx[2].lower().replace('softmax', 'soft').replace('rl', 'discr').replace('gumbel', 'gumb')
        result_row['e2e_acc'] = e2e_acc
        result_row['rho'] = rho
        if e2e_acc is not None:
            print(step, 'e2e_acc=%.3f' % e2e_acc, 'rho=%.3f' % rho)
        else:
            print(step, 'None', 'None')
        results_rows.append(result_row)
    results = pd.DataFrame(results_rows)[['repr', 'send_arch', 'recv_arch', 'e2e_acc', 'rho']]
    print(results)
    return results


def create_result_table_for_refs(args):
    df_by_ref = {}
    for ref in args.refs:
        df_by_ref[ref] = create_result_table_for_ref(ref=ref, args=args)

    df_mean_str, df_mean, df_counts = reduce_common.average_of(
        list(df_by_ref.values()), show_mean=args.show_mean, show_stderr=args.show_stderr)
    print('mean_str', df_mean_str)
    print('mean', df_mean)
    print(df_counts)

    numeric_fields = [field for field, dtype in zip(
        df_mean.columns, df_mean.dtypes) if dtype in [np.float32, np.int64]]

    if args.show_mean:
        minimize = set(numeric_fields) - set(args.maximize)
        maximize = args.maximize
    else:
        minimize, maximize = [], []

    reduce_common.write_tex(
        df_str=df_mean_str, df_mean=df_mean, filepath=args.out_tex,
        minimize=minimize, maximize=maximize,
        keys=['send_arch', 'recv_arch'],
        titles={'repr': 'Repr', 'send arch': 'Send arch', 'recv arch': 'Recv arch',
                'e2e acc': '$\\acc_{e2e}$', 'rho': '$\\rho$'})

    df_mean_str.to_csv(args.out_csv)
    os.system(f'open {args.out_csv}')

    df_counts.to_csv('counts.csv')
    os.system('open counts.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refs', type=str, required=True)
    parser.add_argument('--reprs', type=str, default='soft,gumb,discr')
    parser.add_argument('--exp-name', type=str, default='hp/ec')
    parser.add_argument('--k-steps', type=int, default=50)
    parser.add_argument('--maximize', type=str, default='e2e acc,rho')
    parser.add_argument('--no-mean', action='store_true')
    parser.add_argument('--no-stderr', action='store_true')
    parser.add_argument('--out-csv', type=str, default='foo.csv')
    parser.add_argument('--out-tex', type=str, default='foo.tex')
    args = parser.parse_args()

    args.reprs = args.reprs.split(',')
    args.refs = args.refs.split(',')
    args.maximize = args.maximize.split(',')
    args.show_stderr = not args.no_stderr
    del args.__dict__['no_stderr']
    args.show_mean = not args.no_mean
    del args.__dict__['no_mean']

    create_result_table_for_refs(args)


if __name__ == '__main__':
    main()
