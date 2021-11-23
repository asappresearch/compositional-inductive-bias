import argparse
import math
import statistics
from collections import defaultdict
import pandas as pd
import os

import sqlite3

from ulfs import formatting, tex_utils


g_sql_get_eng_results = """
select replace(replace(replace(task_id, 'eng2', 'eng'), '_pnts', ''), 'eng_', '') as task_id,
       (cast(num_holdout_correct as float) / 3) as acc_holdout,
       score,
       (cast(duration_seconds as float) / 60) as duration_minutes
from game_instance
where status ='COMPLETE' and
        task_id like 'eng_pnts_%' and
        start_datetime >= '20211119140000' and
        score >= 50
order by task_id
"""

g_sql_get_synth_results = """
select replace(replace(replace(task_id, '_autoinc_ho', ''), 'pnts_', ''), '3', '') as task_id,
    (cast(num_holdout_correct as float) / 3) as acc_holdout,
    score,
    (cast(duration_seconds as float) / 60) as duration_minutes
from game_instance
where status ='COMPLETE' and
        task_id not like 'eng%' and
        (task_id like 'pnts_%') and
        (start_datetime >= '20211119140000' or start_datetime <= '20211119010000') and
        score >= 50
order by task_id
"""


def calc_avg_ci95(rows, column_name: str):
    values = [row[column_name] for row in rows]
    N = len(values)
    average = sum(values) / N
    ci95 = statistics.stdev(values) * 1.96 / math.sqrt(N)
    # print(values)
    return average, ci95


def run1(args):
    con = sqlite3.connect(args.in_db)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    rows_by_task_id = defaultdict(list)
    sql = g_sql_get_eng_results if args.eng else g_sql_get_synth_results
    for row in cur.execute(sql):
        print(dict(row))
        rows_by_task_id[row['task_id']].append(row)
    agg_rows = []
    agg_str_rows = []
    for task_id, rows in rows_by_task_id.items():
        agg_row = {'task_id': task_id}
        agg_str_row = {'task_id': task_id}
        for c in [
                'duration_minutes',
                'score',
                'acc_holdout']:
            average, ci95 = calc_avg_ci95(rows, c)
            agg_row[c] = average
            agg_row[f'{c}_ci95'] = ci95
            agg_str_row[c] = '$' + formatting.mean_ci95to_str(
                mean=average, ci95=ci95, ci95_sds=1).replace('+/-', '\\pm') + '$'
        print(agg_str_row)
        agg_rows.append(agg_row)
        agg_str_rows.append(agg_str_row)
    df = pd.DataFrame(agg_str_rows)
    print(df)
    titles = {
        'task id': 'Grammar',
        'score': 'score',
        'acc holdout': '$\\text{acc}_{holdout}$',
        'duration minutes': '$t$ (mins)',
    }
    dataset_name = 'eng' if args.eng else 'synth'
    tex_utils.write_tex(
        df_str=df, filepath=args.out_tex, titles=titles, label=f'tab:turk_{dataset_name}',
        caption=f'Human evaluation results, {dataset_name} dataset')
    os.system(f'open {args.out_tex}')


def run2(args):
    con = sqlite3.connect(args.in_db)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    agg_str_rows_body = []
    for (eng, dataset_name) in [(False, 'synth'), (True, 'eng')]:
        rows_by_task_id = defaultdict(list)
        sql = g_sql_get_eng_results if eng else g_sql_get_synth_results
        for row in cur.execute(sql):
            print(dict(row))
            rows_by_task_id[row['task_id']].append(row)
        # agg_rows = []
        agg_str_rows_appendix = []
        for task_id, rows in rows_by_task_id.items():
            # agg_row = {'dataset_name': dataset_name, 'task_id': task_id}
            agg_str_row_body = {
                'dataset': '\\textsc{' + dataset_name + '}',
                'task_id': '\\textsc{' + task_id + '}'
            }
            agg_str_row_appendix = {
                'task_id': '\\textsc{' + task_id + '}'
            }
            agg_str_row_appendix['N'] = len(rows)
            for c in [
                    'duration_minutes',
                    'score',
                    'acc_holdout']:
                average, ci95 = calc_avg_ci95(rows, c)
                # agg_row[c] = average
                # agg_row[f'{c}_ci95'] = ci95
                agg_str_row_appendix[c] = '$' + formatting.mean_ci95to_str(
                    mean=average, ci95=ci95, ci95_sds=1).replace('+/-', '\\pm') + '$'
            for c in [
                    # 'duration_minutes',
                    # 'score',
                    'acc_holdout']:
                average, ci95 = calc_avg_ci95(rows, c)
                # agg_row[c] = average
                # agg_row[f'{c}_ci95'] = ci95
                agg_str_row_body[c] = '$' + formatting.mean_ci95to_str(
                    mean=average, ci95=ci95, ci95_sds=1).replace('+/-', '\\pm') + '$'
            print(agg_str_row_body)
            # agg_rows.append(agg_row)
            agg_str_rows_body.append(agg_str_row_body)
            agg_str_rows_appendix.append(agg_str_row_appendix)
        df = pd.DataFrame(agg_str_rows_appendix)
        titles = {
            'task id': 'Grammar',
            'score': 'score',
            'acc holdout': '$\\text{acc}_{holdout}$',
            'duration minutes': '$t$ (mins)',
            'n': '$N$'
        }
        tex_utils.write_tex(
            df_str=df, filepath=args.out_tex_appendix.format(dataset_name=dataset_name),
            titles=titles,
            label=f'tab:turk_{dataset_name}',
            caption=f'Human evaluation results, \\textsc{{{dataset_name}}} dataset',
            highlight=[('\\textsc{eng}', '\\textsc{comp{')],
            latex_defines='\\def\\acc{\\text{acc}}'
        )
        os.system(f'open {args.out_tex_body.format(dataset_name=dataset_name)}')
    df = pd.DataFrame(agg_str_rows_body)
    print(df)
    original_row_indices = df['dataset'].unique()
    original_col_indices = df['task_id'].unique()
    pivoted = df.pivot(index='dataset', columns='task_id', values='acc_holdout')
    print(pivoted)
    pivoted = pivoted[original_col_indices]
    pivoted = pivoted.reindex(original_row_indices).reset_index()
    pivoted.columns.name = None
    print(pivoted)

    titles = {
        'task id': 'Grammar',
        # 'score': 'score',
        'acc holdout': '$\\text{acc}_{holdout}$',
        # 'duration minutes': '$t$ (mins)',
    }
    # dataset_name = 'eng' if args.eng else 'synth'
    tex_utils.write_tex(
        df_str=pivoted, filepath=args.out_tex_body, titles=titles,
        label='tab:turk',
        caption='Human evaluation results. Values are $\\acc_{holdout}$',
        highlight=[('\\textsc{eng}', '\\textsc{comp{')],
        latex_defines='\\def\\acc{\\text{acc}}'
        )
    os.system(f'open {args.out_tex_body}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-db', default='pull/turk.db')
    # parser.add_argument('--eng', action='store_true')
    parser.add_argument('--out-tex-body', default='/tmp/body.tex')
    parser.add_argument('--out-tex-appendix', default='/tmp/appendix_{dataset_name}.tex')
    args = parser.parse_args()
    run2(args)
