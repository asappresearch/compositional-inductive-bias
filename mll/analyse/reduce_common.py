import os
from collections import defaultdict
from typing import Dict, Sequence, Tuple, Iterable, Set

import pandas as pd
import numpy as np

from ulfs import formatting


name_mapping = {
    'Comp': 'comp',
    'RandomProj': 'proj',
    'WordPairSums': 'pairsum',
    'Permute': 'perm',
    'Cumrot': 'rot',
    'Cumrot,Permute': 'rot+perm',
    'ShuffleWords': 'shuf',
    'ShuffleWordsDet': 'shufdet',
    'Holistic': 'hol',
    'arch': 'Model'
}


latex_defines = """
\\def\\acc{\\text{acc}}
"""


def average_of(dfs: Sequence[pd.DataFrame], show_stderr: bool, show_mean: bool, show_ci95: bool = False) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Takes in list of dataframes
    for any numeric columns, takes average. other columns
    verifies identical
    """
    assert not show_stderr or not show_ci95
    sums = dfs[0].copy()
    counts = dfs[0].copy()
    fields = dfs[0].columns
    types = dfs[0].dtypes
    key_fields = [field for field, dtype in zip(fields, types) if dtype not in [
        np.float32, np.int64, np.float64, np.int32]]
    numeric_fields = [field for field, dtype in zip(fields, types) if dtype in [
        np.float32, np.int64, np.float64, np.int32]]
    sums[numeric_fields] = 0.0
    counts[numeric_fields] = 0

    np_values_l = []
    for df in dfs:
        sums[numeric_fields] = sums[numeric_fields] + df[numeric_fields].fillna(0)
        counts[numeric_fields] = counts[numeric_fields] + df[numeric_fields].notnull().astype('int')
        values = df[numeric_fields].values
        np_values_l.append(values)
    np_values = np.stack(np_values_l)
    np_stddev = np.nanstd(np_values, axis=0)
    stddev = dfs[0].copy()
    stddev[numeric_fields] = np_stddev
    stderr = dfs[0].copy()
    stderr[numeric_fields] = stddev[numeric_fields] / counts[numeric_fields].pow(0.5)

    average = sums.copy()
    average[numeric_fields] = sums[numeric_fields] / counts[numeric_fields]
    average[numeric_fields] = average[numeric_fields].astype(np.float32)

    average = average.set_index(key_fields)
    stderr = stderr.set_index(key_fields)

    averaged_str = dfs[0].copy()
    averaged_str = averaged_str.set_index(key_fields)
    averaged_str[numeric_fields] = averaged_str[numeric_fields].astype(str)
    for index, average_row in average.iterrows():
        stderr_row = stderr.loc[index]
        averaged_str_row = averaged_str.loc[index]
        for field in numeric_fields:
            averaged_str_row[field] = formatting.mean_err_to_str(
                average_row[field], stderr_row[field], err_sds=1,
                show_err=show_stderr, show_mean=show_mean, show_ci95=show_ci95, na_val='')
    averaged_str = averaged_str.reset_index()
    average = average.reset_index()
    return averaged_str, average, counts


def get_best_archs_by_field_by_repr(
        df_str: pd.DataFrame, df_mean: pd.DataFrame, maximize: Iterable[str], minimize: Iterable[str],
        keys: Iterable[str]):
    best_archs_by_field_by_repr: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    reprs = df_str.repr.unique()
    maximize = [field for field in maximize if field in df_str.columns]
    minimize = [field for field in minimize if field in df_str.columns]
    for repr in reprs:
        group_str = df_str[df_str['repr'] == repr].reset_index()
        group_mean = df_mean[df_str['repr'] == repr].reset_index()
        for field in list(maximize) + list(minimize):
            if field in maximize:
                best_index = group_mean[field].argmax()
            else:
                best_index = group_mean[field].argmin()
            best_value = group_str[field][best_index]
            matches_value = group_str[field] == best_value
            best_arches = set()
            for i, matches in enumerate(matches_value):
                if matches:
                    row = df_str.iloc[i]
                    key_str = '+'.join([row[key] for key in keys])
                    best_arches.add(key_str)
            best_archs_by_field_by_repr[repr][field] = best_arches
    return best_archs_by_field_by_repr


def write_tex(
        df_str: pd.DataFrame, df_mean: pd.DataFrame, filepath: str,
        maximize: Iterable[str], minimize: Iterable[str], keys: Iterable[str],
        longtable: bool = False, titles: Dict[str, str] = {}) -> None:
    df_str = df_str.reset_index()
    df_mean = df_mean.reset_index()

    maximize = set(maximize)
    minimize = set(minimize)

    best_archs_by_field_by_repr = get_best_archs_by_field_by_repr(
        df_str=df_str, df_mean=df_mean, maximize=maximize, minimize=minimize, keys=keys)

    fieldnames = list(df_str.columns)
    if 'index' in fieldnames:
        fieldnames.remove('index')

    with open(filepath, 'w') as f:
        f.write("""
\\documentclass[11pt,a4paper]{{article}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
""".format() + latex_defines)
        if longtable:
            f.write("""
\\usepackage{{longtable}}""".format())
        f.write("""
\\begin{{document}}
""".format())

        if longtable:
            f.write("""
{{
\\small
\\begin{{longtable}}[l]{{ {alignment} }}
""".format(alignment='l' * len(fieldnames)))
        else:
            f.write("""
\\begin{{table*}}[htb!]
\\small
\\centering
    \\begin{{tabular}}{{ {alignment} }}
""".format(alignment='l' * len(fieldnames)))

        f.write("""
    \\toprule
""".format())

        row_l = []
        for field in fieldnames:
            field = field.replace('_', ' ')
            field_str = titles.get(field, field)
            if field in maximize:
                field_str += ' $\\uparrow$'
            elif field in minimize:
                field_str += ' $\\downarrow$'
            row_l.append(field_str)
        f.write(' & '.join(row_l) + ' \\\\ \n')
        f.write('\\midrule \n')
        last_repr = ''
        for n, row in enumerate(df_str.to_dict('records')):
            row_l = []
            if row['repr'] != last_repr and n > 0:
                f.write('\\midrule \n')
            for field in fieldnames:
                if field.lower() == 'repr':
                    if row[field] != last_repr:
                        last_repr = row[field]
                    else:
                        row_l.append(' ')
                        continue
                value_str = str(row[field])
                if field.lower() == 'repr':
                    value_str = f'\\textsc{{{value_str}}}'
                key_str = '+'.join([row[key] for key in keys])
                if field in best_archs_by_field_by_repr[last_repr] and key_str in best_archs_by_field_by_repr[
                        last_repr][field]:
                    value_str = f'\\textbf{{{value_str}}}'
                row_l.append(value_str)
            f.write(' & '.join(row_l) + ' \\\\ \n')
        f.write("""
\\bottomrule
""".format())

        if not longtable:
            f.write("""
\\end{{tabular}}""".format())

        f.write("""
\\caption{{Learning architectures}}
\\label{{tab:learning_architectures}}""".format())

        if longtable:
            f.write("""
\\end{{longtable}}
}}""".format())
        else:
            f.write("""
\\end{{table*}}""".format())

        f.write("""
\\end{{document}}
""".format())

    os.system(f'open {filepath}')
