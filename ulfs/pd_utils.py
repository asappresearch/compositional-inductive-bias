import pandas as pd
from typing import Sequence, Dict, List
import numpy as np

from ulfs import formatting


def do_pivot(df: pd.DataFrame, row_name: str, col_name: str, metric_name: str):
    """
    Works with df.pivot, except preserves the ordering of the rows and columns
    in the pivoted dataframe
    """
    original_row_indices = df[row_name].unique()
    original_col_indices = df[col_name].unique()
    pivoted = df.pivot(index=row_name, columns=col_name, values=metric_name)
    pivoted = pivoted[original_col_indices]
    pivoted = pivoted.reindex(original_row_indices).reset_index()
    pivoted.columns.name = None
    return pivoted


def average_of(
    dfs: Sequence[pd.DataFrame], key_fields: List[str], show_mean: bool, show_ci95: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Takes in list of dataframes
    for any numeric columns, takes average. other columns
    verifies identical
    """
    sums = dfs[0].copy()
    counts = dfs[0].copy()
    fields = dfs[0].columns
    print('fields', fields)
    types = dfs[0].dtypes
    numeric_fields = [field for field, dtype in zip(fields, types) if dtype in [
        np.float32, np.int64, np.float64, np.int32] and field not in key_fields]
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
    ci95 = dfs[0].copy()
    stderr[numeric_fields] = stddev[numeric_fields] / counts[numeric_fields].pow(0.5)
    ci95[numeric_fields] = stddev[numeric_fields] / counts[numeric_fields].pow(0.5) * 1.96

    average = sums.copy()
    average[numeric_fields] = sums[numeric_fields] / counts[numeric_fields]
    average[numeric_fields] = average[numeric_fields].astype(np.float32)

    average = average.set_index(key_fields)
    stderr = stderr.set_index(key_fields)
    ci95 = ci95.set_index(key_fields)

    averaged_str = dfs[0].copy()
    for numeric_field in numeric_fields:
        averaged_str[numeric_field] = averaged_str[numeric_field].astype(str)
    averaged_str = averaged_str.set_index(key_fields)
    for index, average_row in average.iterrows():
        stderr_row = stderr.loc[index]
        averaged_str_row = averaged_str.loc[index]
        for field in numeric_fields:
            averaged_str_row[field] = formatting.mean_err_to_str(
                average_row[field], stderr_row[field], err_sds=1,
                show_err=False, show_mean=show_mean, show_ci95=show_ci95, na_val='')
    averaged_str = averaged_str.reset_index()
    average = average.reset_index()
    return {
        'averaged_str': averaged_str,
        'mean': average,
        'counts': counts,
        'ci95': ci95
    }
