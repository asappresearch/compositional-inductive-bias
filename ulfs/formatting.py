import math


def mean_stderr_to_str(mean: float, stderr: float, max_sig: int = 3, stderr_sds: int = 2,
                       show_stderr: bool = True, show_mean: bool = True, na_val: str = 'n/a'):
    """
    based on stderr, will round mean and stderr to some precision that looks sensible-ish

    eg instead of 12.234+/-0.432 we'd have
    12.2+/-0.5
    """
    if mean != mean:
        return na_val
    if stderr == 0:
        if isinstance(mean, int):
            ret_str = ''
            if show_mean:
                ret_str += f'{mean}'
            if show_stderr:
                ret_str += f'+/-{stderr}'
            return ret_str
        else:
            ret_str = ''
            if show_mean:
                ret_str += f'{mean:.3f}'
            if show_stderr:
                ret_str += f'+/-{stderr:.3f}'
            return ret_str
    prec_stderr = min(max_sig, - math.floor(math.log10(stderr)) + stderr_sds - 1)
    if prec_stderr >= 0:
        mean_formatted = f'%.{prec_stderr}f' % mean
        stderr_formatted = f'%.{prec_stderr}f' % stderr
    else:
        multiplier = round(math.pow(10, - prec_stderr))
        mean_formatted = str(int(mean / multiplier) * multiplier)
        stderr_formatted = str(int(stderr / multiplier) * multiplier)
    ret_str = ''
    if show_mean:
        ret_str += mean_formatted
    if show_stderr:
        ret_str += f'+/-{stderr_formatted}'
    return ret_str
