import datetime


def datetime_diff_seconds(later: datetime.datetime, earlier: datetime.datetime):
    return int((later - earlier).total_seconds())


def datetime_to_str(_datetime: datetime.datetime):
    return _datetime.strftime('%Y%m%d%H%M%S')


def str_to_datetime(_str: str):
    return datetime.datetime.strptime(_str, '%Y%m%d%H%M%S')
