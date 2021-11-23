import datetime
from mll.turk.webservice import datetime_utils


def test_datetime_utils():
    a = datetime.datetime(1973, 5, 3, 12, 51, 32)
    b = datetime.datetime(1973, 5, 3, 12, 51, 35)
    c = datetime.datetime(1973, 5, 4, 12, 51, 35)

    a_str = datetime_utils.datetime_to_str(a)
    assert datetime_utils.str_to_datetime(a_str) == a

    assert datetime_utils.datetime_diff_seconds(b, a) == 3
    assert datetime_utils.datetime_diff_seconds(c, a) == 3 + 24 * 3600
