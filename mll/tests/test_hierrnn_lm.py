import time
from mll import nospaces_dataset
import pytest


@pytest.mark.skip()
def test_dataset():
    dataset = nospaces_dataset.Dataset(in_textfile='~/data/stories/tolstoy_warandpeace.txt')
    s = dataset.sample(batch_size=5, encode_len=20, decode_len=20)

    encode_chars_l, decode_chars_l, encode_chars_t, decode_chars_t, encode_lens_t, decode_lens_t = map(s.__getitem__, [
        'encode_chars_l', 'decode_chars_l', 'encode_chars_t', 'decode_chars_t', 'encode_lens_t', 'decode_lens_t'
    ])

    print('s', s)

    start_time = time.time()
    s = dataset.sample(batch_size=128, encode_len=64, decode_len=64)
    print('gen time', time.time() - start_time)
