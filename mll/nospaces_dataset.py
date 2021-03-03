import torch
import numpy as np
import re
import string
from os.path import expanduser as expand


def string_to_tensor(c2i, s):
    length = len(s)
    t = torch.zeros(length, dtype=torch.int64)
    for j in range(length):
        t[j] = c2i[s[j]]
    return t


def string_list_to_tt(strings_l, c2i, justify='right'):
    N = len(strings_l)

    max_len = np.max([len(s) for s in strings_l]).item()

    utts_t = torch.zeros(max_len, N, dtype=torch.int64)
    lengths_t = torch.zeros(N, dtype=torch.int64)

    for n in range(N):
        this_length = len(strings_l[n])
        lengths_t[n] = this_length
        t = string_to_tensor(c2i=c2i, s=strings_l[n])
        if justify == 'right':
            padding = max_len - this_length
            utts_t[padding:, n] = t
        elif justify == 'left':
            utts_t[:this_length, n] = t
        else:
            raise Exception('justify ' + justify + ' not recognized')
        lengths_t[n] = this_length
    return utts_t, lengths_t


class Dataset(object):
    def __init__(self, in_textfile):
        with open(expand(in_textfile), 'r') as f:
            self.text = f.read()
        self.re_clean_non_letter_space = re.compile('[^a-z ]')

        self.text = self.text.lower().replace('\n', ' ').replace(
            '\t', ' ').replace('.', ' ').replace(',', ' ').replace(
                ':', ' ').replace(';', ' ').replace('!', ' ').replace('?', ' ')
        self.text = self.re_clean_non_letter_space.sub('', self.text)
        last_len = None
        new_len = len(self.text)
        replace_count = 0
        while last_len != new_len:
            replace_count += 1
            last_len = new_len
            self.text = self.text.replace('  ', ' ')
            new_len = len(self.text)
        print('ran', replace_count, 'replacement its')
        print('sample', self.text[:1000])

        self.total_chars_anychar = len(self.text)

        # no terminator character
        self.vocab = string.ascii_lowercase
        self.c2i = {c: i for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def sample(self, batch_size, encode_len, decode_len):
        """
        we'll choose as many words as we can to get to encode_len, without going over,
        and then similarly for decode_len
        """
        positions = torch.from_numpy(np.random.choice(
            self.total_chars_anychar - (encode_len + decode_len) * 2, batch_size, replace=False))
        encode_chars_l = []
        decode_chars_l = []
        encode_words_l = []
        decode_words_l = []
        for n in range(batch_size):
            pos = positions[n]
            while pos != 0 and self.text[pos - 1] != ' ':
                pos += 1
            split_text = self.text[pos:pos + (encode_len + decode_len) * 2].split(' ')

            def eat(split_text, length):
                chars = ''
                i = 0
                words = []
                while True:
                    word = split_text[i]
                    if len(word) + len(chars) > length:
                        break
                    chars += word
                    words.append(word)
                    i += 1
                split_text = split_text[i:]
                return split_text, words, chars

            split_text, encode_words, encode_chars = eat(split_text, encode_len)
            split_text, decode_words, decode_chars = eat(split_text, decode_len)
            encode_chars_l.append(encode_chars)
            decode_chars_l.append(decode_chars)
            encode_words_l.append(encode_words)
            decode_words_l.append(decode_words)

        encode_chars_t, encode_lens_t = string_list_to_tt(c2i=self.c2i, strings_l=encode_chars_l, justify='right')
        decode_chars_t, decode_lens_t = string_list_to_tt(c2i=self.c2i, strings_l=decode_chars_l, justify='left')

        return {
            'encode_words_l': encode_words_l,
            'decode_words_l': decode_words_l,
            'encode_chars_l': encode_chars_l,
            'decode_chars_l': decode_chars_l,
            'encode_chars_t': encode_chars_t,
            'decode_chars_t': decode_chars_t,
            'encode_lens_t': encode_lens_t,
            'decode_lens_t': decode_lens_t
        }
