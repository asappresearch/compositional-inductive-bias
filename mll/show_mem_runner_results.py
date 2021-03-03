"""
Takes a log file from mem_runner.py, and displays in an easy to use format
"""
import argparse
import sys
import os
from collections import defaultdict
from os import path
from os.path import join
import json
import math

import numpy as np


def run(logfile):
    with open(logfile, 'r') as f:
        contents = f.read().split('\n')
    res_by_grammar_by_meanings = defaultdict(dict)
    sps_by_meanings = defaultdict(list)
    parameters_by_meanings = {}
    for line in contents:
        if line.strip() == '':
            continue
        if line.startswith('meta:'):
            meta = json.loads(line.replace('meta:', ''))
            # print('meta', meta)
            continue
        # print('line', line)
        d = json.loads(line)
        # print('d', d)
        direction = 'recv' if '_recv_' in d['params']['ref'] else 'send'
        num_meaning_types = d['params']['num_meaning_types']
        meanings_per_type = d['params']['meanings_per_type']
        meanings = f'{num_meaning_types}x{meanings_per_type}'
        grammar = d['params']['grammar']
        corruptions = d['params']['corruptions']
        acc = d['log_dict']['train_acc']
        episode = d['log_dict']['episode']
        elapsed_time = d['elapsed_time']
        if meanings not in parameters_by_meanings:
            parameters_by_meanings[meanings] = d['num_parameters']
        sps_by_meanings[meanings].append(episode / elapsed_time)
        # print(
        #     meanings, grammar, corruptions, 'e=%i' % episode, 'numparams=%i' % d['num_parameters'],
        #     'eps=%i' % int(episode // elapsed_time),
        #     'acc=%.3f' % acc
        # )
        if corruptions is not None:
            grammar = corruptions
        res_str = f'{episode}({acc:.3f})'
        if d['terminate_reason'] == 'timeout':
            res_str = '>' + res_str
        res_by_grammar_by_meanings[meanings][grammar] = res_str
        # res_by_grammar_by_meanings[meanings]['num_parameter'] = res_str

    direction_str = 'SEND' if direction == 'send' else 'RECEIVE'
    print('')
    print(direction_str)
    print('')

    headers = ['parameters', 'sps', 'Compositional', 'permute', 'cumrot', 'cumrot,permute', 'shuffle_words', 'Holistic']
    print('meanings\t' + '\t'.join(headers))
    for meanings in ['3x10', '5x10']:
        # res_line = ''
        if meanings not in res_by_grammar_by_meanings:
            continue
        res_by_grammar_by_meanings[meanings]['parameters'] = '%i' % parameters_by_meanings[meanings]
        res_by_grammar_by_meanings[meanings]['sps'] = '%i' % int(np.mean(sps_by_meanings[meanings]).item())

        res_l = []
        # print(meanings)
        # print(res_by_grammar_by_meanings[meanings])
        for header in headers:
            # res_l.append('%i' % res_by_grammar_by_meanings[meanings].get(grammar, -1))
            res_l.append(res_by_grammar_by_meanings[meanings].get(header, '-'))
        print(meanings + '\t' + '\t'.join(res_l))
    # for meanings, res_by_grammar in res_by_grammar_by_meanings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str, required=True)
    args = parser.parse_args()
    run(**args.__dict__)
