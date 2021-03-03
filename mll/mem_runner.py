"""
run multiple mem experiments, somehow
"""
import argparse
import json

from ulfs.params import Params
from ulfs import name_utils
from ulfs import utils

import run_mem_recv
import run_mem_send

# giveup_time_seconds = 300

default_params = {
    'embedding_size': 128,
    'tokens_per_meaning': 4,
    'vocab_size': 4,
    'batch_size': 128,
    'enable_cuda': False,
    'terminate_time': 900,
    'dropout': 0,
    'clip_grad': 5.0,
    'num_layers': 1,
    'opt': 'Adam',
}

scenarios = [
    # {'meanings': '3x10', 'corruptions': None, 'grammar': 'Compositional', 'target': 0.99},
    {'meanings': '3x10', 'corruptions': 'Permute', 'grammar': 'Compositional', 'target': 0.99},
    {'meanings': '3x10', 'corruptions': 'Cumrot', 'grammar': 'Compositional', 'target': 0.99},
    {'meanings': '3x10', 'corruptions': 'Cumrot,Permute', 'grammar': 'Compositional', 'target': 0.99},
    {'meanings': '3x10', 'corruptions': 'ShuffleWords', 'grammar': 'Compositional', 'target': 0.99},
    {'meanings': '3x10', 'corruptions': None, 'grammar': 'Holistic', 'target': 0.99},

    # {'meanings': '5x10', 'corruptions': None, 'grammar': 'Compositional', 'target': 0.99},
    # {'meanings': '5x10', 'corruptions': 'Permute', 'grammar': 'Compositional', 'target': 0.99},
    # {'meanings': '5x10', 'corruptions': 'Cumrot', 'grammar': 'Compositional', 'target': 0.99},
    # {'meanings': '5x10', 'corruptions': 'Cumrot,Permute', 'grammar': 'Compositional', 'target': 0.99},
    # {'meanings': '5x10', 'corruptions': 'ShuffleWords', 'grammar': 'Compositional', 'target': 0.99},
    # {'meanings': '5x10', 'corruptions': None, 'grammar': 'Holistic', 'target': 0.99},
]


def run(directions, model, rnn_type, ref, log_path, dropout):
    for direction in directions.split(','):
        print(direction)
        Runner = {
            'recv': run_mem_recv.Runner,
            'send': run_mem_send.Runner
        }[direction]
        f_log = open(log_path, 'w')
        meta = {
            'directions': directions,
            'model': model,
            'rnn_type': rnn_type,
            'ref': ref,
            'log_path': log_path,
            'dropout': dropout
        }
        f_log.write('meta: ' + json.dumps(meta) + '\n')
        for i, scenario in enumerate(scenarios):
            sub_ref = f'{ref}_{direction}_{scenario["meanings"]}'
            sub_ref += 'C' if scenario['grammar'] == 'Compositional' else 'H'
            if scenario['corruptions'] is not None:
                sub_ref += '_' + scenario['corruptions']
            runner = Runner()

            name = 'mem_runner'
            std_args = {
                'render_every_seconds': 0.01,
                'save_every_seconds': 300,
                'name': name,
                'model_file': 'tmp/{name}_{name}_{hostname}_{date}_{time}.dat',
                'logfile': 'logs/log_{name}_{ref}_{hostname}_{date}_{time}.log',
                'ref': sub_ref
            }
            runner._extract_standard_args(Params(std_args))

            runner.args_parsed = True
            runner.enable_cuda = False
            runner.render_every_seconds = 0.01
            # runner.terminate_time = giveup_time_seconds
            # runner.terminate_acc = scenario['target']
            # runner.setup_base()
            # runner.run_base()
            print(runner)
            num_meaning_types, meanings_per_type = [int(v) for v in scenario['meanings'].split('x')]
            params_dict = {
                'corruptions': scenario['corruptions'],
                'num_meaning_types': num_meaning_types,
                'meanings_per_type': meanings_per_type,
                'grammar': scenario['grammar'],
                'model': model,
                'rnn_type': rnn_type,
                'terminate_acc': scenario['target'],
                'ref': sub_ref,
                'dropout': dropout
            }
            for k, v in default_params.items():
                params_dict[k] = v
            runner.params = Params(params_dict)
            print(runner.params)
            # runner.setup_base()
            runner.setup(runner.params)
            runner.run_base()
            print('res', runner.res)
            runner.res['scenario'] = scenario
            f_log.write(json.dumps(runner.res) + '\n')
            f_log.flush()
            print('')


if __name__ == '__main__':
    utils.clean_argv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--directions', type=str, default='send', help='[recv|send] either or both (comma-separated)')
    parser.add_argument('--model', type=str, default='RNN')
    parser.add_argument('--rnn-type', type=str, default='GRU')
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--log-path', type=str, default='logs/log_{name}_{ref}_{hostname}_{date}_{time}.log')
    parser.add_argument('--dropout', type=float, default=0)
    # parser.add_argument('--corruptions', type=str, default='')
    args = parser.parse_args()
    args.log_path = args.log_path.format(
        name='mem_runner',
        ref=args.ref,
        hostname=name_utils.hostname(),
        date=name_utils.date_string(),
        time=name_utils.time_string()
    )
    run(**args.__dict__)
