"""
run multiple e2e experiments

forked from mem_runner_2.py, 2021 feb 03
goal is to run end 2 end in similar fashion than we have been
running single sender or single receiver
so, we will run compositional training, look at how many steps
that takes, then leave e2e running for 10-20x that.

we will have the reference working in this version, not
just use some dummy references...

after running for compositional, we will do the same thing
for each other grammar type that achieves the target accuracy.

each time we put end to end, we will train for 10-20x the
amount that it took to train the compositional supervised
(?)
"""
import argparse
from collections import defaultdict
import csv
import sys
import datetime
import os
from typing import Optional

import mlflow

from ulfs.params import Params
from ulfs import utils, git_info

from mll import mem_defaults

import e2e_fixpoint


default_params = {
    'tokens_per_meaning': 4,
    'vocab_size': 4,
    'batch_size': 128,
    'enable_cuda': False,
    'dropout': mem_defaults.drop,
    'num_layers': 1,
    'clip_grad': mem_defaults.clip_grad,
    'e2e_batch_size': 128,
    'window_noise': False,
    'predict_correct': False,
    'l2': mem_defaults.l2,
    'lr': mem_defaults.lr,
    'normalize_reward_std': True,
    'gumbel_tau': mem_defaults.gumbel_tau,
    'gumbel_hard': True,
}

std_args = {
    'render_every_seconds': mem_defaults.render_every_seconds,
    'save_every_seconds': 300,
    'model_file': 'tmp/{name}_{ref}_{hostname}_{date}_{time}.dat',
    'logfile': 'logs/log_{name}_{ref}_{hostname}_{date}_{time}.log',
    'load_last_model': None,
    'render_every_steps': 50,
    # 'name': 'mem_runner_e2e',
}


def run_single(
        args,
        arch_pair: str, grammar: str,
        train_acc: float,
        max_sup_steps: Optional[int] = None,
        max_e2e_steps: Optional[int] = None,
        max_sup_mins: Optional[int] = None,
        max_e2e_mins: Optional[int] = None,
        max_e2e_ratio: Optional[float] = None):
    mode: str
    rnn_type: Optional[str]
    if grammar == 'Holistic':
        grammar_family = 'Holistic'
        corruption = None
    elif grammar == 'Comp':
        grammar_family = 'Compositional'
        corruption = None
    else:
        corruption = grammar
        grammar_family = 'Compositional'
    send_arch, recv_arch = arch_pair.split('+')
    Runner = e2e_fixpoint.Runner
    runner = Runner()
    std_args['ref'] = 'grid_' + args.ref + '_' + send_arch.replace(':', '') + '_' + \
        recv_arch.replace(':', '') + '_' + grammar
    std_args['name'] = 'e2e_fixpoint'
    print('std_args', std_args)
    runner._extract_standard_args(Params(std_args))
    print('runner.logfile', runner.logfile)
    runner.args_parsed = True
    runner.enable_cuda = default_params['enable_cuda']
    runner.render_every_seconds = std_args['render_every_seconds']
    runner.render_every_steps = std_args['render_every_steps']
    params_dict = {
        'meanings': args.meanings,
        'corruptions': corruption,
        'grammar': grammar_family,
        'max_sup_steps': max_sup_steps,
        'max_e2e_ratio': max_e2e_ratio,
        'max_e2e_steps': max_e2e_steps,
        'max_mins': None,
        'train_acc': train_acc,
        'send_arch': send_arch,
        'recv_arch': recv_arch,
        'max_e2e_mins': max_e2e_mins,
        'max_sup_mins': max_sup_mins,
        # 'ref': args.ref,
        'seed': args.seed,
        'ref': std_args['ref'],
        'link': args.link,
        'opt': args.opt,
        'rl_reward': args.rl_reward,
        'send_ent_reg': args.send_ent_reg,
        'recv_ent_reg': args.recv_ent_reg,
        'softmax_sup': args.softmax_sup,
        'gumbel_sup': args.gumbel_sup,
        # 'ref': args.ref + '-' + send_arch.replace(':', '') + '-' + recv_arch.replace(':', '') + '-' + grammar
    }
    params_dict.update(default_params)
    runner.params = Params(params_dict)
    print('runner.params', runner.params)
    runner.args_parsed = True
    runner.setup_base(delay_mlflow=True)
    runner.run_base()
    res = runner.res
    return res


def run(args):
    results_by_arch_pair = defaultdict(list)
    result_grid = []
    for arch_pair in args.arch_pairs:
        results = results_by_arch_pair[arch_pair]
        print(arch_pair)
        arch_result = {}
        res = run_single(
            arch_pair=arch_pair, args=args, grammar='Comp',
            train_acc=args.train_acc, max_e2e_ratio=args.max_e2e_ratio,
            max_e2e_steps=args.max_e2e_steps,
            max_e2e_mins=args.max_e2e_comp_mins,
            max_sup_mins=args.max_sup_comp_mins)
        print('res', res)
        comp_send_steps = res['sup_send_steps']
        comp_recv_steps = res['sup_recv_steps']
        comp_e2e_steps = res['e2e_steps']
        comp_max_sup_steps = max(comp_send_steps, comp_recv_steps)
        max_sup_steps = int(comp_max_sup_steps * args.max_sup_ratio) if args.max_sup_ratio is not None else None
        max_e2e_steps = int(comp_max_sup_steps * args.max_e2e_ratio) if args.max_e2e_ratio is not None else None
        if args.max_e2e_steps is not None:
            if max_e2e_steps is None:
                # max_e2e_steps = 0
                max_e2e_steps = args.max_e2e_steps
            else:
                max_e2e_steps = min(max_e2e_steps, args.max_e2e_steps)
        print('max_e2e_steps', max_e2e_steps)
        print('comp send steps', comp_send_steps, 'comp_recv_steps', comp_recv_steps, 'comp_e2e_steps', comp_e2e_steps)
        # print('params', res['num_parameters'])
        results.append(('Compositional', res['e2e_send_acc'], res['e2e_recv_acc']))
        arch_result['send_arch'] = arch_pair.split('+')[0]
        arch_result['recv_arch'] = arch_pair.split('+')[1]
        arch_result['send_params'] = res['send_num_parameters']
        arch_result['recv_params'] = res['recv_num_parameters']
        arch_result['comp_sup_send_acc'] = '%.3f' % res['sup_send_acc']
        arch_result['comp_sup_send_steps'] = res['sup_send_steps']
        arch_result['comp_sup_recv_acc'] = '%.3f' % res['sup_recv_acc']
        arch_result['comp_sup_recv_steps'] = res['sup_recv_steps']
        arch_result['comp_e2e_acc'] = '%.3f' % res['e2e_acc']
        arch_result['comp_e2e_steps'] = res['e2e_steps']
        arch_result['comp_e2e_send_acc'] = '%.3f' % res['e2e_send_acc']
        arch_result['comp_e2e_recv_acc'] = '%.3f' % res['e2e_recv_acc']
        arch_result['comp_send_log'] = res['sup_send_log']
        arch_result['comp_recv_log'] = res['sup_recv_log']
        arch_result['comp_e2e_log'] = res['e2e_log']
        fieldnames = [
            'send_arch', 'recv_arch', 'send_params', 'recv_params',
            'comp_sup_send_steps', 'comp_sup_recv_steps', 'comp_sup_send_acc', 'comp_sup_recv_acc',
            'comp_e2e_steps', 'comp_e2e_acc', 'comp_e2e_send_acc', 'comp_e2e_recv_acc',
            'comp_send_log', 'comp_recv_log', 'comp_e2e_log'
        ]
        for grammar in args.grammars:
            res = run_single(arch_pair=arch_pair, args=args, grammar=grammar,
                             train_acc=args.train_acc, max_e2e_steps=max_e2e_steps, max_sup_steps=max_sup_steps,
                             max_sup_mins=args.max_sup_mins)
            print(grammar)
            results.append((grammar, res['e2e_send_acc'], res['e2e_recv_acc']))
            arch_result[f'{grammar}_sup_send_acc'] = '%.3f' % res['sup_send_acc']
            arch_result[f'{grammar}_sup_send_steps'] = res['sup_send_steps']
            arch_result[f'{grammar}_sup_recv_acc'] = '%.3f' % res['sup_recv_acc']
            arch_result[f'{grammar}_sup_recv_steps'] = res['sup_recv_steps']
            arch_result[f'{grammar}_e2e_send_acc'] = '%.3f' % res['e2e_send_acc']
            arch_result[f'{grammar}_e2e_recv_acc'] = '%.3f' % res['e2e_recv_acc']
            arch_result[f'{grammar}_e2e_acc'] = '%.3f' % res['e2e_acc']
            arch_result[f'{grammar}_e2e_steps'] = res['e2e_steps']
            arch_result[f'{grammar}_send_log'] = res['sup_send_log']
            arch_result[f'{grammar}_recv_log'] = res['sup_recv_log']
            arch_result[f'{grammar}_e2e_log'] = res['e2e_log']
            for k in ['sup_send_acc', 'sup_send_steps', 'sup_recv_acc', 'sup_recv_steps',
                      'e2e_acc', 'e2e_steps', 'e2e_send_acc', 'e2e_recv_acc',
                      'send_log', 'recv_log', 'e2e_log']:
                fieldnames.append(f'{grammar}_{k}')
            # print(arch)
            for r in results:
                print(r)
        result_grid.append(arch_result)
        print('====================')
        for arch, results in results_by_arch_pair.items():
            print(arch)
            for r in results:
                print(r)
        print('====================')
        with open(args.out_csv, 'w') as f_out:
            dict_writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            dict_writer.writeheader()
            for r in result_grid:
                dict_writer.writerow(r)
        print('wrote to', args.out_csv)

    if 'MLFLOW_TRACKING_URI' in os.environ and os.environ['MLFLOW_TRACKING_URI'] != '':
        mlflow.set_experiment('hp/ec')
        mlflow.start_run(run_name=args.ref + '_summary')

        with open(args.out_csv, 'r') as f_in:
            csv_text = f_in.read()
        mlflow.log_text(csv_text, f'{args.ref}.csv')

        meta = {}
        meta['params'] = args.__dict__
        meta['argv'] = sys.argv
        meta['hostname'] = os.uname().nodename
        meta['gitlog'] = git_info.get_git_log()
        meta['gitdiff'] = git_info.get_git_diff()
        meta['start_datetime'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        meta['argv'] = ' '.join(meta['argv'])
        gitdiff = meta['gitdiff']
        gitlog = meta['gitlog']
        mlflow.log_text(gitdiff, 'gitdiff.txt')
        mlflow.log_text(gitlog, 'gitlog.txt')

        params_to_log = dict(meta['params'])
        arch_pairs = params_to_log['arch_pairs']
        del params_to_log['arch_pairs']
        mlflow.log_text('\n'.join(arch_pairs), 'arch_pairs.txt')

        mlflow.log_params(params_to_log)

        del meta['gitdiff']
        del meta['gitlog']
        del meta['params']
        mlflow.log_params(meta)
        mlflow.set_tags(meta)

        mlflow.end_run()


def parse_args():
    utils.clean_argv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True, help='experiment ref')
    parser.add_argument('--out-csv', type=str, default='{ref}.csv', help='where to output the results grid')
    parser.add_argument('-m', '--meanings', type=str, default='5x10', help='meaning space to use')
    parser.add_argument('--train-acc', type=float, default=0.95, help='when to stop supervised training')
    parser.add_argument('--seed', type=int, default=123, help='seed for lang and model')
    parser.add_argument('--embedding-size', type=int, default=mem_defaults.embedding_size)
    # parser.add_argument('--grammars', type=str, default='Permute,Cumrot,ShuffleWordsDet,WordPairSums')
    parser.add_argument('--grammars', type=str, default='Permute,RandomProj,Cumrot,ShuffleWordsDet')
    parser.add_argument(
        '--arch-pairs', type=str,
        default=(
            'RNNAutoReg:dgsend+RNN:dgrecv,'
            'RNNAutoReg:RNN+RNN:RNN,RNNAutoReg:GRU+RNN:GRU,'
            'RNNAutoReg:LSTM+RNN:LSTM,'
            'HierZero:RNN+Hier:RNN,HierZero:GRU+Hier:GRU,'
            'HierZero:LSTM+Hier:LSTM,'
            'RNNZero2L:RNN+RNN2L:RNN,RNNZero2L:GRU+RNN2L:GRU,RNNZero2L:LSTM+RNN2L:LSTM,'
            'RNNZero:RNN+RNN:RNN,RNNZero:GRU+RNN:GRU,'
            'RNNZero:LSTM+RNN:LSTM,'
            'FC1L+FC'
        )
    )
    parser.add_argument('--max-e2e-ratio', type=float, default=20, help='how much to multiply comp steps for cutoff')
    parser.add_argument('--max-sup-ratio', type=float, default=20, help='how much to multiply comp steps for cutoff')
    parser.add_argument('--max-e2e-steps', type=int, help='after how many steps to stop e2e')
    parser.add_argument('--max-e2e-comp-mins', type=int, default=60,
                        help='after how many minutes terminate compositional baseline and abort')
    parser.add_argument('--max-sup-comp-mins', type=int, default=60,
                        help='after how many minutes terminate compositional baseline and abort')
    parser.add_argument('--max-sup-mins', type=int, default=60,
                        help='after how many minutes terminate any supervised training and abort that particular '
                             'grammar')
    parser.add_argument('--softmax-sup', action='store_true')
    parser.add_argument('--gumbel-sup', action='store_true')
    parser.add_argument('--opt', type=str, default=mem_defaults.opt)
    parser.add_argument('--link', type=str, default='Softmax', help='[Softmax|Gumbel|RL]')
    parser.add_argument('--rl-reward', type=str, default='count_meanings', help=['ce|count_meanings]'])
    parser.add_argument('--send-ent-reg', type=float, default=mem_defaults.send_ent_reg)
    parser.add_argument('--recv-ent-reg', type=float, default=mem_defaults.recv_ent_reg)
    args = parser.parse_args()

    if args.grammars == 'None':
        args.grammars = []
    else:
        args.grammars = [g.replace('+', ',') for g in args.grammars.split(',')]
    args.arch_pairs = args.arch_pairs.split(',')
    args.out_csv = args.out_csv.format(**args.__dict__)

    if args.max_sup_ratio == 0:
        args.max_sup_ratio = None
    if args.max_e2e_ratio == 0:
        args.max_sup_ratio = None

    run(args)


if __name__ == '__main__':
    parse_args()
