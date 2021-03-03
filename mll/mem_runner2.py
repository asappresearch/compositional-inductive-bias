"""
run multiple mem experiments, somehow

forked from mem_runner.py, 2021 jan 31
at about this time, we look at number of steps to train compoiional
to 99% accuracy, then run ssame number of steps for other grammars,
and measure resulting training accuracy for each.

(just before this, we were using jupyter/inductive_bias[_recv].ipynb,
but it kept crashing, reasons unknown, just hung, no exception or
error message, so moving to this script instead)
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

import run_mem_recv
import run_mem_send

default_params = {
    'tokens_per_meaning': 4,
    'vocab_size': 4,
    'batch_size': 128,
    'enable_cuda': False,
    # 'num_layers': 1,
    # 'terminate_time_minutes': None,
}

std_args = {
    'render_every_seconds': mem_defaults.render_every_seconds,
    'save_every_seconds': 300,
    'model_file': 'tmp/{name}_{ref}_{hostname}_{date}_{time}.dat',
    'logfile': 'logs/log_{name}_{ref}_{hostname}_{date}_{time}.log',
    'load_last_model': None,
}


def run_single(
        arch: str, args, grammar: str,
        target_acc: Optional[float] = None, target_steps: Optional[int] = None):
    mode: str
    rnn_type: Optional[str]
    if ':' in arch:
        model, _, rnn_type = arch.partition(':')
    else:
        model, rnn_type = arch, None
    if grammar == 'Holistic':
        grammar_family = 'Holistic'
        corruption = None
    elif grammar == 'Comp':
        grammar_family = 'Compositional'
        corruption = None
    else:
        corruption = grammar
        grammar_family = 'Compositional'
    Runner = {
        'recv': run_mem_recv.Runner,
        'send': run_mem_send.Runner
    }[args.dir]
    runner = Runner()
    learn_type = 'soft'
    if args.gumbel:
        learn_type = 'gumbel'
    if args.rl:
        learn_type = 'rl'
    std_args['ref'] = 'grid_' + args.ref + '_' + learn_type + '_' + arch.replace(':', '') + '_' + grammar
    std_args['name'] = f'run_mem_{args.dir}'
    runner._extract_standard_args(Params(std_args))
    runner.args_parsed = True
    runner.enable_cuda = default_params['enable_cuda']
    runner.render_every_seconds = std_args['render_every_seconds']
    num_meaning_types, meanings_per_type = [int(v) for v in args.meanings.split('x')]
    ent_reg = {
        'send': args.send_ent_reg,
        'recv': args.recv_ent_reg
    }[args.dir]
    params_dict = {
        'corruptions': corruption,
        'num_meaning_types': num_meaning_types,
        'meanings_per_type': meanings_per_type,
        'grammar': grammar_family,
        'model': model,
        'rnn_type': rnn_type,
        'terminate_acc': target_acc,
        'terminate_steps': target_steps,
        'max_mins': args.max_mins,
        # 'dropout': default_params['dropout'],
        'seed': args.seed,
        'embedding_size': args.embedding_size,
        'gumbel': args.gumbel,
        'gumbel_tau': args.gumbel_tau,
        'rl': args.rl,
        'lr': args.lr,
        'l2': args.l2,
        'opt': args.opt,
        'clip_grad': args.clip_grad,
        'ent_reg': ent_reg,
        'drop': args.drop,
        'ref': std_args['ref'],
        'normalize_reward_std': args.normalize_reward_std,
    }
    params_dict.update(default_params)
    runner.params = Params(params_dict)
#     print(runner.params)
    runner.setup(runner.params)
    runner.run_base()
    res = runner.res
#     print('res', res)
    return res


def run(args):
    results_by_arch = defaultdict(list)
    result_grid = []
    for arch in args.architectures:
        results = results_by_arch[arch]
        print(arch)
        arch_result = {}
        res = run_single(
            arch=arch, args=args, grammar=args.baseline_gram, target_acc=args.tgt_acc)
        steps = res['episode']
        comp_steps = steps
        if args.count_steps:
            max_steps = steps * args.steps_cutoff_ratio
        else:
            max_steps = steps
        print('steps', steps)
        print('params', res['num_parameters'])
        acc = res['log_dict']['test_acc']
        results.append((args.baseline_gram, acc))
        arch_result['params'] = res['num_parameters']
        arch_result['arch'] = arch
        arch_result['sps'] = '%.2f' % res['log_dict']['sps']
        arch_result[f'{args.baseline_gram}_time'] = '%.0f' % res['log_dict']['elapsed_time']
        if args.count_steps:
            arch_result[f'{args.baseline_gram}_acc'] = '%.3f' % acc
            arch_result[f'{args.baseline_gram}_steps'] = steps
            arch_result[f'{args.baseline_gram}_reason'] = res['terminate_reason']
            fieldnames = [
                'arch', 'params', 'sps',
                f'{args.baseline_gram}_steps', f'{args.baseline_gram}_time',
                f'{args.baseline_gram}_acc', f'{args.baseline_gram}_reason']
        else:
            arch_result[args.baseline_gram] = '%.3f' % acc
            arch_result[f'{args.baseline_gram}_max'] = '%.3f' % res['log_dict']['max_test_acc']
            arch_result['steps'] = steps
            fieldnames = [
                'arch', 'sps', 'steps', 'params', f'{args.baseline_gram}_time',
                args.baseline_gram, f'{args.baseline_gram}_max']
        for grammar in args.grammars:
            if grammar == args.baseline_gram:
                continue
            res = run_single(arch=arch, args=args, grammar=grammar, target_steps=max_steps)
            print(grammar)
            acc = res['log_dict']['test_acc']
            print('acc', acc)
            results.append((grammar, acc))
            if args.count_steps:
                arch_result[f'{grammar}_acc'] = '%.3f' % acc
                arch_result[f'{grammar}_steps'] = res['episode']
                arch_result[f'{grammar}_ratio'] = '%.1f' % (res['episode'] / comp_steps)
                arch_result[f'{grammar}_reason'] = res['terminate_reason']
                fieldnames.append(f'{grammar}_acc')
                fieldnames.append(f'{grammar}_maxacc')
                fieldnames.append(f'{grammar}_steps')
                fieldnames.append(f'{grammar}_ratio')
                fieldnames.append(f'{grammar}_reason')
            else:
                arch_result[grammar] = '%.3f' % acc
                arch_result[f'{grammar}_max'] = '%.3f' % res['log_dict']['max_test_acc']
                fieldnames.append(grammar)
                fieldnames.append(f'{grammar}_max')
            print(arch)
            for r in results:
                print(r)
    #         for r in results_grid:
    #             print(r)
        result_grid.append(arch_result)
        print('====================')
        for arch, results in results_by_arch.items():
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

    if 'MLFLOW_TRACKING_URI' in os.environ:
        mlflow.set_experiment('hp/ec')
        mlflow.start_run(run_name=args.ref + '_summary')

        with open(args.out_csv, 'r') as f_in:
            csv_text = f_in.read()
        mlflow.log_text(csv_text, f'{args.ref}.csv')

        meta = {}
        meta['params'] = args.__dict__
        # meta['file'] = path.splitext(path.basename(file))[0]
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
        # mlflow.log_text(json.dumps(meta['params'], indent=2), 'params.txt')

        params_to_log = dict(meta['params'])
        for too_long in ['send_arch', 'recv_arch']:
            _contents = params_to_log[too_long]
            if isinstance(_contents, list):
                _contents = '\n'.join(_contents)
            del params_to_log[too_long]
            mlflow.log_text(_contents, f'{too_long}.txt')

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
    parser.add_argument('-d', '--dir', type=str, required=True, help='[recv|send] either or both (comma-separated)')
    parser.add_argument('--out-csv', type=str, default='{ref}.csv', help='where to output the results grid')

    parser.add_argument('--seed', type=int, default=123, help='seed for lang and model')

    parser.add_argument('-m', '--meanings', type=str, default='5x10', help='meaning space to use')
    parser.add_argument('--embedding-size', type=int, default=mem_defaults.embedding_size)

    parser.add_argument('--send-ent-reg', type=float, default=mem_defaults.send_ent_reg)
    parser.add_argument('--recv-ent-reg', type=float, default=mem_defaults.recv_ent_reg)
    parser.add_argument('--tgt-acc', type=float, default=0.99, help='when to stop compositional run')
    parser.add_argument('--lr', type=float, default=mem_defaults.lr)
    parser.add_argument('--l2', type=float, default=mem_defaults.l2)
    parser.add_argument('--clip-grad', type=float, default=mem_defaults.clip_grad)
    parser.add_argument('--drop', type=float, default=mem_defaults.drop)
    parser.add_argument('--opt', type=str, default=mem_defaults.opt)
    parser.add_argument('--count-steps', action='store_true')
    parser.add_argument('--steps-cutoff-ratio', type=int, default=20, help='how much to multiply comp steps for cutoff')
    parser.add_argument('--max-mins', type=float, help='how many minutes before terminating a run')
    parser.add_argument('--no-normalize-reward-std', action='store_true')

    parser.add_argument('--gumbel', action='store_true')
    parser.add_argument('--gumbel_tau', type=float, default=mem_defaults.gumbel_tau)
    parser.add_argument('--rl', action='store_true')

    parser.add_argument('--baseline-gram', type=str, default='Comp')
    parser.add_argument('--send-gram', type=str,
                        default='Comp,RandomProj,WordPairSums,Permute,Cumrot,Cumrot+Permute,ShuffleWordsDet,Holistic')
    parser.add_argument('--recv-gram', type=str,
                        default='Comp,RandomProj,WordPairSums,Permute,Cumrot,Cumrot+Permute,ShuffleWords,'
                                'ShuffleWordsDet,Holistic')
    parser.add_argument('--send-arch', type=str,
                        default='FC1L,FC2L,'
                                'RNNZero:RNN,RNNZero:GRU,RNNZero:LSTM,'
                                'RNNAutoReg:RNN,RNNAutoReg:GRU,RNNAutoReg:LSTM,'
                                'HierZero:RNN,HierZero:GRU,HierAutoReg:RNN,HierAutoReg:GRU,'
                                'RNNZero2L:RNN,RNNZero2L:GRU,RNNZero2L:SRU,RNNZero2L:LSTM,'
                                'RNNAutoReg2L:RNN,RNNAutoReg2L:GRU,RNNAutoReg2L:LSTM,'
                                'RNNZero:dgsend,RNNAutoReg:dgsend,HierZero:dgsend,'
                                'TransDecSoft,TransDecSoft2L,'
                                'Hashtable')
    parser.add_argument('--recv-arch', type=str,
                        default='RNN:RNN,RNN:GRU,RNN:LSTM,'
                                'Hier:RNN,Hier:GRU,Hier:LSTM,'
                                'RNN2L:RNN,RNN2L:SRU,RNN2L:LSTM,RNN2L:GRU,'
                                'RNN:dgrecv,RNN2L:dgrecv,Hier:dgrecv,'
                                'FC2L,FC,CNN,'
                                'KNN,Hashtable')
    args = parser.parse_args()
    if args.dir == 'send':
        args.grammars = args.send_gram
        args.architectures = args.send_arch
    elif args.dir == 'recv':
        args.grammars = args.recv_gram
        args.architectures = args.recv_arch
    args.out_csv = args.out_csv.format(**args.__dict__)

    if args.grammars != 'None':
        args.grammars = [g.replace('+', ',') for g in args.grammars.split(',')]
    else:
        args.grammars = []
    args.architectures = args.architectures.split(',')
    args.normalize_reward_std = not args.no_normalize_reward_std
    del args.__dict__['no_normalize_reward_std']

    run(args)


if __name__ == '__main__':
    parse_args()
