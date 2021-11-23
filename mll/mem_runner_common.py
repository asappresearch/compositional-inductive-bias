import sys
import datetime
import os
from typing import Optional

import mlflow

from ulfs.params import Params
from ulfs import git_info

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
    'save_every_seconds': 0,
    'model_file': 'tmp/{name}_{ref}_{hostname}_{date}_{time}.dat',
    'logfile': 'logs/log_{name}_{ref}_{hostname}_{date}_{time}.log',
    'load_last_model': None,
    # 'render_every_steps': 50,
}


def log_to_mlflow(ref: str, out_csv: str, args):
    if 'MLFLOW_TRACKING_URI' in os.environ:
        mlflow.set_experiment('hp/ec')
        mlflow.start_run(run_name=ref + '_summary')

        with open(out_csv, 'r') as f_in:
            csv_text = f_in.read()
        mlflow.log_text(csv_text, f'{ref}.csv')

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

        params_to_log = {}
        for k, v in dict(meta['params']).items():
            if len(str(v)) > 200:
                print('param', k, 'too long, logging as text')
                too_long = k
                _contents = dict(meta['params'])[too_long]
                if isinstance(_contents, list):
                    _contents = '\n'.join(_contents)
                mlflow.log_text(_contents, f'{too_long}.txt')
            else:
                params_to_log[k] = v
        print('params to log', params_to_log)

        mlflow.log_params(params_to_log)

        del meta['gitdiff']
        del meta['gitlog']
        del meta['params']
        mlflow.log_params(meta)
        mlflow.set_tags(meta)

        mlflow.end_run()


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
    learn_type = args.link.lower()
    # if args.gumbel:
    #     learn_type = 'gumbel'
    # if args.rl:
    #     learn_type = 'rl'
    std_args['ref'] = 'grid_' + args.ref + '_' + learn_type + '_' + arch.replace(':', '') + '_' + grammar
    std_args['name'] = f'run_mem_{args.dir}'
    std_args['render_every_steps'] = args.render_every_steps
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
        'link': args.link,
        'embedding_size': args.embedding_size,
        # 'gumbel': args.gumbel,
        'gumbel_tau': args.gumbel_tau,
        # 'rl': args.rl,
        'lr': args.lr,
        'l2': args.l2,
        'opt': args.opt,
        'clip_grad': args.clip_grad,
        'ent_reg': ent_reg,
        'dropout': args.drop,
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


def run_for_grammars(arch, args):
    results = []
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
        _tgt_acc = args.tgt_acc if args.count_steps else None
        res = run_single(
            arch=arch, args=args, grammar=grammar, target_steps=max_steps,
            target_acc=_tgt_acc)
        print(grammar)
        acc = res['log_dict']['test_acc']
        print('acc', acc)
        results.append((grammar, acc))
        if args.count_steps:
            arch_result[f'{grammar}_acc'] = '%.3f' % acc
            arch_result[f'{grammar}_steps'] = res['episode']
            if comp_steps > 0:
                arch_result[f'{grammar}_ratio'] = '%.1f' % (res['episode'] / comp_steps)
            arch_result[f'{grammar}_reason'] = res['terminate_reason']
            fieldnames.append(f'{grammar}_acc')
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
    return arch_result, fieldnames
