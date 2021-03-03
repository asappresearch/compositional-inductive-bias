#!/usr/bin/env python3
"""
This will start multiple jobs, all running e2e from scratch
It wont tail the jobs, just start them. Though it might
possibly tail some of them after starting them all (?)

The following scripts are expected to be on your path (you
will need to create/supply these; they are site/infra-specific)

ulfs_logs.sh [ref]
  - dumps the logs from job with name [ref]

ulfs_submit.py -r [ref] --no-follow -- [script_path] [script args]
  - runs script [script_path], passing in args [script args]
  - [ref] is a job name, that will be alphanumeric plus hyphens
    also allowed
  - ulfs_submit.py, with the argument --no-follow, should simply
    start the job running, executing the script [script_path]
    asynchronously, and leaving it to run
"""
import subprocess
import os
import time
import argparse


def run(args):
    sub_refs = []
    for arch_pair in args.arch_pairs:
        for link in args.links:
            arch_pair_ref = arch_pair.replace('+', '-').replace(':', '').lower()
            sub_link = {
                'rl': 'RL',
                'soft': 'Softmax',
                'gumb': 'Gumbel'
            }[link]
            send_arch, recv_arch = arch_pair.split('+')
            sub_ref = f'{args.ref}-{arch_pair_ref}-{link}'
            sub_cmd = f'ulfs_submit.py -r {sub_ref} --no-follow'
            sub_cmd += ' --'
            sub_cmd += ' mll/e2e_fixpoint.py'
            if args.meanings is not None:
                sub_cmd += f' --meanings {args.meanings}'
            if args.opt is not None:
                sub_cmd += f' --opt {args.opt}'
            if args.lr is not None:
                sub_cmd += f' --lr {args.lr}'
            if args.ent_reg is not None:
                sub_cmd += f' --ent-reg {args.ent_reg}'
            if args.seed is not None:
                sub_cmd += f' --seed {args.seed}'
            if args.train_acc is not None:
                sub_cmd += f' --train-acc {args.train_acc}'
            if args.comp_only:
                sub_cmd += ' --grammars None'
            if args.max_e2e_steps is not None:
                sub_cmd += f' --max-e2e-steps {args.max_e2e_steps}'
            sub_cmd += f' --link {sub_link}'
            sub_cmd += ' --softmax-sup'
            sub_cmd += f' --send-arch {send_arch}'
            sub_cmd += f' --recv-arch {recv_arch}'
            print(sub_cmd)
            # print(subprocess.check_output(sub_cmd.split(' ')).decode('utf-8'))
            os.system(sub_cmd)
            sub_refs.append(sub_ref)
    while True:
        for sub_ref in sub_refs:
            print(sub_ref)
            logs = subprocess.check_output(['ulfs_logs.sh', sub_ref]).decode('utf-8').split('\n')[-5:]
            print('\n'.join(logs))
        time.sleep(3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meanings', type=str)
    parser.add_argument('--links', type=str, default='soft,gumb,rl')
    parser.add_argument('--arch-pairs', type=str,
                        default='HierZero:GRU+Hier:GRU,'
                                'HierZero:dgsend+Hier:dgrecv,'
                                'RNNZero2L:SRU+RNN2L:SRU')
    parser.add_argument('--opt', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--ent-reg', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--train-acc', type=float, default=0.0)
    parser.add_argument('--comp-only', action='store_true')
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--max-e2e-steps', type=int)
    args = parser.parse_args()
    args.links = args.links.split(',')
    args.arch_pairs = args.arch_pairs.split(',')
    run(args)
