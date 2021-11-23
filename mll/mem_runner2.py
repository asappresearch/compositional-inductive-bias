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
import csv
from mll import mem_defaults
from mll import mem_runner_common

from ulfs import utils


def run(args):
    # results_by_arch = defaultdict(list)
    result_grid = []
    for arch in args.architectures:
        # results = results_by_arch[arch]
        print(arch)
        # results_by_arch[arch] = run_for_grammars(arch=arch, args=args)
        arch_result, fieldnames = mem_runner_common.run_for_grammars(arch=arch, args=args)
        result_grid.append(arch_result)
        # print('====================')
        # for arch, results in results_by_arch.items():
        #     print(arch)
        #     for r in results:
        #         print(r)
        print('====================')
        with open(args.out_csv, 'w') as f_out:
            dict_writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            dict_writer.writeheader()
            for r in result_grid:
                dict_writer.writerow(r)
        print('wrote to', args.out_csv)

    mem_runner_common.log_to_mlflow(ref=args.ref, out_csv=args.out_csv, args=args)


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
    parser.add_argument('--render-every-steps', type=int, default=50)

    # parser.add_argument('--gumbel', action='store_true')
    parser.add_argument('--gumbel_tau', type=float, default=mem_defaults.gumbel_tau)
    # parser.add_argument('--rl', action='store_true')
    parser.add_argument('--link', type=str, default='Soft', choices=['Soft', 'Gumbel', 'RL'])

    parser.add_argument('--baseline-gram', type=str, default='Comp')
    # parser.add_argument('--send-gram', type=str,
    # default='Comp,RandomProj,WordPairSums,Permute,Cumrot,Cumrot+Permute,ShuffleWordsDet,Holistic')
    parser.add_argument('--send-gram', type=str,
                        default='Comp,Permute,RandomProj,Cumrot,ShuffleWordsDet,Holistic')
    parser.add_argument('--recv-gram', type=str,
                        default='Comp,RandomProj,WordPairSums,Permute,Cumrot,Cumrot+Permute,ShuffleWords,'
                                'ShuffleWordsDet,Holistic')
    # parser.add_argument('--send-arch', type=str,
    #                     default='FC1L,FC2L,'
    #                             'RNNZero:RNN,RNNZero:GRU,RNNZero:LSTM,'
    #                             'RNNAutoReg:RNN,RNNAutoReg:GRU,RNNAutoReg:LSTM,'
    #                             'HierZero:RNN,HierZero:GRU,HierAutoReg:RNN,HierAutoReg:GRU,'
    #                             'RNNZero2L:RNN,RNNZero2L:GRU,RNNZero2L:SRU,RNNZero2L:LSTM,'
    #                             'RNNAutoReg2L:RNN,RNNAutoReg2L:GRU,RNNAutoReg2L:LSTM,'
    #                             'RNNZero:dgsend,RNNAutoReg:dgsend,HierZero:dgsend,'
    #                             'TransDecSoft,TransDecSoft2L,'
    #                             'Hashtable')
    parser.add_argument(
        '--send-arch', type=str,
        default='FC1L,FC2L,'
                'RNNAutoReg:LSTM,'
                'RNNAutoReg2L:LSTM,'
                'TransDecSoft,TransDecSoft2L,'
                'Hashtable,'
                'RNNZero:RNN,RNNZero:GRU,RNNZero:LSTM,'
                'RNNAutoReg:RNN,RNNAutoReg:GRU,'
                'HierZero:RNN,HierZero:GRU,HierAutoReg:RNN,HierAutoReg:GRU,HierAutoReg:dgsend,'
                'RNNZero2L:RNN,RNNZero2L:GRU,RNNZero2L:SRU,RNNZero2L:LSTM,'
                'RNNAutoReg2L:RNN,RNNAutoReg2L:GRU,'
                'RNNZero:dgsend,RNNAutoReg:dgsend,HierZero:dgsend'
    )
    parser.add_argument('--recv-arch', type=str,
                        default='RNN:RNN,RNN:GRU,RNN:LSTM,'
                                'Hier:RNN,Hier:GRU,Hier:LSTM,'
                                'RNN2L:RNN,RNN2L:SRU,RNN2L:LSTM,RNN2L:GRU,'
                                'RNN:dgrecv,RNN2L:dgrecv,Hier:dgrecv,'
                                'FC2L,FC1L,CNN,'
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
