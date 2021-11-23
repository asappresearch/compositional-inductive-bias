"""
First train both sender and receiver supervised, then put them
end to end, and see what happens...
"""
import time
import random

import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

from ulfs.params import Params
from ulfs import rl_common, metrics
from ulfs import utils, nn_modules
from ulfs.runner_base_v1 import RunnerBase
from ulfs.stats import Stats

from mll import run_mem_recv, run_mem_send
from mll import mem_common, mem_defaults


default_sup_params = {
    'embedding_size': mem_defaults.embedding_size,
    'batch_size': 128,
    # 'terminate_time_minutes': None,
    'terminate_steps': None,
    'num_layers': 1,
    'seed': 123,
}


class SoftmaxLink(object):
    def __init__(self, params):
        self.params = params

    def __call__(self, meanings, utts_logits, receiver_agent, sender_agent, opt_both):
        p = self.params
        utts_probs = F.softmax(utts_logits, dim=-1)
        meanings_logits = receiver_agent.model(utts=utts_probs)
        crit = nn.CrossEntropyLoss()
        _, meanings_pred = meanings_logits.max(dim=-1)
        correct = (meanings_pred == meanings)
        acc = correct.float().mean().item()
        loss = crit(meanings_logits.view(-1, p.meanings_per_type), meanings.view(-1))
        opt_both.zero_grad()
        loss.backward()
        opt_both.step()
        loss = loss.item()
        return 0, 0, loss, acc


class GumbelLink(object):
    def __init__(self, params):
        self.params = params

    def __call__(self, meanings, utts_logits, receiver_agent, sender_agent, opt_both):
        p = self.params
        utts = nn_modules.gumbel_softmax(utts_logits, tau=p.gumbel_tau, hard=p.gumbel_hard, eps=1e-6)
        meanings_logits = receiver_agent.model(utts=utts)
        crit = nn.CrossEntropyLoss()
        _, meanings_pred = meanings_logits.max(dim=-1)
        correct = (meanings_pred == meanings)
        acc = correct.float().mean().item()
        loss = crit(meanings_logits.view(-1, p.meanings_per_type), meanings.view(-1))
        opt_both.zero_grad()
        loss.backward()
        opt_both.step()
        loss = loss.item()
        return 0, 0, loss, acc


class RLLink(object):
    def __init__(self, params):
        self.params = params

    def __call__(self, meanings, utts_logits, receiver_agent, sender_agent, opt_both):
        p = self.params
        utts_probs = F.softmax(utts_logits, dim=-1)
        s_sender = rl_common.draw_categorical_sample(
            action_probs=utts_probs,
            batch_idxes=None
        )
        utts = s_sender.actions
        if p.predict_correct:
            pred_correct_logits, meanings_logits = receiver_agent.model(utts=utts.detach(), do_predict_correct=True)
            _, pred_correct_pred = pred_correct_logits.max(dim=-1)
        else:
            meanings_logits = receiver_agent.model(utts=utts.detach())
        _, meanings_pred = meanings_logits.max(dim=-1)
        correct = (meanings_pred == meanings)
        # we will always report the argmax acc perhaps?
        acc = correct.float().mean().item()

        if p.rl_reward == 'ce':
            crit = nn.CrossEntropyLoss(reduction='none')
            receiver_loss_per_ex = crit(meanings_logits.view(-1, p.meanings_per_type), meanings.view(-1))
            receiver_loss_per_ex = receiver_loss_per_ex.view(p.e2e_batch_size, p.num_meaning_types)
            receiver_loss_per_ex = receiver_loss_per_ex.mean(dim=-1)
            receiver_loss = receiver_loss_per_ex.mean()
            receiver_agent.opt.zero_grad()
            receiver_loss.backward()
            if p.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(receiver_agent.model_params, p.clip_grad)
            receiver_agent.opt.step()

            sender_loss = s_sender.calc_loss(- receiver_loss_per_ex.data)
            sender_agent.opt.zero_grad()
            sender_loss.backward()
            if p.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(sender_agent.model_params, p.clip_grad)
            sender_agent.opt.step()

            loss = receiver_loss.item()
        elif p.rl_reward == 'count_meanings':
            # need to sample receiver too, so we can send REINFORCE loss back to it too
            recv_probs = F.softmax(meanings_logits, dim=-1)
            s_recv = rl_common.draw_categorical_sample(
                action_probs=recv_probs,
                batch_idxes=None
            )
            meanings_right = (s_recv.actions == meanings).float()
            if p.predict_correct:
                pred_correct_probs = F.softmax(pred_correct_logits, dim=-1)
                s_pred_correct = rl_common.draw_categorical_sample(
                    action_probs=pred_correct_probs,
                    batch_idxes=None
                )
                # reward of +1 for those which are right and we predicted right
                # reward of -0.5 for those which we predicted right, which are wrong
                # in other words, rewards only for those we predicted right, ignore everything else
                # (but punish those we predicted right that are wrong, cf before, those just got zero)
                rewards = (meanings_right * 1.5 - 0.5) * s_pred_correct.actions.float()
            else:
                # this is the reward
                rewards = meanings_right.float()

            # for reporting purposes:
            loss = - rewards.mean().item()

            # lets baseline the reward first
            rewards_mean = rewards.mean().item()
            rewards_std = rewards.std().item()
            rewards = rewards - rewards_mean
            if rewards_std > 1e-1:
                rewards = rewards / rewards_std

            rl_loss = s_sender.calc_loss(rewards.mean(dim=-1)) + s_recv.calc_loss(rewards)
            if p.predict_correct:
                rl_loss += s_pred_correct.calc_loss(rewards)
            loss_all = rl_loss
            ent_loss = 0
            if p.send_ent_reg is not None and p.send_ent_reg > 0:
                ent_loss -= s_sender.entropy * p.send_ent_reg
            if p.recv_ent_reg is not None and p.recv_ent_reg > 0:
                ent_loss -= s_recv.entropy * p.recv_ent_reg
            loss_all = loss_all + ent_loss

            opt_both.zero_grad()
            loss_all.backward()
            if p.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(receiver_agent.trainer.model_params, p.clip_grad)
                torch.nn.utils.clip_grad_norm_(sender_agent.trainer.model_params, p.clip_grad)
            opt_both.step()
        else:
            raise Exception(f'rl_reward {p.rl_reward} not recognized')

        return rewards_mean, rewards_std, loss, acc


class Runner(RunnerBase):
    def __init__(self):
        super().__init__(
            save_as_statedict_keys=['sender_model', 'receiver_model'],
            additional_save_keys=[],
            step_key='episode'
        )

    def setup(self, p):
        default_sup_params['vocab_size'] = p.vocab_size
        default_sup_params['tokens_per_meaning'] = p.tokens_per_meaning

        self.runner_by_direction = {}
        self.grammar_source = None
        self.sup_res_by_direction = {}
        self.max_sup = 0
        for direction in ['recv', 'send']:
            print(direction)
            Runner = {
                'recv': run_mem_recv.Runner,
                'send': run_mem_send.Runner
            }[direction]

            p.num_meaning_types, p.meanings_per_type = [
                int(v) for v in p.meanings.split('x')]

            sub_ref = f'{p.ref}_{direction}_{p.meanings}'
            sub_ref += 'C' if p.grammar == 'Compositional' else 'H'
            if p.corruptions is not None:
                sub_ref += '_' + p.corruptions
            sub_runner = Runner()
            self.runner_by_direction[direction] = sub_runner

            name = 'mem_runner'
            std_args = {
                'render_every_seconds': mem_defaults.render_every_seconds,
                'save_every_seconds': -1,
                'render_every_steps': 50,
                'name': name,
                'model_file': 'tmp/{name}_{name}_{hostname}_{date}_{time}.dat',
                'logfile': 'logs/log_{name}_{ref}_{hostname}_{date}_{time}.log',
                'ref': sub_ref,
                'disable_cuda': not p.enable_cuda,
                'load_last_model': None,
            }
            sub_runner._extract_standard_args(Params(std_args))
            sub_runner.enable_cuda = p.enable_cuda

            sub_runner.args_parsed = True
            sub_runner.render_every_seconds = mem_defaults.render_every_seconds

            arch = p.send_arch if direction == 'send' else p.recv_arch
            if ':' in arch:
                model, _, rnn_type = arch.partition(':')
            else:
                model, rnn_type = arch, None

            child_link = p.link
            if p.gumbel_sup:
                child_link = 'Gumbel'
            if p.softmax_sup:
                child_link = 'Softmax'
            ent_reg = {
                'send': p.send_ent_reg,
                'recv': p.recv_ent_reg
            }[direction]
            params_dict = {
                'corruptions': p.corruptions,
                'num_meaning_types': p.num_meaning_types,
                'meanings_per_type': p.meanings_per_type,
                'grammar': p.grammar,
                'model': model,
                'rnn_type': rnn_type,
                'max_mins': p.max_sup_mins,
                'terminate_acc': p.train_acc,
                'terminate_steps': p.max_sup_steps,
                'ref': sub_ref,
                'clip_grad': p.clip_grad,
                'dropout': p.dropout,
                'opt': p.opt,
                'enable_cuda': p.enable_cuda,
                'seed': p.seed,
                'l2': p.l2,
                'link': child_link,
                # 'gumbel': child_link == 'Gumbel',
                # 'rl': child_link == 'RL',
                'gumbel_tau': p.gumbel_tau,
                'lr': p.lr,
                'ent_reg': ent_reg,
                'normalize_reward_std': p.normalize_reward_std,
            }
            params_dict.update(default_sup_params)
            print('params_dict', params_dict)
            sub_runner.params = Params(params_dict)
            print(sub_runner.params)
            # sub_runner.setup(sub_runner.params)
            sub_runner.setup_base(params=sub_runner.params, delay_mlflow=True)
            if self.grammar_source is None:
                self.grammar_source = sub_runner.grammar
            else:
                sub_runner.grammar = self.grammar_source
            if p.train_acc > 0:
                sub_runner.start_mlflow()
                sub_runner.run_base()
                assert sub_runner.grammar.id == self.grammar_source.id
                print('res', sub_runner.res)
                sub_runner.res['record_type'] = 'sup_train_res'
                sub_runner.res['dir'] = direction
                self.log(sub_runner.res)
                self.sup_res_by_direction[direction] = sub_runner.res
                self.max_sup = max(self.max_sup, sub_runner.res['episode'])
                print('max_sup', self.max_sup)
                print('')

        print('done supervised training...')
        print('max_sup', self.max_sup)
        if p.max_e2e_ratio is not None and p.max_e2e_ratio > 0:
            if p.max_e2e_steps is None:
                p.max_e2e_steps = int(p.max_e2e_ratio * self.max_sup)
            else:
                p.max_e2e_steps = min(p.max_e2e_steps, int(p.max_e2e_ratio * self.max_sup))
            print('p.max_e2e_steps', p.max_e2e_steps)
        for direction, sub_runner in self.runner_by_direction.items():
            r = sub_runner
            sub_p = r.agent.params
            meanings = torch.from_numpy(np.random.choice(
                p.meanings_per_type, (sub_p.batch_size, p.num_meaning_types), replace=True))
            utts = self.grammar_source.meanings_to_utterances(meanings)
            if self.enable_cuda:
                utts = utts.cuda()
                meanings = meanings.cuda()
            acc = r.agent.eval(utts=utts, meanings=meanings)
            print(direction, ' a=%.3f' % acc)
        self.stats = Stats([
            'episodes_count',
            'e2e_acc_sum',
            'e2e_loss_sum',
            'recv_acc_sum',
            'send_acc_sum',
            'rewards_mean_sum',
            'rewards_std_sum'
        ])

        self.vocab_size = default_sup_params['vocab_size']
        self.receiver = self.runner_by_direction['recv']
        self.sender = self.runner_by_direction['send']
        self.sender_model = self.sender.agent.model
        self.receiver_model = self.receiver.agent.model

        print('e2e applying seed', p.seed)
        random.seed(p.seed)
        torch.manual_seed(p.seed)
        np.random.seed(p.seed)

        self.Opt = getattr(optim, p.opt)
        self.opt_both = self.Opt(
            lr=p.lr,
            params=list(self.receiver.agent.model.parameters()) + list(self.sender.agent.model.parameters()),
            weight_decay=p.l2
        )
        self.start_mlflow()

    def step(self, p):
        episode = self.episode
        render = self.should_render()
        stats = self.stats

        for sub_runner in self.runner_by_direction.values():
            sub_runner.agent.model.train()

        meanings = torch.from_numpy(np.random.choice(
            p.meanings_per_type, (p.e2e_batch_size, p.num_meaning_types), replace=True))
        if self.enable_cuda:
            meanings = meanings.cuda()

        utts_logits = self.sender.agent.model(meanings=meanings.detach())
        if p.window_noise:
            # shall we just apply the same window to the entire batch?
            # that would at least save runtime...
            # we'll just sample two non-identical integers, order them, and
            # they are the window pos
            utt_len = utts_logits.size(0)
            window_idxes = sorted(list(np.random.choice(utt_len + 1, 2, replace=False)))
            utts_logits = utts_logits[window_idxes[0]:window_idxes[1]]
        Link = globals()[f'{p.link}Link']
        link = Link(params=p)
        rewards_mean, rewards_std, loss, acc = link(
            meanings=meanings,
            utts_logits=utts_logits,
            receiver_agent=self.receiver.agent,
            sender_agent=self.sender.agent,
            opt_both=self.opt_both
        )

        stats.episodes_count += 1
        stats.e2e_acc_sum += acc
        stats.e2e_loss_sum += loss
        stats.rewards_mean_sum += rewards_mean
        stats.rewards_std_sum += rewards_std

        if p.max_e2e_steps is not None and episode + 1 >= p.max_e2e_steps:
            print('reached max steps => terminating')
            self.finish = True
        if p.max_e2e_mins is not None and time.time() - self.start_time >= p.max_e2e_mins * 60:
            print('reached max e2e time => terminating')
            self.finish = True

        if render or self.finish:
            log_dict = {
                'sps': int(episode / (time.time() - self.start_time)),
                'elapsed_time': time.time() - self.start_time,
                'e2e_loss': stats.e2e_loss_sum / stats.episodes_count,
                'e2e_acc': stats.e2e_acc_sum / stats.episodes_count,
                'r_mean': stats.rewards_mean_sum / stats.episodes_count,
                'r_std': stats.rewards_std_sum / stats.episodes_count
            }

            # evaluate rho on current utterances
            utts = utts_logits.max(dim=-1)[1]
            utts = utts.transpose(0, 1)  # [N][seq_len]
            rho = metrics.topographic_similarity(
                utts=utts, labels=meanings)
            log_dict['rho'] = rho

            for sub_runner in self.runner_by_direction.values():
                sub_runner.agent.model.eval()
            for direction, sub_runner in self.runner_by_direction.items():
                r = sub_runner
                meanings = torch.from_numpy(np.random.choice(
                    p.meanings_per_type, (p.e2e_batch_size, p.num_meaning_types), replace=True))
                utts = self.grammar_source.meanings_to_utterances(meanings)
                if self.enable_cuda:
                    meanings = meanings.cuda()
                    utts = utts.cuda()
                acc = r.agent.eval(utts=utts, meanings=meanings)
                log_dict[f'{direction}_acc'] = acc

            self.print_and_log(
                log_dict,
                formatstr='e={episode} '
                          't={elapsed_time:.0f} '
                          'sps={sps:.0f} '
                          '|e2e '
                          'l={e2e_loss:.3f} '
                          'a={e2e_acc:.3f} '
                          'r_mean={r_mean:.3f} '
                          'r_std={r_std:.3f} '
                          'rho={rho:.3f} '
                          '|sup '
                          'send {send_acc:.3f} '
                          'recv {recv_acc:.3f} ',
                step=episode
            )

            if self.finish:
                self.res = {
                    'e2e_acc': log_dict['e2e_acc'],
                    'e2e_steps': self.episode,
                    'e2e_sps': log_dict['sps'],
                    'e2e_time': log_dict['elapsed_time'],
                    'e2e_send_acc': log_dict['send_acc'],
                    'e2e_recv_acc': log_dict['recv_acc'],
                    'e2e_log': self.logfile,
                }
                if 'send' in self.sup_res_by_direction:
                    sup_send_res = self.sup_res_by_direction['send']
                    self.res['send_num_parameters'] = sup_send_res['num_parameters']
                    self.res['sup_send_steps'] = sup_send_res['episode']
                    self.res['sup_send_acc'] = sup_send_res['log_dict']['test_acc']
                    self.res['sup_send_terminate_reason'] = sup_send_res['terminate_reason']
                    self.res['sup_send_time'] = sup_send_res['elapsed_time']
                    self.res['sup_send_log'] = sup_send_res['logfile']
                if 'recv' in self.sup_res_by_direction:
                    sup_recv_res = self.sup_res_by_direction['recv']
                    self.res['recv_num_parameters'] = sup_recv_res['num_parameters']
                    self.res['sup_recv_steps'] = sup_recv_res['episode']
                    self.res['sup_recv_acc'] = sup_recv_res['log_dict']['test_acc']
                    self.res['sup_recv_terminate_reason'] = sup_recv_res['terminate_reason']
                    self.res['sup_recv_time'] = sup_recv_res['elapsed_time']
                    self.res['sup_recv_log'] = sup_recv_res['logfile']
                print('res', self.res)

            stats.reset()

        if self.finish:
            print('terminating')
        else:
            self.render_every_seconds = mem_common.adjust_render_time(
                self.render_every_seconds, time.time() - self.start_time)


if __name__ == '__main__':
    utils.clean_argv()
    runner = Runner()

    runner.add_param('--e2e-batch-size', type=int, default=128)
    runner.add_param('--train-acc', type=float, default=0.99)
    runner.add_param('--max-sup-steps', type=int)
    runner.add_param('--max-e2e-steps', type=int)
    runner.add_param('--max-sup-mins', type=float)
    runner.add_param('--max-e2e-mins', type=float)
    runner.add_param('--max-e2e-ratio', type=float,
                     help='e2e steps wont be more than this ratio times max(sup send steps, sup recv steps)')
    # runner.add_param('--max-mins', type=float)
    runner.add_param('--opt', type=str, default=mem_defaults.opt)
    runner.add_param('--send-ent-reg', type=float, default=mem_defaults.send_ent_reg)
    runner.add_param('--recv-ent-reg', type=float, default=mem_defaults.recv_ent_reg)
    runner.add_param('--lr', type=float, default=mem_defaults.lr)
    runner.add_param('--l2', type=float, default=mem_defaults.l2, help='l2 weight regularization')
    runner.add_param('--no-normalize-reward-std', action='store_true')
    runner.add_param('--softmax-sup', action='store_true')
    runner.add_param('--gumbel-sup', action='store_true')

    runner.add_param('--grammar', type=str, default='Compositional')
    runner.add_param('--meanings', type=str, default='5x10')
    runner.add_param('--tokens-per-meaning', type=int, default=4)
    runner.add_param('--vocab-size', type=int, default=4)
    runner.add_param('--corruptions', type=str)

    runner.add_param('--predict-correct', action='store_true')

    runner.add_param('--send-arch', type=str, default='RNNZero:RNN')
    runner.add_param('--recv-arch', type=str, default='RNN:RNN')
    runner.add_param('--clip-grad', type=float, default=mem_defaults.clip_grad)
    runner.add_param('--drop', type=float, default=mem_defaults.drop)
    runner.add_param('--seed', type=int, default=123)

    runner.add_param('--gumbel-tau', type=float, default=mem_defaults.gumbel_tau)
    runner.add_param('--no-gumbel-hard', action='store_true')

    runner.add_param('--rl-reward', type=str, default='count_meanings', help=['ce|count_meanings]'])

    runner.add_param('--link', type=str, default='Softmax', help='[Softmax|Gumbel|RL]')

    runner.add_param('--window-noise', action='store_true')

    runner.parse_args()
    runner.params.gumbel_hard = not runner.params.no_gumbel_hard
    del runner.params.__dict__['no_gumbel_hard']
    runner.params.normalize_reward_std = not runner.params.no_normalize_reward_std
    del runner.params.__dict__['no_normalize_reward_std']
    runner.render_every_seconds = mem_defaults.render_every_seconds
    runner.setup_base(delay_mlflow=True)
    runner.run_base()
