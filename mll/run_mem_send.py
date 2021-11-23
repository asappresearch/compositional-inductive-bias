"""
as for test_rnn_memorization.py, but this time the sender, not the receiver

same idea: train supervised, and compare compositional vs non-compoisitoinal grammars
"""
import time
import random

import torch
from torch import optim
import torch.nn.functional as F
import numpy as np

from ulfs import rl_common
from ulfs.stats import Stats
from ulfs.stochastic_trajectory import StochasticTrajectory
from ulfs.runner_base_v1 import RunnerBase

from mll import mem_common, send_models, gumbel_loss, mem_defaults


class NonNeuralTrainer(object):
    def __init__(self, model):
        self.model = model
        self.num_parameters = 0

    def train(self, utts, meanings):
        output = self.model(meanings)
        _, preds = output.max(dim=-1)
        self.model.train(utts=utts, meanings=meanings)
        loss = -1
        return preds, loss


class NeuralTrainer(object):
    def __init__(self, model, p):
        self.model = model
        self.p = p

        self.num_parameters = mem_common.count_parameters(self.model)
        self.model_params = self.model.parameters()
        self.Opt = getattr(optim, p.opt)
        self.opt = self.Opt(lr=p.lr, params=self.model_params, weight_decay=p.l2)

    def forwardprop(self, meanings):
        self.model.train()
        res = self.model(meanings)
        return res

    def backprop(self, loss):
        self.opt.zero_grad()
        loss.backward()
        if self.p.clip_grad is not None and self.p.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.model_params, self.p.clip_grad)
        self.opt.step()


class SoftTrainer(NeuralTrainer):
    def __init__(self, model, p):
        super().__init__(model=model, p=p)

    def train(self, utts, meanings):
        res = self.forwardprop(meanings=meanings)
        assert not isinstance(res, dict)
        logits = res
        _, preds = logits.max(dim=-1)
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(logits.contiguous().view(
            -1, vocab_size), utts.contiguous().view(-1))
        loss_v = loss.item()
        self.backprop(loss)
        return preds, loss_v


class GumbelTrainer(NeuralTrainer):
    def __init__(self, model, p):
        super().__init__(model=model, p=p)

    def train(self, utts, meanings):
        res = self.forwardprop(meanings=meanings)
        assert not isinstance(res, dict)
        if self.model.supports_gumbel:
            probs = res
        else:
            # do gumbel ourselves. The output is logits
            logits = res
            probs = F.gumbel_softmax(logits, tau=self.p.gumbel_tau, hard=True)
        loss = gumbel_loss.gumbel_loss(probs, utts)
        _, preds = probs.max(dim=-1)
        self.backprop(loss)
        loss_v = loss.item()
        return preds, loss_v


class RLTrainer(NeuralTrainer):
    def __init__(self, model, p):
        super().__init__(model=model, p=p)
        self.p = p

    def train(self, utts, meanings):
        p = self.p
        res = self.forwardprop(meanings=meanings)
        if isinstance(res, dict):
            stochastic_trajectory = res['stochastic_trajectory']
            preds = res['utterances']
        else:
            logits = res
            seq_len, batch_size, vocab_size = logits.size()
            probs = torch.softmax(logits, dim=-1)
            stochastic_trajectory = StochasticTrajectory()
            preds = torch.zeros(seq_len, batch_size, dtype=torch.int64, device=utts.device)
            for t in range(seq_len):
                s = rl_common.draw_categorical_sample(
                    action_probs=probs[t], batch_idxes=None)
                stochastic_trajectory.append_stochastic_sample(s=s)
                preds[t] = s.actions
        correct = preds == utts
        # acc = correct.float().mean().item()
        rewards = (correct.float()).sum(dim=0)

        # for reporting purposes:
        loss_v = - rewards.mean().item()

        # assuming batch size large enough:
        rewards_mean = rewards.mean().item()
        rewards_std = rewards.std().item()
        rewards = rewards - rewards_mean
        if rewards_std > 1e-1 and p.normalize_reward_std:
            rewards = rewards / rewards_std

        rl_loss = stochastic_trajectory.calc_loss(rewards)
        loss_all = rl_loss
        if p.ent_reg is not None and p.ent_reg > 0:
            ent_loss = 0
            ent_loss -= stochastic_trajectory.entropy * p.ent_reg
            loss_all += ent_loss

        loss = loss_all
        self.backprop(loss=loss)
        return preds, loss_v


class Agent(object):
    def __init__(self, params):
        self.params = params
        p = params
        # assert 'tokens_per_meaning' not in p.__dict__
        # assert 'utt_len' not in p.__dict__
        if 'utt_len' in p.__dict__:
            print('using utt_len from params', p.utt_len)
            self.utt_len = p.utt_len
        else:
            print('calcing utt len = tokens per menaing * num meaning types')
            self.utt_len = p.tokens_per_meaning * p.num_meaning_types
        print('self.utt_len', self.utt_len)
        Model = getattr(send_models, f'{p.model}Model')
        model_params = {
            'embedding_size': p.embedding_size,
            'vocab_size': p.vocab_size,
            'utt_len': self.utt_len,
            'num_meaning_types': p.num_meaning_types,
            'meanings_per_type': p.meanings_per_type,
            # 'rnn_type': p.rnn_type
        }

        if Model.supports_gumbel:
            model_params['gumbel'] = p.link == 'Gumbel'
            model_params['gumbel_tau'] = p.gumbel_tau

        if Model.supports_dropout:
            model_params['dropout'] = p.dropout
        else:
            print('warning: dropout requested, but not available for ' + p.model)

        if 'RNN' in p.model or 'Hier' in p.model:
            model_params['rnn_type'] = p.rnn_type

        self.model = Model(
            **model_params
        )
        if p.enable_cuda:
            self.model = self.model.cuda()
        if p.model in ['Hashtable', 'KNN']:
            self.trainer = NonNeuralTrainer(self.model)
        else:
            if p.link == 'Gumbel':
                self.trainer = GumbelTrainer(model=self.model, p=p)
            elif p.link == 'RL':
                self.trainer = RLTrainer(model=self.model, p=p)
            else:
                self.trainer = SoftTrainer(model=self.model, p=p)
        self.num_parameters = self.trainer.num_parameters

    def eval(self, meanings, utts):
        if self.params.model not in ['Hashtable', 'KNN']:
            self.model.eval()
        with torch.no_grad():
            res = self.model(meanings)
            if isinstance(res, dict):
                preds = res['utterances']
            else:
                logits = res
                _, preds = logits.max(dim=-1)
        correct = preds == utts
        acc = correct.float().mean().item()
        return acc

    def train(self, meanings, utts):
        preds, loss_v = self.trainer.train(utts=utts, meanings=meanings)
        correct = preds == utts
        acc = correct.float().mean().item()

        return loss_v, acc

    def state_dict(self):
        return {
            'model_state': self.model.state_dict()
        }

    def load_state_dict(self, statedict):
        self.model.load_state_dict(statedict['model_state'])


class Runner(RunnerBase):
    def __init__(self):
        super().__init__(
            save_as_statedict_keys=[],
            additional_save_keys=[],
            step_key='episode'
        )

    def setup(self, p):
        print('applying seed', p.seed)
        random.seed(p.seed)
        torch.manual_seed(p.seed)
        np.random.seed(p.seed)
        Grammar = getattr(mem_common, f'{p.grammar}Grammar')
        self.grammar = Grammar(
            num_meaning_types=p.num_meaning_types,
            tokens_per_meaning=p.tokens_per_meaning,
            meanings_per_type=p.meanings_per_type,
            vocab_size=p.vocab_size,
            corruptions=p.corruptions
        )
        self.agent = Agent(
            params=p
        )
        self.stats = Stats([
            'episodes_count',
            'train_acc_sum',
            'train_loss_sum',
        ])
        self.max_test_acc = None
        self.max_train_acc = None

    def step(self, p):
        episode = self.episode
        render = self.should_render()
        stats = self.stats

        meanings = torch.from_numpy(
            np.random.choice(p.meanings_per_type, (p.batch_size, p.num_meaning_types), replace=True))
        utts = self.grammar.meanings_to_utterances(meanings)
        if p.enable_cuda:
            meanings, utts = meanings.cuda(), utts.cuda()

        loss, acc = self.agent.train(utts=utts, meanings=meanings)

        stats.train_acc_sum += acc
        stats.train_loss_sum += loss
        stats.episodes_count += 1

        if render:
            stats = self.stats
            log_dict = {
                'sps': stats.episodes_count / (time.time() - self.last_print),
                'train_acc': stats.train_acc_sum / stats.episodes_count,
                'train_loss': stats.train_loss_sum / stats.episodes_count
            }

            meanings = torch.from_numpy(
                np.random.choice(p.meanings_per_type, (p.batch_size, p.num_meaning_types), replace=True))
            utts = self.grammar.meanings_to_utterances(meanings)
            if p.enable_cuda:
                meanings, utts = meanings.cuda(), utts.cuda()
            test_acc = self.agent.eval(utts=utts, meanings=meanings)
            log_dict['test_acc'] = test_acc

            if self.max_train_acc is None or log_dict['train_acc'] > self.max_train_acc:
                self.max_train_acc = log_dict['train_acc']
            if self.max_test_acc is None or log_dict['test_acc'] > self.max_test_acc:
                self.max_test_acc = log_dict['test_acc']
            log_dict['max_train_acc'] = self.max_train_acc
            log_dict['max_test_acc'] = self.max_test_acc

            self.print_and_log(
                log_dict,
                formatstr='e={episode} '
                          't={elapsed_time:.0f} '
                          'sps={sps:.0f} '
                          '| train '
                          'loss {train_loss:.3f} '
                          'acc {train_acc:.3f} '
                          'max {max_train_acc:.3f} '
                          '| test '
                          'acc {test_acc:.3f} '
                          'max {max_test_acc:.3f} ',
                step=episode
            )
            stats.reset()
            self.render_every_seconds = mem_common.adjust_render_time(
                self.render_every_seconds, time.time() - self.start_time)
            if p.terminate_acc is not None and log_dict['test_acc'] >= p.terminate_acc:
                print('reached terminate test acc => terminating')
                self.finish = True
                terminate_reason = 'acc'
            if p.max_mins is not None and time.time() - self.start_time >= p.max_mins * 60:
                print('reached terminate time => terminating')
                self.finish = True
                terminate_reason = 'timeout'
            if p.terminate_steps is not None and episode >= p.terminate_steps:
                print('reached terminate steps => terminating')
                self.finish = True
                terminate_reason = 'steps'
            if self.finish:
                self.res = {
                    'num_parameters': self.agent.num_parameters,
                    'params': p.__dict__,
                    'episode': self.episode,
                    'log_dict': log_dict,
                    'elapsed_time': time.time() - self.start_time,
                    'terminate_reason': terminate_reason,
                    'logfile': self.logfile,
                }


if __name__ == '__main__':
    runner = Runner()
    runner.add_param('--grammar', type=str, default='Compositional')
    runner.add_param('--corruptions', type=str, help='[permute]')

    runner.add_param('--meanings', type=str, default='5x10', help='eg 5x10 for 5 meaning types, with 10 meanings each')
    runner.add_param('--tokens-per-meaning', type=int, default=4)
    runner.add_param('--vocab-size', type=int, default=4, help='excludes any terminator')
    runner.add_param('--batch-size', type=int, default=128)

    runner.add_param('--ent-reg', type=float, default=mem_defaults.send_ent_reg)
    runner.add_param('--lr', type=float, default=mem_defaults.lr)
    runner.add_param('--l2', type=float, default=mem_defaults.l2, help='l2 weight regularization')
    runner.add_param('--clip-grad', type=float, default=mem_defaults.clip_grad)
    runner.add_param('--drop', type=float, default=mem_defaults.drop)
    runner.add_param('--opt', type=str, default=mem_defaults.opt)

    runner.add_param('--link', type=str, default='soft', choices=['Soft', 'Gumb', 'RL'])
    # runner.add_param('--rl', action='store_true')
    runner.add_param('--no-normalize-reward-std', action='store_true')
    # runner.add_param('--gumbel', action='store_true')
    runner.add_param('--gumbel-tau', type=str, default=mem_defaults.gumbel_tau)

    runner.add_param('--seed', type=int, default=123)

    runner.add_param('--model', type=str, default='RNNZero')
    runner.add_param('--rnn-type', type=str, default='GRU')
    # runner.add_param('--num-layers', type=int, default=1)
    runner.add_param('--embedding-size', type=int, default=mem_defaults.embedding_size)

    runner.add_param('--terminate-acc', type=float, help='finish running if reach this test acc')
    runner.add_param('--terminate-steps', type=int, help='finish running if reach this num steps')
    runner.add_param('--max-mins', type=float, help='finish running if reach this elapsed time')

    runner.parse_args()
    runner.render_every_seconds = mem_defaults.render_every_seconds
    runner.params.num_meaning_types, runner.params.meanings_per_type = [
        int(v) for v in runner.params.meanings.split('x')]
    del runner.params.__dict__['meanings']
    print('runner.args', runner.args)
    assert not runner.params.gumbel or not runner.params.rl
    runner.params.normalize_reward_std = not runner.params.no_normalize_reward_std
    del runner.params.__dict__['no_normalize_reward_std']
    runner.setup_base()
    runner.run_base()
