"""
forked from mll/ilm_rnn2018janb.py, 23 mar 2019

This is going to be ILM, but we're going to GA slightly, something like:
- train one model
- instead of then teaching one student, teach eg three students
- then train the three students
- pick the best student, and train three new students from that
(and we can then compare with the pure ILM approach)
"""
import time
import math

import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

from ulfs import rl_common, utils, nn_modules, metrics
from ulfs.runner_base_v1 import RunnerBase
from ulfs.stats import Stats

from mll import run_mem_recv, run_mem_send


class SoftmaxLink(object):
    def __init__(self, params):
        self.params = params

    def __call__(self, meanings, utts_logits, receiver_agent, sender_agent, opt_both, do_backward=True):
        p = self.params
        utts_probs = F.softmax(utts_logits, dim=-1)
        meanings_logits = receiver_agent.model(utts=utts_probs)
        crit = nn.CrossEntropyLoss()
        _, meanings_pred = meanings_logits.max(dim=-1)
        correct = (meanings_pred == meanings)
        acc = correct.float().mean().item()
        if do_backward:
            loss = crit(meanings_logits.view(-1, p.meanings_per_type), meanings.view(-1))
            opt_both.zero_grad()
            loss.backward()
            opt_both.step()
            loss = loss.item()
            return loss, acc
        else:
            return acc


class GumbelLink(object):
    def __init__(self, params):
        self.params = params

    def __call__(self, meanings, utts_logits, receiver_agent, sender_agent, opt_both):
        p = self.params
        utts = nn_modules.gumbel_softmax(utts_logits, tau=p.gumbel_temp, hard=p.gumbel_hard, eps=1e-6)
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
        return loss, acc


class RLLink(object):
    def __init__(self, params):
        self.params = params

    def __call__(self, meanings, utts_logits, receiver_agent, sender_agent, opt_both, do_backward=True):
        p = self.params
        utts_probs = F.softmax(utts_logits, dim=-1)
        s_sender = rl_common.draw_categorical_sample(
            action_probs=utts_probs,
            batch_idxes=None
        )
        utts = s_sender.actions
        meanings_logits = receiver_agent.model(utts=utts.detach())
        _, meanings_pred = meanings_logits.max(dim=-1)
        correct = (meanings_pred == meanings)
        # we will always report the argmax acc perhaps?
        acc = correct.float().mean().item()

        if not do_backward:
            return acc

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
            loss_all = rl_loss
            if p.ent_reg is not None and p.ent_reg > 0:
                ent_loss = 0
                ent_loss -= s_sender.entropy * p.ent_reg
                ent_loss -= s_recv.entropy * p.ent_reg
                loss_all += ent_loss

            opt_both.zero_grad()
            loss_all.backward()
            if p.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(receiver_agent.model_params, p.clip_grad)
                torch.nn.utils.clip_grad_norm_(sender_agent.model_params, p.clip_grad)
            opt_both.step()
        else:
            raise Exception(f'rl_reward {p.rl_reward} not recognized')

        return rewards_mean, rewards_std, loss, acc


class SendReceivePair(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        p = params

        self.sender_agent = run_mem_send.Agent(
            params=params
        )
        self.receiver_agent = run_mem_recv.Agent(
            params=params
        )

        self.sender_model = self.sender_agent.model
        self.receiver_model = self.receiver_agent.model
        self.Opt = getattr(optim, p.opt)
        self.opt_both = self.Opt(
            lr=0.001,
            params=list(self.receiver_agent.model.parameters()) + list(self.sender_agent.model.parameters())
        )

    def forward(self):
        pass

    def state_dict(self):
        return {
            'receiver_agent_state': self.receiver_agent.state_dict(),
            'sender_agent_state': self.sender_agent.state_dict(),
        }

    def load_state_dict(self, statedict):
        self.receiver_agent.load_state_dict(statedict['receiver_agent_state'])
        self.sender_agent.load_state_dict(statedict['sender_agent_state'])


class MeaningUnflattener(object):
    def __init__(self, num_meaning_types, meanings_per_type):
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type

    def __call__(self, flat_meanings):
        assert len(flat_meanings.size()) == 1
        N = flat_meanings.size(0)
        print('N', N)
        meanings = torch.zeros(N, self.num_meaning_types, device=flat_meanings.device, dtype=torch.int64)
        print('meanings.size()', meanings.size())
        for t in range(self.num_meaning_types):
            print('t', t, 'num_meaning_types', self.num_meaning_types, 'self.num_meaning_types - t - 1',
                  self.num_meaning_types - t - 1)
            token = flat_meanings % self.meanings_per_type
            meanings[:, self.num_meaning_types - t - 1] = token
            flat_meanings = flat_meanings // self.meanings_per_type
        assert flat_meanings.sum().item() == 0
        return meanings


def batched_run_nograd(params, model, inputs, batch_size, input_batch_dim, output_batch_dim):
    """
    assumes N is multiple of batch_size
    """
    p = params
    N = inputs.size(input_batch_dim)
    num_batches = (N + batch_size - 1) // batch_size
    outputs = None
    count = 0
    for b in range(num_batches):
        b_start = b * batch_size
        b_end = min(b_start + batch_size, N)
        count += (b_end - b_start)
        input_batch = inputs.narrow(dim=input_batch_dim, start=b_start, length=b_end - b_start)
        if p.enable_cuda:
            input_batch = input_batch.cuda()
        with torch.no_grad():
            output_batch = model(input_batch).detach().cpu()
        if outputs is None:
            out_size_full = list(output_batch.size())
            out_size_full[output_batch_dim] = N
            outputs = torch.zeros(*out_size_full, dtype=output_batch.dtype, device='cpu')
        out_narrow = outputs.narrow(dim=output_batch_dim, start=b_start, length=b_end - b_start)
        out_narrow[:] = output_batch
    assert count == N
    return outputs


def round_down_multiple(v, multiple):
    return (v // multiple) * multiple


def meanings_to_set(meanings_per_type, meanings):
    # s = set()
    N, T = meanings.size()
    meanings_ints = torch.zeros(N, dtype=torch.int64)
    for t in range(T):
        meanings_ints *= meanings_per_type
        meanings_ints += meanings[:, t]
    s = set(meanings_ints.tolist())
    return s


def generate_masks(seq_len, num_zeros):
    num_masks = math.factorial(seq_len) // math.factorial(num_zeros) // math.factorial(seq_len - num_zeros)
    masks = torch.zeros(num_masks, seq_len, dtype=torch.uint8)
    print('num_masks', num_masks)
    poses = torch.zeros(num_zeros, dtype=torch.int64)
    for i in range(num_zeros):
        poses[i] = i
    n = 0
    while True:
        this_mask = torch.ones(seq_len, dtype=torch.uint8)
        this_mask[poses] = 0
        yield this_mask
        masks[n] = this_mask
        moved_ok = False
        for j in range(num_zeros - 1, -1, -1):
            if poses[j] < seq_len - 1:
                if j == num_zeros - 1 or poses[j] < poses[j + 1] - 1:
                    moved_ok = True
                    break
        if not moved_ok:
            break
        poses[j] += 1
        for i in range(j + 1, num_zeros):
            poses[i] = poses[j] + (i - j)
        n += 1


def remove_common_ngrams(params, train, holdout, max_allowed_ngram_length):
    p = params
    N_train = train.size(0)
    num_holdout = holdout.size(0)
    print('N_train', N_train, 'num_holdout', num_holdout)
    valid = torch.ones(N_train, dtype=torch.uint8)
    for num_compare_zeros in range(0, p.num_meaning_types - max_allowed_ngram_length):
        print('num_compare_zeros', num_compare_zeros, 'num_compare_meanings', p.num_meaning_types - num_compare_zeros)
        for compare_mask in generate_masks(seq_len=p.num_meaning_types, num_zeros=num_compare_zeros):
            for n in range(num_holdout):
                holdout_v = holdout[n]
                equal = holdout_v == train
                equal[:, (1 - compare_mask).nonzero().long().view(-1)] = 1
                equal = equal.min(dim=-1)[0]
                valid[equal.nonzero()] = 0
        print('num_compare_zeros', num_compare_zeros, 'valid.sum()', valid.sum().item())
    train = train[valid.nonzero().long().view(-1)]
    N_train = train.size(0)
    print('N_train', N_train)
    return train


def remove_duplicates(batch_size, meanings, utterances):
    """
    utterances are [M][N]
    meanings are [N][num meaning types]
    """
    vocab_size = utterances.max().item() + 1
    M, N = utterances.size()
    utts_flat = torch.zeros(N, dtype=torch.int64)
    for m in range(M):
        utts_flat = utts_flat * vocab_size + utterances[m]
    keep_indices = []
    seen_flats = set()
    for n in range(N):
        utt_flat = utts_flat[n].item()
        if utt_flat in seen_flats:
            continue
        seen_flats.add(utt_flat)
        same_idxes = (utts_flat == utt_flat).nonzero().view(-1).long()
        chosen_idx = same_idxes[np.random.randint(same_idxes.size(0))].item()
        keep_indices.append(chosen_idx)
    print('unique ', len(keep_indices), 'out of', N)
    keep_indices = torch.LongTensor(keep_indices)

    meanings = meanings[keep_indices]
    utterances = utterances[:, keep_indices]
    unique = len(keep_indices)
    return meanings, utterances, unique


class Runner(RunnerBase):
    def __init__(self):
        super().__init__(
            save_as_statedict_keys=['teacher'],
            additional_save_keys=['last_res'],
            step_key='episode'
        )

    def setup(self, p):
        if p.seed is not None:
            torch.manual_seed(p.seed)
            np.random.seed(p.seed)
            print('seeding torch and numpy using ', p.seed)

        self.teacher = SendReceivePair(params=p)

        print('meanings per type', p.meanings_per_type, 'num_meaning_types', p.num_meaning_types)
        self.N_all_meanings = int(math.pow(p.meanings_per_type, p.num_meaning_types))
        print('N_all_meanings', self.N_all_meanings)
        self.all_meanings_shuffled = torch.from_numpy(np.random.choice(
            self.N_all_meanings, self.N_all_meanings, replace=False))
        self.N_holdout = p.num_holdout
        meaning_unflattener = MeaningUnflattener(
            num_meaning_types=p.num_meaning_types,
            meanings_per_type=p.meanings_per_type
        )
        self.heldout_meanings = meaning_unflattener(self.all_meanings_shuffled[:self.N_holdout])
        self.train_val_meanings = meaning_unflattener(self.all_meanings_shuffled[self.N_holdout:])
        print('self.heldout_meanings.size()', self.heldout_meanings.size())

        print('self.train_val_meanings.size(0)', self.train_val_meanings.size(0))
        print('self.train_val_meanings.size()', self.train_val_meanings.size())

        self.train_val_meanings = remove_common_ngrams(
            params=p,
            train=self.train_val_meanings,
            holdout=self.heldout_meanings,
            max_allowed_ngram_length=p.max_allowed_ngram_length
        )

        self.N_train_val = self.train_val_meanings.size(0)
        print('self.train_val_meanings.size(0)', self.train_val_meanings.size(0))
        print('self.train_val_meanings.size()', self.train_val_meanings.size())
        self.N_train = int(self.N_train_val * p.train_frac)
        print('N_all_meanings', self.N_all_meanings, 'N_train', self.N_train, 'N_holdout', self.N_holdout)

        self.stats = Stats([
            'episodes_count',
            'holdout_acc_sum',
            'e2e_acc_sum',
            'recv_acc_sum',
            'send_acc_sum',
        ])

        self.last_res = None

    def step(self, p):
        episode = self.episode
        stats = self.stats

        reses = []
        students = []
        for student_it in range(p.students_per_generation):
            student, res = self.run_student()
            reses.append(res)
            students.append(student)

        best_res = None
        best_idx = None
        if self.last_res is not None:
            best_res = self.last_res[p.student_compare_key]
            best_idx = -1
        for i, res in enumerate(reses):
            _res = res[p.student_compare_key]
            if best_res is None or _res > best_res:
                best_res = _res
                best_idx = i
        # print('best idx', best_idx, best_res)
        if best_idx >= 0:
            self.teacher = students[best_idx]
            print('swapped teacher with student', best_idx)
            self.last_res = reses[best_idx]

        if True:
            log_dict = {
                'type': 'ilm',
                'sps': int(episode / (time.time() - self.start_time)),
                'elapsed_time': time.time() - self.start_time
            }
            for k, v in self.last_res.items():
                log_dict[k] = v

            formatstr = (
                'ilm i={episode} '
                't={elapsed_time:.0f} '
                'sps={sps:.0f} '
                'snd[e={sup_sender_epochs} a={sup_sender_acc:.3f} t={sup_sender_time:.0f}] '
                'rcv[e={sup_receiver_epochs} a={sup_receiver_acc:.3f} t={sup_receiver_time:.0f}] '
                'e2e[acc={e2e_acc:.3f} t={e2e_time:.0f}] '
                'uniq={uniqueness:.3f} '
                'ho={holdout_acc:.3f} '
                'rho={rho:.3f} '
            )

            self.print_and_log(log_dict, formatstr=formatstr)
            stats.reset()
        if p.max_episodes is not None and episode + 1 >= p.max_episodes:
            print('reached max episodes', p.max_episodes, '=> terminating')
            self.finish = True

    def run_student(self):
        episode = self.episode
        stats = self.stats
        p = self.params

        # lets first take N_train meanings, and pass them through the teacher, to get a training set
        _train_meaning_idxes = torch.from_numpy(np.random.choice(self.N_train_val, self.N_train, replace=False))
        train_meanings = self.train_val_meanings[_train_meaning_idxes]

        # print('generating teacher utterances...', end='', flush=True)
        self.teacher.sender_agent.model.eval()
        train_utterances_logits = batched_run_nograd(
            params=p,
            model=self.teacher.sender_agent.model,
            inputs=train_meanings,
            batch_size=p.batch_size,
            input_batch_dim=0,
            output_batch_dim=1
        )
        _, train_utterances = train_utterances_logits.max(dim=-1)
        if p.remove_duplicates:
            # print('')
            # print('removing duplicates...')
            train_meanings, train_utterances, unique = remove_duplicates(
                batch_size=p.batch_size, meanings=train_meanings, utterances=train_utterances)
        # print(' ... done')

        student = SendReceivePair(params=p)

        def run_holdout(send_receive_pair):
            num_holdout_batches = self.N_holdout // p.batch_size
            holdout_acc_sum = 0
            utts_l, meanings_l = [], []
            for b in range(num_holdout_batches):
                b_start = b * p.batch_size
                b_end = b_start + p.batch_size
                meanings_batch = self.heldout_meanings[b_start:b_end]
                if p.enable_cuda:
                    meanings_batch = meanings_batch.cuda()
                utts_logits_batch = student.sender_agent.model(meanings=meanings_batch)
                if p.train_e2e:
                    acc_batch = link(
                        meanings=meanings_batch,
                        utts_logits=utts_logits_batch,
                        receiver_agent=send_receive_pair.receiver_agent,
                        sender_agent=send_receive_pair.sender_agent,
                        opt_both=None,
                        do_backward=False
                    )
                    holdout_acc_sum += acc_batch

                _, utts = utts_logits_batch.max(dim=-1)
                utts_l.append(utts.transpose(0, 1))
                meanings_l.append(meanings_batch)
            utts_t = torch.cat(utts_l)
            meanings_t = torch.cat(meanings_l)

            rho = metrics.topographic_similarity(utts_t, meanings_t)
            holdout_acc = holdout_acc_sum / num_holdout_batches
            uniqueness = metrics.uniqueness(utts_t)

            return rho, holdout_acc, uniqueness

        # and then train each half supervised on this data
        # sender first...
        sup_sender_epochs = 0
        sup_receiver_epochs = 0
        sup_sender_acc = 0
        sup_receiver_acc = 0
        sup_sender_time = 0
        sup_receiver_time = 0
        if (
                (p.sup_train_acc is not None and p.sup_train_acc > 0)
                or (p.sup_train_steps is not None and p.sup_train_steps > 0)
        ) and (episode > 0 or not p.train_e2e):
            for (agent_str, agent) in [('sender', student.sender_agent), ('recv', student.receiver_agent)]:
                # print('sup training on' + agent_str)
                _epoch = 0
                _last_print = time.time()
                _start = time.time()
                while True:
                    N_train = train_meanings.size(0)
                    num_batches = (N_train + p.batch_size - 1) // p.batch_size
                    epoch_acc_sum = 0
                    epoch_loss_sum = 0
                    for b in range(num_batches):
                        b_start = b * p.batch_size
                        b_end = b_start + p.batch_size
                        b_end = min(b_end, N_train)
                        b_utts = train_utterances[:, b_start:b_end]
                        b_meanings = train_meanings[b_start:b_end]
                        if p.enable_cuda:
                            b_utts = b_utts.cuda()
                            b_meanings = b_meanings.cuda()
                        b_loss, b_acc = agent.train(utts=b_utts, meanings=b_meanings)
                        epoch_loss_sum += b_loss * (b_end - b_start)
                        epoch_acc_sum += b_acc * (b_end - b_start)
                    epoch_loss = epoch_loss_sum / N_train
                    epoch_acc = epoch_acc_sum / N_train
                    _epoch += 1
                    _done_training = False
                    if p.sup_train_acc is not None and epoch_acc >= p.sup_train_acc:
                        # print('done sup training (reason: acc)')
                        _done_training = True
                    if p.sup_train_steps is not None and _epoch >= p.sup_train_steps:
                        # print('done sup training (reason: steps)')
                        _done_training = True
                    if _done_training or time.time() - _last_print >= 3.0:
                        print(f'sup {agent_str} e={_epoch} loss {epoch_loss:.3f} acc={epoch_acc:.3f}')
                        _last_print = time.time()
                    if _done_training:
                        # print('done training for agent', agent.__class__.__name__)
                        break
                if agent == student.sender_agent:
                    sup_sender_epochs = _epoch
                    sup_sender_acc = epoch_acc
                    sup_sender_time = time.time() - _start
                    stats.send_acc_sum += epoch_acc
                elif agent == student.receiver_agent:
                    sup_receiver_epochs = _epoch
                    sup_receiver_acc = epoch_acc
                    sup_receiver_time = time.time() - _start
                    stats.recv_acc_sum += epoch_acc
                else:
                    raise Exception('invalid agent value')
            # print('done supervised training')

        e2e_time = 0
        if p.train_e2e:
            # then train end to end for a bit, as decoder-encoder, looking at reconstruction accuracy
            # we'll do this on the same meanings as we got from the teacher? or different ones? or
            # just rnadomly sampled from everything except heldout?
            # maybe train on everything except holdout?
            epoch = 0
            Link = globals()[f'{p.link}Link']
            link = Link(params=p)
            last_print = time.time()
            _e2e_start = time.time()
            e2e_stats = Stats([
                'episodes_count',
                'e2e_acc_sum',
                'e2e_loss_sum'
            ])
            comms_dropout = nn.Dropout(p.comms_dropout)
            while True:
                # we'll just sample from all available meanings for a bit, train on that
                sample_idxes = torch.from_numpy(np.random.choice(self.N_train_val, p.batch_size, replace=False))
                meanings = self.train_val_meanings[sample_idxes]

                # then feed through the decoder/encoder
                if p.enable_cuda:
                    meanings = meanings.cuda()
                utts_logits = student.sender_agent.model(meanings=meanings)
                utts_logits = comms_dropout(utts_logits)
                ret = link(
                    meanings=meanings,
                    utts_logits=utts_logits,
                    receiver_agent=student.receiver_agent,
                    sender_agent=student.sender_agent,
                    opt_both=student.opt_both
                )
                if p.link == 'RL':
                    rewards_mean, rewards_std, loss, acc = ret
                else:
                    loss, acc = ret
                e2e_stats.episodes_count += 1
                e2e_stats.e2e_acc_sum += acc
                e2e_stats.e2e_loss_sum += loss
                _done_training = False
                if p.e2e_train_acc is not None and acc >= p.e2e_train_acc:
                    # print('reached target e2e acc %.3f' % acc, ' => breaking')
                    _done_training = True
                if p.e2e_train_steps is not None and epoch >= p.e2e_train_steps:
                    # print('reached target e2e step', epoch, ' => breaking')
                    _done_training = True
                if time.time() - last_print >= self.render_every_seconds or _done_training:
                    rho, holdout_acc, uniqueness = run_holdout(send_receive_pair=student)
                    acc = e2e_stats.e2e_acc_sum / e2e_stats.episodes_count
                    loss = e2e_stats.e2e_loss_sum / e2e_stats.episodes_count

                    _elapsed_time = time.time() - _e2e_start
                    log_dict = {
                        'record_type': 'e2e',
                        'ilm_epoch': episode,
                        'epoch': epoch,
                        'sps': int(epoch / _elapsed_time),
                        'elapsed_time': int(_elapsed_time),
                        'acc': acc,
                        'uniq': uniqueness,
                        'holdout_acc': holdout_acc,
                        'rho': rho,
                        'loss': loss
                    }
                    formatstr = (
                        'e2e e={epoch} i={ilm_epoch} '
                        't={elapsed_time:.0f} '
                        'sps={sps:.0f} '
                        'acc={acc:.3f} '
                        'loss={loss:.3f} '
                        'uniq={uniq:.3f} '
                        'holdout_acc={holdout_acc:.3f} '
                        'rho={rho:.3f} '
                    )
                    self.print_and_log(log_dict, formatstr=formatstr)

                    e2e_stats.reset()
                    last_print = time.time()
                epoch += 1
                if _done_training:
                    break
            e2e_time = time.time() - _e2e_start
            e2e_loss = loss
            e2e_acc = acc
            e2e_epochs = e2e_stats.episodes_count
            stats.e2e_acc_sum += acc

        # sample from the holdout set.  or just use entire holdout set perhaps?
        rho, holdout_acc, uniqueness = run_holdout(send_receive_pair=student)

        res = {
            'sup_sender_epochs': sup_sender_epochs,
            'sup_sender_acc': sup_sender_acc,
            'sup_sender_time': sup_sender_time,
            'sup_receiver_epochs': sup_receiver_epochs,
            'sup_receiver_acc': sup_receiver_acc,
            'sup_receiver_time': sup_receiver_time,
            'e2e_acc': e2e_acc,
            'e2e_loss': e2e_loss,
            'e2e_epochs': e2e_epochs,
            'e2e_time': e2e_time,
            'rho': rho,
            'uniqueness': uniqueness,
            'holdout_acc': holdout_acc
        }
        return student, res


if __name__ == '__main__':
    utils.clean_argv()
    runner = Runner()

    runner.add_param('--students-per-generation', type=int, default=1)
    runner.add_param('--student-compare-key', type=str, default='e2e_acc')

    runner.add_param('--seed', type=int)
    runner.add_param('--batch-size', type=int, default=128)
    runner.add_param('--sup-train-acc', type=float)
    runner.add_param('--sup-train-steps', type=int)
    runner.add_param('--e2e-train-acc', type=float)
    runner.add_param('--e2e-train-steps', type=int)
    runner.add_param('--max-episodes', type=int)
    runner.add_param('--opt', type=str, default='Adam')
    runner.add_param('--ent-reg', type=float, default=0)

    runner.add_param('--no-train-e2e', action='store_true')
    runner.add_param('--remove-duplicates', action='store_true', help='remove teacher duplicates')

    runner.add_param('--train-frac', type=float, default=0.4)
    runner.add_param('--num-holdout', type=int, default=128, help='never used for training, always same meanings')

    runner.add_param('--meanings', type=str, default='5x10')
    runner.add_param('--embedding-size', type=int, default=50)
    runner.add_param('--num-layers', type=int, default=1)
    runner.add_param('--utt-len', type=int, default=8)
    runner.add_param('--vocab-size', type=int, default=4)

    runner.add_param('--model', type=str, default='RNN')
    runner.add_param('--rnn-type', type=str, default='GRU')
    runner.add_param('--clip-grad', type=float, default=0)
    runner.add_param('--dropout', type=float, default=0)
    runner.add_param('--max-allowed-ngram-length', type=int, default=2)

    runner.add_param('--gumbel-temp', type=float, default=1.0)
    runner.add_param('--gumbel-hard', action='store_true')
    runner.add_param('--comms-dropout', type=float, default=0, help='between sender and receiver')

    runner.add_param('--rl-reward', type=str, default='count_meanings', help='[hard|ce|count_meanings]')

    runner.add_param('--link', type=str, default='Softmax', help='[Softmax|Gumbel|RL]')

    runner.parse_args()
    # runner.render_every_seconds = 0.01
    runner.params.num_meaning_types, runner.params.meanings_per_type = [
        int(v) for v in runner.params.meanings.split('x')]
    del runner.params.__dict__['meanings']
    utils.reverse_args(runner.params, 'no_train_e2e', 'train_e2e')
    print('runner.params', runner.params)
    runner.setup_base()
    runner.run_base()
