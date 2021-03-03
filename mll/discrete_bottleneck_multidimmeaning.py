"""
takes a discrete value, concretely a 1-in-12 one-hot vector, or index equivalent,
passes through a discrete/utterance bottleneck, and reconstructs on the other side

we're going to look at lexicon size. ideally we'd hope to find it is 12. but we probably
wont :P

this file was forked from game1/embeddings/test_discrete_bottleneck_discrete_input.py,
to discrete_bottleneck_discrete_input.py,
and then forked again, to this file

This file makes the meanings multidimensional (5 dimensional), so that non-compositional
becomes ridiculous
"""
import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np

# from envs.world3c import World
from ulfs import alive_sieve, rl_common
from ulfs.stats import Stats
from ulfs.stochastic_trajectory import StochasticTrajectory
from ulfs.lexicon_recorder import LexiconRecorder
from ulfs.runner_base_v1 import RunnerBase


class MeaningEmbedder(nn.Module):
    """
    we'll take in random sample of multidimensional discrete meanings,
    and embed each meaning dimension separately, and add the embeddings
    (or should we feed through an LSTM??? I guess we can try both...
    let's feed through an LSTM for now; more flexible...;
    hmmm...
    that kind of introduces bias...
    let's just add them :P)
    """
    def __init__(self):
        pass

    # def forward(self, meanings):


class AgentOne(nn.Module):
    """
    takes in a discrete action (1-in-k), converts to utterance
    """
    def __init__(self, embedding_size, utterance_max, vocab_size, num_actions):
        """
        Note that vocab_size includes terminator character 0
        """
        self.embedding_size = embedding_size
        self.utterance_max = utterance_max
        self.num_actions = num_actions
        super().__init__()

        self.h1 = nn.Embedding(num_actions, embedding_size)

        # d2e "discrete to embed"
        self.d2e = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRUCell(embedding_size, embedding_size)
        self.e2d = nn.Linear(embedding_size, vocab_size)

    def forward(self, actions, global_idxes):
        """
        This agent will receive the image of the world
        x might have been sieved. global_idxes too. but global_idxes contents are the global
        indexes
        """
        batch_size = actions.size()[0]

        x = self.h1(actions)

        state = x
        global_idxes = global_idxes.clone()
        # note that this sieve might start off smaller than the global batch_size
        sieve = alive_sieve.AliveSieve(batch_size=batch_size, enable_cuda=x.is_cuda)
        type_constr = torch.cuda if x.is_cuda else torch
        last_token = type_constr.LongTensor(batch_size).fill_(0)
        utterance = type_constr.LongTensor(batch_size, self.utterance_max).fill_(0)
        # N_outer might not be the full episode batch size, but a subset
        N_outer = type_constr.LongTensor(batch_size).fill_(self.utterance_max)
        stochastic_trajectory = StochasticTrajectory()
        for t in range(self.utterance_max):
            emb = self.d2e(last_token)
            state = self.rnn(emb, state)
            token_logits = self.e2d(state)
            token_probs = F.softmax(token_logits, dim=-1)

            if self.training:
                s = rl_common.draw_categorical_sample(
                    action_probs=token_probs, batch_idxes=global_idxes[sieve.global_idxes])
                stochastic_trajectory.append_stochastic_sample(s=s)
                token = s.actions.view(-1)
                # print('stochastic')
                # print('token.size()', token.size())
                # die()
            else:
                # print('argmax')
                _, token = token_probs.max(-1)
                # print('token.size()', token.size())
                # die()

            utterance[:, t][sieve.global_idxes] = token
            last_token = token
            sieve.mark_dead(last_token == 0)
            sieve.set_global_dead(N_outer, t + 1)
            if sieve.all_dead():
                break
            state = state[sieve.alive_idxes]
            last_token = last_token[sieve.alive_idxes]
            sieve.self_sieve_()
        res = {
            'stochastic_trajectory': stochastic_trajectory,
            'utterance': utterance,
            'utterance_lens': N_outer
        }
        return res


class AgentTwo(nn.Module):
    def __init__(self, embedding_size, vocab_size, num_actions):
        """
        - input: utterance
        - output: action
        """
        super().__init__()
        self.num_actions = num_actions
        self.embedding_size = embedding_size
        self.d2e = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRUCell(embedding_size, embedding_size)

        self.h1 = nn.Linear(embedding_size, num_actions)

    def forward(self, utterance, global_idxes):
        """
        utterance etc might be sieved, which is why we receive global_idxes
        alive_masks will then create subsets of this already-sieved set
        """
        batch_size = utterance.size()[0]
        utterance_max = utterance.size()[1]
        type_constr = torch.cuda if utterance.is_cuda else torch
        sieve = alive_sieve.AliveSieve(batch_size=batch_size, enable_cuda=utterance.is_cuda)
        state = type_constr.FloatTensor(batch_size, self.embedding_size).fill_(0)
        output_state = state.clone()
        for t in range(utterance_max):
            emb = self.d2e(utterance[:, t])
            state = self.rnn(emb, state)
            output_state[sieve.global_idxes] = state
            sieve.mark_dead(utterance[:, t] == 0)
            if sieve.all_dead():
                break

            utterance = utterance[sieve.alive_idxes]
            state = state[sieve.alive_idxes]
            sieve.self_sieve_()
        state = output_state

        action_logits = self.h1(state)
        action_probs = F.softmax(action_logits, dim=-1)

        s = rl_common.draw_categorical_sample(
            action_probs=action_probs, batch_idxes=global_idxes)
        return s


def run_episode(actions, one, two, utterance_len_reg, enable_cuda, render=False):
    batch_size = actions.size()[0]
    global_idxes = torch.LongTensor(batch_size).fill_(1).cumsum(-1) - 1

    one_res = one(
        actions=actions, global_idxes=global_idxes)
    one_stochastic_trajectory, utterances, utterances_lens = map(one_res.__getitem__, [
        'stochastic_trajectory', 'utterance', 'utterance_lens'
    ])
    two_s = two(
        utterance=utterances, global_idxes=global_idxes)

    stats = Stats([])
    episode_result = {
        'one_stochastic_trajectory': one_stochastic_trajectory,
        'two_s': two_s,
        'utterances': utterances,
        'utterances_lens': utterances_lens,
        'stats': stats
    }
    return episode_result


class Runner(RunnerBase):
    def __init__(self):
        super().__init__(
            save_as_statedict_keys=['one', 'two', 'opt_one', 'opt_two'],
            additional_save_keys=['baseline'],
            step_key='episode'
        )

    def setup(self, p):
        num_actions = p.num_actions

        self.lexicon_recorder = LexiconRecorder(num_actions=num_actions)
        self.argmaxed_lexicon_recorder = LexiconRecorder(num_actions=num_actions)

        self.one = AgentOne(
            embedding_size=p.embedding_size, vocab_size=p.vocab_size, utterance_max=p.utterance_max,
            num_actions=num_actions)
        self.two = AgentTwo(
            embedding_size=p.embedding_size, vocab_size=p.vocab_size, num_actions=num_actions)
        if p.enable_cuda:
            self.one = self.one.cuda()
            self.two = self.two.cuda()
        self.opt_one = optim.Adam(lr=0.001, params=self.one.parameters())
        self.opt_two = optim.Adam(lr=0.001, params=self.two.parameters())

        self.stats = Stats([
            'batches_count',
            'episodes_count',
            'len_sum', 'len_count'
        ])
        self.baseline = 0

    def step(self, p):
        render = self.should_render()

        actions_in = torch.from_numpy(
                np.random.choice(p.num_actions, p.batch_size, replace=True)
        ).long()
        if p.enable_cuda:
            actions_in = actions_in.cuda()
        episode_result = run_episode(
            actions=actions_in,
            one=self.one, two=self.two,
            render=render, utterance_len_reg=p.utterance_len_reg, enable_cuda=p.enable_cuda)
        utterances, utterances_lens, _stats = map(episode_result.__getitem__, [
            'utterances', 'utterances_lens', 'stats'
        ])
        one_stochastic_trajectory, two_s = map(episode_result.__getitem__, [
            'one_stochastic_trajectory', 'two_s'
        ])
        self.lexicon_recorder.record(
            action_probs_l=[two_s.action_probs], utterances_by_t=[utterances], utterance_lens_by_t=[utterances_lens])
        self.stats += _stats

        self.stats.len_sum += utterances_lens.sum().item()
        self.stats.len_count += len(utterances_lens)

        self.stats.episodes_count += 1

        rewards = (two_s.actions == actions_in).float() * 2 - 1
        rewards -= utterances_lens.float() * p.utterance_len_reg
        baselined_rewards = rewards - self.baseline

        reinforce_loss_one = one_stochastic_trajectory.calc_loss(baselined_rewards)
        reinforce_loss_two = two_s.calc_loss(baselined_rewards)

        ent_loss_one = - p.entropy_reg * one_stochastic_trajectory.entropy
        ent_loss_two = - p.entropy_reg * two_s.entropy

        loss_one = reinforce_loss_one + ent_loss_one
        loss_two = reinforce_loss_two + ent_loss_two

        self.opt_one.zero_grad()
        loss_one.backward()
        self.opt_one.step()

        self.opt_two.zero_grad()
        loss_two.backward()
        self.opt_two.step()

        self.baseline = 0.7 * self.baseline + 0.3 * rewards.mean().item()

        # ========================
        self.one.eval()
        self.two.eval()
        episode_result = run_episode(
            actions=actions_in,
            one=self.one, two=self.two,
            render=render, utterance_len_reg=p.utterance_len_reg, enable_cuda=p.enable_cuda)
        utterances, utterances_lens, _stats = map(episode_result.__getitem__, [
            'utterances', 'utterances_lens', 'stats'
        ])
        one_stochastic_trajectory, two_s = map(episode_result.__getitem__, [
            'one_stochastic_trajectory', 'two_s'
        ])
        self.argmaxed_lexicon_recorder.record(
            action_probs_l=[two_s.action_probs], utterances_by_t=[utterances], utterance_lens_by_t=[utterances_lens])
        self.one.train()
        self.two.train()

        if render:
            lex_stats = self.lexicon_recorder.calc_stats()
            argmaxed_lex_stats = self.argmaxed_lexicon_recorder.calc_stats()
            self.lexicon_recorder.reset()
            self.argmaxed_lexicon_recorder.reset()

            stats = self.stats
            log_dict = {
                'avg_reward': rewards.mean().item(),
                'avg_utt_len': stats.len_sum / stats.len_count,
                'lex_size': lex_stats['total_unique'],
                'argmaxed_lex_size': argmaxed_lex_stats['total_unique']
            }
            for k, v in lex_stats.items():
                log_dict[k] = v
            self.print_and_log(
                log_dict,
                formatstr='e={episode} '
                          'avguttlen {avg_utt_len:.2f} '
                          'r {avg_reward:.3f} '
                          'lex_size {lex_size} '
                          'argmaxed_lex_size {argmaxed_lex_size} '
            )
            stats.reset()


if __name__ == '__main__':
    runner = Runner()
    runner.add_param('--embedding-size', type=int, default=128)
    runner.add_param('--num-meanings-per-type', type=str, default=10)
    runner.add_param('--num-meaning-types', type=str, default=5)
    runner.add_param('--bits-per-meaning-type', type=str, default=6)
    runner.add_param('--utterance-max', type=int, default=10)
    runner.add_param('--utterance-len-reg', type=float, default=0.0, help='how much to penalize longer utterances')
    runner.add_param('--entropy-reg', type=int, default=0.05)
    runner.add_param('--vocab-size', type=int, default=20, help='includes terminator')
    runner.add_param('--batch-size', type=int, default=128)
    runner.parse_args()
    runner.setup_base()
    runner.run_base()
