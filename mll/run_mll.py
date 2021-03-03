# flake8: noqa
"""
agents in a soup/bucket together
have to learn to come to some consensus together on language
not a million miles away from "Tipping points in social convention"

this is copied from game1/run_agent_soup.py

I quite like the name agent soup actually... and also the idea I didnt remember I had of relating
this to 'tipping points in social convention' :)
"""
import time
import argparse
import datetime
import os
from os import path
import json

import numpy as np
import torch
from torch import nn, optim

from mylib.params import Params
from mylib.logger import Logger
from mylib import name_utils
from mylib.runner_base_v1 import RunnerBase


class Conciousness(nn.Module):
    """
    Decides whether to train on an existing buffer example, or to draw a new rl sample
    bases this on things like:
    - how many buffer examples drawn so far
    - training loss on buffer examples
    - .... ?

    actions are therefore:
    - 0 => draw sample from replay buffer, or
    - 1 => draw sample from environment
    """
    def __init__(self, context_size, num_actions=2, embedding_size=50):
        super().__init__()
        self.h1 = nn.Linear(context_size, embedding_size)
        self.h2 = nn.Linear(embedding_size, num_actions)

    def forward(self, context):
        x = self.h1(context)
        x = F.tanh(x)
        x = self.h2(x)
        probs = F.softmax(x)

        s = rl_common.draw_categorical_sample(probs, batch_idxes=None)
        return s


        # - number of around-the-tables run so far
        # - how many agents in total
        # - how many agents voted for trial last time
        # - how many trials the agent has run so far
        # - the result of the last trial


class Speaker(nn.Module):
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

            s = rl_common.draw_categorical_sample(
                action_probs=token_probs, batch_idxes=global_idxes[sieve.global_idxes])
            stochastic_trajectory.append_stochastic_sample(s=s)

            token = s.actions.view(-1)
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


class Listener(nn.Module):
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


class Agent(object):
    """
    Note: not a network module

    Can return/load a state_dict though, and handle .cuda()
    """
    def __init__(self, p, consciousness):
        self.speaker = Speaker(
            embedding_size=p.embedding_size, vocab_size=p.vocab_size, utterance_max=p.utterance_max,
            num_actions=p.num_actions)
        self.listener = Listener(
            embedding_size=p.embedding_size, vocab_size=p.vocab_size, num_actions=p.num_actions)
        self.consciousness = consciousness

        self.speaker_opt = optim.Adam(lr=0.001, params=self.speaker.parameters())
        self.listener_opt = optim.Adam(lr=0.001, params=self.listener.parameters())

        self.time_since_last_trial = torch.FloatTensor()
        self.won_last_trial = 0

    def cuda(self):
        self.speaker = self.speaker.cuda()
        self.listener = self.listener.cuda()

    def state_dict(self):
        state_dict = {}
        state_dict['speaker_state'] = self.speaker.state_dict()
        state_dict['listener_state'] = self.listener.state_dict()
        state_dict['speaker_opt_state'] = self.speaker_opt.state_dict()
        state_dict['listener_opt_state'] = self.listener_opt.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.speaker.load_state_dict(state_dict['speaker_state'])
        self.listener.load_state_dict(state_dict['listener_state'])
        self.speaker_opt.load_state_dict(state_dict['speaker_opt_state'])
        self.listener_opt.load_state_dict(state_dict['listener_opt_state'])

    def _get_context(self):
        raise Exception('not implemented')
        pass

    def bid(self):
        """
        in a turn, decide whether to reflect, run a trial, or watch
        """


class Season(object):
    def __init__(self, p, consciousness):
        self.p = p
        self.consciousness = consciousness
        self.agents = []
        for i in range(self.num_agents_per_pool):
            agent = Agent(p=p)
            self.agents.append(agent)

    def run(self):
        for turn_id in range(self.p.turns_per_season):
            turn = Turn(p=p, consciousness=self.consciousness, season=season)
            turn.run()
            # self.run_episode(season)


class Turn(object):
    def __init__(self, p, consciousness, season):
        self.p = p
        self.consciousness = consciousness
        self.season = season

    def run(self):
        for i, agent in enumerate(season.agents):
            bid = agent.bid()


def run_episode(actions, one, two, utterance_len_reg, enable_cuda, render=False):
    type_constr = torch.cuda if enable_cuda else torch

    batch_size = actions.size()[0]
    global_idxes = torch.LongTensor(batch_size).fill_(1).cumsum(-1) - 1

    t = 0
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
            save_as_statedict_keys=[
                'consciousness', 'consciousness_opt'
            ],
            additional_save_keys=['baseline'],
            step_key='season'
        )

    def setup(self, p):
        num_actions = p.num_actions

        self.lexicon_recorder = LexiconRecorder(num_actions=num_actions)

        # context will be:
        # time since last trial (as a real)
        # result of last trial (win/lose, one_hot)
        # self.agent = Agent(p=p)
        self.consciousness = Conciousness(context_size=3, embedding_size=18)
        self.consciousness_opt = optim.Adam(lr=0.001, params=self.consciousness.parameters())

        self.stats = Stats([
            'batches_count',
            'episodes_count',
            'len_sum', 'len_count'
        ])
        self.baseline = 0

    def step(self, p):
        """
        A step is a single season. Let's think/doc this through

        In a season, we take a bunch of agents. The agents are initialized randomly, and independently
        - we then go around the table of agents
        - each agent can choose to think (run against its replay buffer), or to run a trial
        - if no agents have run trial yet, obviously thinking is impossible:
             - each agent loses 0.1 reward
             - go to next around-the-table
        - if two or more agents chose to run a trial; pair up randomly. if an odd number, one will be left out
        - the agents run their trials
        - after the trials ran, each agent is given access to what happened (the entire trajectories), and allowed to think for one step
           (the ones who ran the trial already trained on that trial, not given additional learning time or steps)
        - if only one agent chooses to run a trial, that agent is treated as per 'odd number' above

        The input to the consciousness of each agent, to decide whether to think or run a trial is in theory all the context/information
        up to that point

        In practice, for simplicity, at least to start with, let's engineer that a bit :P
        - number of around-the-tables run so far
        - how many agents in total
        - how many agents voted for trial last time
        - how many trials the agent has run so far
        - the result of the last trial

        actually, let's not. lets just feed it through an rnn... what data do we have available? (without thinking about 
        how to represent it, or if we can)

        - how each agent voted
        - which agents ran trial
        - which agents paired together
        - the actions and outcomes of each pairing

        wait, no, let's keep it simple for now. then go rnn later; feature engineer for now, who/check it works
        """
        # season_id = self.season
        # self.run_season(season_id)
        season = Season(p=p)
        season.run()
        # render = self.should_render()

    # def create_season(self):
    #     class Season(object):
    #         def __init__(self):
    #             pass

    #     p = self.params
    #     season = Season(p=p)

    # def run_season(self, season_id):
    #     season = self.create_season()
    #     for episode_id in range(self.params.episodes_per_season):
    #         self.run_episode(season)

    def run_turn(self, season):
        pass

    def run_match(self, one, two):
        # so, we will team up agent_one and agent_two
        # agent_one will recieve a batch of discrete 1-in-k, convert to utterance, and
        # send to agent_two
        # agent_two has to predict what was the original bathc of discrete 1-in-k
        # no other information is required I think (dont need to know season etc)
        p = self.params
        render = False
        actions_in = torch.from_numpy(
                np.random.choice(p.num_actions, p.batch_size, replace=True)
        ).long()
        if p.enable_cuda:
            actions_in = actions_in.cuda()
        episode_result = self.run_episode(
            actions=actions_in,
            one=one, two=two,
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

        return rewards.mean().item()

    def run_episode(self, actions, one, two):
        type_constr = torch.cuda if self.enable_cuda else torch

        batch_size = actions.size()[0]
        global_idxes = torch.LongTensor(batch_size).fill_(1).cumsum(-1) - 1

        t = 0
        one_res = one(
            actions=actions, global_idxes=global_idxes)
        one_stochastic_trajectory, utterances, utterances_lens = map(one_res.__getitem__, [
            'stochastic_trajectory', 'utterance', 'utterance_lens'
        ])
        two_s = two(
            utterance=utterances, global_idxes=global_idxes)

        # stats = Stats([])
        episode_result = {
            'one_stochastic_trajectory': one_stochastic_trajectory,
            'two_s': two_s,
            'utterances': utterances,
            'utterances_lens': utterances_lens
            # 'stats': stats
        }
        return episode_result


if __name__ == '__main__':
    runner = Runner()
    runner.add_param('--num-agents-per-pool', type=int, default=5)
    runner.add_param('--turns-per-season', type=int, default=100)
    runner.add_param('--episodes-per-match', type=int, default=1)
    runner.add_param('--embedding-size', type=int, default=50)
    runner.add_param('--num-actions', type=int, default=12)
    runner.add_param('--utterance-max', type=int, default=6)
    runner.add_param('--utterance-len-reg', type=float, default=0.01, help='how much to penalize longer utterances')
    runner.add_param('--entropy-reg', type=int, default=0.05)
    runner.add_param('--vocab-size', type=int, default=4, help='includes terminator')
    runner.add_param('--batch-size', type=int, default=128)
    runner.parse_args()
    runner.setup_base()
    runner.run_base()
