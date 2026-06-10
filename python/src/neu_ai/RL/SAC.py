from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box
from torch.distributions import Normal
from torch.optim import Adam

from neu_ai.nn_utils import mlp
from neu_ai.RL.RLBase import RLBase


class SACActor(nn.Module):
    def __init__(s, d_obs, hid, d_act, act_type, Act):
        assert act_type is float
        super().__init__()
        s.body = mlp([d_obs, *hid], Act, end=[Act()])
        s.mu = nn.Linear(hid[-1], d_act)
        s.log_std = nn.Linear(hid[-1], d_act)

    def forward(s, obs, act=None):
        x = s.body(obs)
        mu = s.mu(x)
        std = tc.clamp(s.log_std(x), -20, 2).exp()
        pi = Normal(mu, std)
        act = pi.rsample()
        logp = pi.log_prob(act).sum(-1)
        logp -= (2 * (np.log(2) - act - F.softplus(-2 * act))).sum(-1)
        return tc.tanh(act), logp


class SACActorCritic(nn.Module):
    def __init__(s, d_obs, hid, d_act, act_type, Act):
        super().__init__()
        s.actor = SACActor(d_obs, hid, d_act, act_type, Act)
        s.q1 = mlp([d_obs + d_act, *hid, 1], Act)
        s.q2 = mlp([d_obs + d_act, *hid, 1], Act)

    def q1q2(s, obs, act):
        x = tc.cat([obs, act], dim=-1)
        return tc.squeeze(s.q1(x), -1), tc.squeeze(s.q2(x), -1)


class SAC(RLBase):
    on_policy = False
    mem_size = int(1e6)

    batch_size = 100
    alpha = 0.2
    polyak = 0.995

    def __init__(
        s,
        env=gym.make("HalfCheetah-v5"),
        seed=0,
        hid=[64, 64],
        Act=nn.ReLU,
        lr=1e-3,
    ):
        sp: Box = env.action_space
        assert np.all(sp.low == -1) and np.all(sp.high == 1)

        d_obs, d_act, act_type = s.set_env(env, seed)
        s.ac = SACActorCritic(d_obs, hid, d_act, act_type, Act)
        s.ac_tar = deepcopy(s.ac)
        s.ac_tar.requires_grad_(False)
        s.q_params = [*s.ac.q1.parameters(), *s.ac.q2.parameters()]
        s.actor_opt = Adam(s.ac.actor.parameters(), lr)
        s.critic_opt = Adam(s.q_params, lr)

    def get_act(s, obs):
        return s.ac.actor(obs)[0]

    def learn_from_memory(s):
        mem = s.sample_mem(s.batch_size)
        obs, act, next_obs, rew, term, trunc = map(tc.from_numpy, mem)

        q1, q2 = s.ac.q1q2(obs, act)
        with tc.no_grad():
            next_act, next_logp = s.ac.actor(next_obs)
            next_q = tc.min(*s.ac_tar.q1q2(next_obs, next_act))
            q_tar = rew + s.gam * (1 - term) * (next_q - s.alpha * next_logp)
        q_loss = F.mse_loss(q1, q_tar) + F.mse_loss(q2, q_tar)

        s.critic_opt.zero_grad()
        q_loss.backward()
        s.critic_opt.step()

        for p in s.q_params:
            p.requires_grad = False

        act, logp = s.ac.actor(obs)
        q = tc.min(*s.ac.q1q2(obs, act))
        actor_loss = tc.mean(s.alpha * logp - q)

        s.actor_opt.zero_grad()
        actor_loss.backward()
        s.actor_opt.step()

        for p in s.q_params:
            p.requires_grad = True

        with tc.no_grad():
            for p, p_tar in zip(s.ac.parameters(), s.ac_tar.parameters()):
                p_tar.data.mul_(s.polyak)
                p_tar.data.add_((1 - s.polyak) * p.data)


if __name__ == "__main__":
    sac = SAC()
    sac.run()
