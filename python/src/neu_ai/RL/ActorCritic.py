from typing import List

import gymnasium as gym
import numpy as np
import torch as tc
import torch.nn as nn
from gymnasium.spaces import Box, Discrete
from torch.distributions import Categorical, Normal
from torch.optim import Adam

from neu_ai.nn_utils import mlp


class ActorCritic:
    mem_size = 4000
    gam = 0.99

    def __init__(
        s,
        env=gym.make("HalfCheetah-v5"),
        seed=0,
        hidden=[64, 64],
        Act=nn.Tanh,
        actor_lr=3e-4,
        critic_lr=1e-3,
    ):
        s.env = env
        tc.manual_seed(seed)
        np.random.seed(seed)
        s.env.reset(seed=seed)

        # Hippocampus 海马体:
        s.memory: List[np.ndarray] = []
        s.step_cnt = 0

        d_obs = s.env.observation_space.shape[0]
        s.act_sp = s.env.action_space
        if isinstance(s.act_sp, Box):
            d_act = s.act_sp.shape[0]
        elif isinstance(s.act_sp, Discrete):
            d_act = s.act_sp.n

        # BasalGanglia.Striatum.Matrix 基底核.纹状体.基质:
        s.critic = mlp([d_obs, *hidden, 1], Act)
        # BasalGanglia.Striatum.Striosome 基底核.纹状体.小体:
        s.actor = mlp([d_obs, *hidden, d_act], Act)
        log_std = nn.Parameter(-0.5 * tc.ones(d_act))
        s.actor.register_parameter("log_std", log_std)

        s.actor_opt = Adam(s.actor.parameters(), lr=actor_lr)
        s.critic_opt = Adam(s.critic.parameters(), lr=critic_lr)

    def collect_memory(s):
        obs, _ = s.env.reset()
        eps_ret = 0
        for i in range(s.mem_size):
            act = s.get_act(obs)
            next_obs, rew, term, trunc, _ = s.env.step(act)
            s.memorize(i, [obs, act, next_obs, rew, term, trunc])
            eps_ret += rew
            s.step_cnt += 1
            if term or trunc:
                obs, _ = s.env.reset()
                print(f"step_cnt: {s.step_cnt}, eps_ret: {eps_ret}")
                eps_ret = 0
            else:
                obs = next_obs

    def memorize(s, i, values):
        if not len(s.memory):
            for v in values:
                shape = (s.mem_size, *np.shape(v))
                s.memory.append(np.zeros(shape, dtype=np.float32))
        for k, v in enumerate(values):
            s.memory[k][i] = v

    @tc.no_grad()
    def get_act(s, obs):
        obs = tc.from_numpy(obs).float()
        if isinstance(s.act_sp, Box):
            pi = Normal(s.actor(obs), tc.exp(s.actor.log_std))
        elif isinstance(s.act_sp, Discrete):
            pi = Categorical(logits=s.actor(obs))
        return pi.sample().numpy()

    def get_logp(s, obs, act):
        if isinstance(s.act_sp, Box):
            pi = Normal(s.actor(obs), tc.exp(s.actor.log_std))
            return pi.log_prob(act).sum(-1)
        elif isinstance(s.act_sp, Discrete):
            pi = Categorical(logits=s.actor(obs))
            return pi.log_prob(act)

    def get_V(s, obs):
        return tc.squeeze(s.critic(obs), -1)

    def learn_from_memory(s):
        raise NotImplementedError

    def run(s, n_step=1e6):
        while s.step_cnt < n_step:
            s.collect_memory()
            s.learn_from_memory()
