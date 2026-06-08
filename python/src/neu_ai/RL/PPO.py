from typing import List

import gymnasium as gym
import numba
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

from neu_ai.nn_utils import mlp


def to_np(x: tc.Tensor):
    return x.numpy()


class ActorCritic:
    def __init__(s):
        s.env = gym.make("HalfCheetah-v5")
        s.Act = nn.Tanh
        s.seed = 0
        s.n_step = 1e6
        s.n_opt_step = 80
        s.mem_size = 4000
        s.gam = 0.99
        s.lam = 0.97
        s.target_kl = 0.01
        s.clip_ratio = 0.2

        tc.manual_seed(s.seed)
        np.random.seed(s.seed)
        s.env.reset(seed=s.seed)

        s.memory: List[np.ndarray] = []
        s.step_cnt = 0

        d_obs = s.env.observation_space.shape[0]
        d_act = s.env.action_space.shape[0]

        s.critic = mlp([d_obs, 64, 64, 1], s.Act)
        s.actor = mlp([d_obs, 64, 64, d_act], s.Act)
        log_std = nn.Parameter(-0.5 * tc.ones(d_act))
        s.actor.register_parameter("log_std", log_std)

        s.actor_opt = Adam(s.actor.parameters())
        s.critic_opt = Adam(s.critic.parameters())

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
        pi = Normal(s.actor(obs), tc.exp(s.actor.log_std))
        return pi.sample().numpy()

    def get_logp(s, obs, act):
        pi = Normal(s.actor(obs), tc.exp(s.actor.log_std))
        return pi.log_prob(act).sum(-1)

    def get_V(s, obs):
        return s.critic(obs)[:, 0]

    def learn_from_memory(s):
        with tc.no_grad():
            # obs, act, next_obs, rew, term, trunc = s.memory
            obs, act, next_obs = map(tc.from_numpy, s.memory[:3])
            rew, term, trunc = s.memory[3:]

            V, next_V = map(s.get_V, (obs, next_obs))
            V, next_V = map(to_np, (V, next_V))
            adv, ret = ppo_adv_ret(rew, term, trunc, V, next_V, s.gam, s.lam)
            adv, ret = map(tc.from_numpy, (adv, ret))
            logp_old = s.get_logp(obs, act)

        for _ in range(s.n_opt_step):
            logp = s.get_logp(obs, act)
            kl = (logp_old - logp).mean()
            if kl > 1.5 * s.target_kl:
                break
            ratio = tc.exp(logp - logp_old)
            clip_r = tc.clamp(ratio, 1 - s.clip_ratio, 1 + s.clip_ratio)
            actor_loss = -(tc.min(ratio * adv, clip_r * adv)).mean()
            s.actor_opt.zero_grad()
            actor_loss.backward()
            s.actor_opt.step()

        for _ in range(s.n_opt_step):
            critic_loss = F.mse_loss(s.get_V(obs), ret)
            s.critic_opt.zero_grad()
            critic_loss.backward()
            s.critic_opt.step()

        print(f"actor_loss: {actor_loss}, critic_loss: {critic_loss}")

    def run(s):
        while s.step_cnt < s.n_step:
            s.collect_memory()
            s.learn_from_memory()


@numba.njit
def ppo_adv_ret(rew, term, trunc, V, next_V, gam, lam):
    T = len(rew)
    adv = np.zeros(T, dtype=np.float32)
    ret = np.zeros(T, dtype=np.float32)
    for t in range(T - 1, -1, -1):
        td_err = rew[t] + gam * next_V[t] * (1 - term[t]) - V[t]
        if t == T - 1 or term[t] or trunc[t]:
            next_adv, next_ret = 0, 0
        else:
            next_adv, next_ret = adv[t + 1], ret[t + 1]
        adv[t] = td_err + gam * lam * next_adv
        ret[t] = rew[t] + gam * next_ret
    return adv, ret


if __name__ == "__main__":
    ac = ActorCritic()
    ac.run()

"""
using ReLU:
step_cnt: 989000, eps_ret: 1547.7574297572628
step_cnt: 990000, eps_ret: 1543.7184097482661
step_cnt: 991000, eps_ret: 1789.5662179863832
step_cnt: 992000, eps_ret: 1483.7044744785414
actor_loss: -3.586716413497925, critic_loss: 1241.2152099609375
step_cnt: 993000, eps_ret: 1590.6616158790546
step_cnt: 994000, eps_ret: 1644.5497730226207
step_cnt: 995000, eps_ret: 1593.651727476987
step_cnt: 996000, eps_ret: 1854.2449030636492
actor_loss: -5.355538368225098, critic_loss: 1549.6956787109375
step_cnt: 997000, eps_ret: 1740.2928368916453
step_cnt: 998000, eps_ret: 1707.827679076744
step_cnt: 999000, eps_ret: 1753.0616736776747
step_cnt: 1000000, eps_ret: 1494.0993530010605
actor_loss: -4.089791774749756, critic_loss: 1457.18408203125


using Tanh:
step_cnt: 989000, eps_ret: 1874.5549389077771
step_cnt: 990000, eps_ret: 1983.5296331224183
step_cnt: 991000, eps_ret: 2026.9806721950135
step_cnt: 992000, eps_ret: 1958.7981243126394
actor_loss: -4.591017723083496, critic_loss: 1985.41845703125
step_cnt: 993000, eps_ret: 2007.2393171840376
step_cnt: 994000, eps_ret: 1963.7229473297293
step_cnt: 995000, eps_ret: 1890.3192891013575
step_cnt: 996000, eps_ret: 1933.5531001293127
actor_loss: -4.1950178146362305, critic_loss: 1824.7340087890625
step_cnt: 997000, eps_ret: 1957.6206318330733
step_cnt: 998000, eps_ret: 1861.5483210175507
step_cnt: 999000, eps_ret: 1874.6255172288181
step_cnt: 1000000, eps_ret: 1864.8998433288539
actor_loss: -4.388137340545654, critic_loss: 1618.2156982421875
"""
