import gymnasium as gym
import numba
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.optim import Adam

from neu_ai.nn_utils import mlp, to_np
from neu_ai.RL.RLBase import RLBase


class PPOActor(nn.Module):
    log_std: tc.Tensor

    def __init__(s, d_obs, hid, d_act, act_type, Act):
        super().__init__()
        s.body = mlp([d_obs, *hid], Act, end=[Act()])
        s.head = nn.Linear(hid[-1], d_act)
        s.act_type = act_type
        if act_type is float:
            log_std = nn.Parameter(-0.5 * tc.ones(d_act))
            s.register_parameter("log_std", log_std)

    def forward(s, obs, act=None):
        x = s.body(obs)
        if s.act_type is float:
            pi = Normal(s.head(x), s.log_std.exp())
            act = pi.sample() if act is None else act
            logp = pi.log_prob(act).sum(-1)
        elif s.act_type is int:
            pi = Categorical(logits=s.head(x))
            act = pi.sample() if act is None else act
            logp = pi.log_prob(act)
        return act, logp


class PPO(RLBase):
    on_policy = True
    mem_size = 4000

    n_opt_step = 80
    target_kl = 0.01
    clip_ratio = 0.2
    lam = 0.97

    def __init__(
        s,
        env=gym.make("HalfCheetah-v5"),
        seed=0,
        hid=[64, 64],
        Act=nn.Tanh,
        actor_lr=3e-4,
        critic_lr=1e-3,
    ):
        d_obs, d_act, act_type = s.set_env(env, seed)
        s.actor = PPOActor(d_obs, hid, d_act, act_type, Act)
        s.critic = mlp([d_obs, *hid, 1], Act)
        s.actor_opt = Adam(s.actor.parameters(), actor_lr)
        s.critic_opt = Adam(s.critic.parameters(), critic_lr)

    def get_act(s, obs):
        return s.actor(obs)[0]

    def get_V(s, obs):
        return tc.squeeze(s.critic(obs), -1)

    def learn_from_memory(s):
        with tc.no_grad():
            # obs, act, next_obs, rew, term, trunc = s.memory
            obs, act, next_obs = map(tc.from_numpy, s.memory[:3])
            rew, term, trunc = s.memory[3:]

            V, next_V = map(s.get_V, (obs, next_obs))
            V, next_V = map(to_np, (V, next_V))
            adv, ret = ppo_adv_ret(rew, term, trunc, V, next_V, s.gam, s.lam)
            adv, ret = map(tc.from_numpy, (adv, ret))
            logp_old = s.actor(obs, act)[1]

        for _ in range(s.n_opt_step):
            logp = s.actor(obs, act)[1]
            kl = tc.mean(logp_old - logp)
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


@numba.njit
def ppo_adv_ret(rew, term, trunc, V, next_V, gam, lam):
    T = len(rew)
    adv = np.zeros(T, dtype=np.float32)
    ret = np.zeros(T, dtype=np.float32)
    for t in range(T - 1, -1, -1):
        td_err = rew[t] + gam * next_V[t] * (1 - term[t]) - V[t]
        if term[t]:
            next_adv, next_ret = 0, 0
        elif t == T - 1 or trunc[t]:
            next_adv, next_ret = 0, next_V[t]
        else:
            next_adv, next_ret = adv[t + 1], ret[t + 1]
        adv[t] = td_err + gam * lam * next_adv
        ret[t] = rew[t] + gam * next_ret
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret


if __name__ == "__main__":
    ppo = PPO()
    ppo.run()

"""
step: 989000, eps_ret: 1652.3620277037098
step: 990000, eps_ret: 2046.193352009298
step: 991000, eps_ret: 2665.7532212771403
step: 992000, eps_ret: 2167.9729595426525
actor_loss: -0.031430037976004625, critic_loss: 1244.3602294921875
step: 993000, eps_ret: 2566.200958702792
step: 994000, eps_ret: 2579.5729588507465
step: 995000, eps_ret: 864.0350353725199
step: 996000, eps_ret: 2532.0488680306553
actor_loss: -0.020459600404258878, critic_loss: 502.8697509765625
step: 997000, eps_ret: 1276.5223480369687
step: 998000, eps_ret: 1219.594475779663
step: 999000, eps_ret: 946.6770729546146
step: 1000000, eps_ret: 1662.9552099059229
actor_loss: -0.006879132759010595, critic_loss: 631.43798828125
"""
