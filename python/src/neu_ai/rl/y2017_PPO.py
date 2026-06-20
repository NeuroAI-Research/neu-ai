import gymnasium as gym
import numba
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.optim import Adam

from neu_ai.nn_utils import mlp, opt_step, tensor, to_np
from neu_ai.rl.m1_RLBase import RLBase


class PPOPi(nn.Module):
    log_std: tc.Tensor

    def __init__(s, sizes, Act, act_type):
        super().__init__()
        s.act_type = act_type
        s.net = mlp(sizes, Act)
        if act_type is float:
            log_std = nn.Parameter(-0.5 * tc.ones(sizes[-1]))
            s.register_parameter("log_std", log_std)

    def forward(s, s_t, a_t):
        if s.act_type is float:
            mu, std = s.net(s_t), tc.exp(s.log_std)
            dist = Normal(mu, std)
            a_t = dist.sample() if a_t is None else a_t
            logp = dist.log_prob(a_t).sum(-1)
        elif s.act_type is int:
            dist = Categorical(logits=s.net(s_t))
            a_t = dist.sample() if a_t is None else a_t
            logp = dist.log_prob(a_t)
        return a_t, logp


class PPO2017(RLBase):
    n_rand_step = 0
    D_size = 4000

    n_env_step = int(1e6)
    lr = 3e-4
    n_grad_step = 80
    gamma = 0.99
    GAE_lam = 0.95
    epsilon = 0.2
    KL_tar = 0.015
    hidden = [64, 64]
    Act = nn.ReLU

    def __init__(s, env: gym.Env, seed=0):
        d_obs, d_act, act_type = s.init(env, seed)
        s.pi = PPOPi([d_obs, *s.hidden, d_act], s.Act, act_type)
        s.pi_opt = Adam(s.pi.parameters(), s.lr)
        s.V = mlp([d_obs, *s.hidden, 1])
        s.V_opt = Adam(s.V.parameters(), s.lr)

    def get_act(s, s_t):
        return s.pi(s_t, None)[0]

    def run(s):
        for _ in range(int(s.n_env_step / s.D_size)):
            for _ in range(s.D_size):
                s.step_env()
            s.update_params()

    def get_V(s, s_t):
        return tc.squeeze(s.V(s_t), -1)

    def update_params(s):
        # - Compute advantage estimates $A_1, ..., A_T$
        s_t, a_t, r_t, s_next, term, trunc = s.D
        s_t, a_t, s_next = tensor([s_t, a_t, s_next])
        with tc.no_grad():
            V_t, V_next = [to_np(s.get_V(x)) for x in (s_t, s_next)]
            A_t, R_t = ppo_A_R(r_t, V_t, V_next, term, trunc, s.gamma, s.GAE_lam)
            A_t, R_t = tensor([A_t, R_t])
            logp_old = s.pi(s_t, a_t)[1]

        for _ in range(s.n_grad_step):
            # must use old a_t!
            logp = s.pi(s_t, a_t)[1]
            KL = tc.mean(logp_old - logp)
            if KL > s.KL_tar:
                break
            ratio = tc.exp(logp - logp_old)
            r_clip = tc.clip(ratio, 1 - s.epsilon, 1 + s.epsilon)
            # the paper maximizes L_clip! so add a minus sign!
            pi_loss = -tc.mean(tc.min(ratio * A_t, r_clip * A_t))
            opt_step(s.pi_opt, pi_loss)

        for _ in range(s.n_grad_step):
            V_loss = F.mse_loss(s.get_V(s_t), R_t)
            opt_step(s.V_opt, V_loss)


@numba.njit
def ppo_A_R(r, V, V_next, term, trunc, gamma, GAE_lam):
    T = len(r)
    A = np.zeros(T, dtype=np.float32)
    R = np.zeros(T, dtype=np.float32)
    for t in range(T - 1, -1, -1):
        delta_t = r[t] + gamma * V_next[t] * (1 - term[t]) - V[t]
        if term[t]:
            A_next, R_next = 0, 0
        elif t == T - 1 or trunc[t]:
            A_next, R_next = 0, V_next[t]
        else:
            A_next, R_next = A[t + 1], R[t + 1]
        A[t] = delta_t + gamma * GAE_lam * A_next
        R[t] = r[t] + gamma * R_next
    A = (A - A.mean()) / (A.std() + 1e-8)
    return A, R


if __name__ == "__main__":
    ppo = PPO2017(env=gym.make("HalfCheetah-v5"))
    ppo.run()
