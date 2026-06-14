from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box
from torch.distributions import Normal
from torch.optim import Adam

from neu_ai.nn_utils import mlp, opt_step, tensor
from neu_ai.rl.RLBase import RLBase


class SACNormalPi(nn.Module):
    def __init__(s, sizes, Act=nn.ReLU):
        super().__init__()
        s.body = mlp(sizes[:-1], Act, end=[Act()])
        s.mu = nn.Linear(sizes[-2], sizes[-1])
        s.log_std = nn.Linear(sizes[-2], sizes[-1])

    def forward(s, s_t):
        x = s.body(s_t)
        mu = s.mu(x)
        std = tc.exp(tc.clip(s.log_std(x), -20, 2))
        dist = Normal(mu, std)
        a_t = dist.rsample()
        logp = dist.log_prob(a_t).sum(-1)
        a_t_new = tc.tanh(a_t)
        # NaN crash: logp_new = logp - tc.log(1 - a_t_new**2).sum(-1)
        logp_new = logp - tc.sum(2 * (np.log(2) - a_t - F.softplus(-2 * a_t)), dim=-1)
        return a_t_new, logp_new


class Q1Q2(nn.Module):
    def __init__(s, sizes):
        super().__init__()
        s.Q1 = mlp(sizes)
        s.Q2 = mlp(sizes)

    def forward(s, s_t, a_t):
        x = tc.cat((s_t, a_t), dim=-1)
        return tc.squeeze(s.Q1(x), -1), tc.squeeze(s.Q2(x), -1)


class SAC2018(RLBase):
    n_rand_step = int(1e4)
    D_size = int(1e6)

    n_iteration = int(1e6)
    n_env_step = 1
    n_grad_step = 1
    mini_batch_size = 100
    gamma = 0.99
    lr = 1e-3
    tau = 0.005
    hidden = [64, 64]

    def __init__(s, env: gym.Env, seed=0):
        sp: Box = env.action_space
        assert np.all(sp.low == -1) and np.all(sp.high == 1)
        d_obs, d_act, act_type = s.init(env, seed)

        # - Input: $\theta_1, \theta_2, \phi$. (Initial parameters)
        s.Q1Q2 = Q1Q2([d_obs + d_act, *s.hidden, 1])
        s.Q_opt = Adam(s.Q1Q2.parameters(), s.lr)
        s.pi = SACNormalPi([d_obs, *s.hidden, d_act])
        s.pi_opt = Adam(s.pi.parameters(), s.lr)
        s.alpha = nn.Parameter(tc.tensor(0.2))
        s.alpha_opt = Adam([s.alpha], s.lr)
        s.H_tar = -d_act  # (e.g., -6 for HalfCheetah-v1)
        # - $\bar{\theta}_1 \gets \theta_1, \bar{\theta}_2 \gets \theta_2$  (Initialize target network weights)
        s.Q1Q2_tar = deepcopy(s.Q1Q2)
        s.Q1Q2_tar.requires_grad_(False)

    def get_act(s, s_t):
        return s.pi(s_t)[0]

    def run(s):
        for _ in range(s.n_iteration):
            for _ in range(s.n_env_step):
                s.step_env()
            for _ in range(s.n_grad_step):
                s.update_params()

    def update_params(s):
        if s.mini_batch_size > s.D_cnt:
            return
        data = s.sample(s.mini_batch_size)
        s_t, a_t, r_t, s_next, term, trunc = tensor(data)

        # - $\theta_i \gets \theta_i - \lambda_Q \nabla_{\theta_i} J_Q(\theta_i) \quad i \in \{1, 2\}$. (Update the Q-function parameters)
        with tc.no_grad():
            a_next, logp_next = s.pi(s_next)
            Q_tar_next = tc.min(*s.Q1Q2_tar(s_next, a_next))
            V_tar_next = Q_tar_next - s.alpha * logp_next
            Q_tar_t = r_t + s.gamma * V_tar_next * (1 - term)
        Q1_t, Q2_t = s.Q1Q2(s_t, a_t)
        Q_loss = F.mse_loss(Q1_t, Q_tar_t) + F.mse_loss(Q2_t, Q_tar_t)
        opt_step(s.Q_opt, Q_loss)

        # - $\phi \gets \phi - \lambda_\pi \nabla_\phi J_\pi(\phi)$. (Update policy weights)
        s.Q1Q2.requires_grad_(False)
        a_t_new, logp_new = s.pi(s_t)
        Q_t_new = tc.min(*s.Q1Q2(s_t, a_t_new))
        pi_loss = tc.mean(s.alpha * logp_new - Q_t_new)
        opt_step(s.pi_opt, pi_loss)
        s.Q1Q2.requires_grad_(True)

        # - $\alpha \gets \alpha - \lambda \nabla_\alpha J(\alpha)$. (Adjust temperature)
        H = -logp_new
        alpha_loss = s.alpha * tc.mean(H - s.H_tar).detach()
        opt_step(s.alpha_opt, alpha_loss)
        if s.D_cnt % 10000 == 0:
            print(f"alpha (temperature): {s.alpha.item()}")

        # - $\bar{\theta}_i \gets \tau \theta_i + (1 - \tau) \bar{\theta}_i \quad i \in \{1, 2\}$. (Update target network weights)
        with tc.no_grad():
            for p, p_tar in zip(s.Q1Q2.parameters(), s.Q1Q2_tar.parameters()):
                p_tar.copy_(s.tau * p + (1 - s.tau) * p_tar)


if __name__ == "__main__":
    sac = SAC2018(env=gym.make("HalfCheetah-v5"))
    sac.run()
