"""
- entropy target: -dim(A) (e.g., -6 for HalfCheetah-v1)
"""

from copy import deepcopy
from typing import List

import gymnasium as gym
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

from neu_ai.nn_utils import mlp


def tensor(x):
    if isinstance(x, np.ndarray):
        return tc.from_numpy(x).float()
    if isinstance(x, (list, tuple)):
        return [tensor(v) for v in x]


def to_np(x: tc.Tensor):
    return x.detach().numpy()


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


class SAC:
    n_iteration = int(1e6)
    n_env_step = 1
    n_grad_step = 1
    n_rand_step = int(1e4)
    D_size = int(1e6)
    mini_batch_size = 64
    gamma = 0.99
    lr = 3e-4
    tau = 0.005

    def __init__(s, env: gym.Env, seed=0):
        tc.manual_seed(seed)
        np.random.seed(seed)
        env.reset(seed=seed)
        env.action_space.seed(seed)

        s.env = env
        d_obs = env.observation_space.shape[0]
        d_act = env.action_space.shape[0]

        # - Input: $\theta_1, \theta_2, \phi$. (Initial parameters)
        s.Q1Q2 = Q1Q2([d_obs + d_act, 64, 64, 1])
        s.Q_opt = Adam(s.Q1Q2.parameters(), s.lr)
        s.pi = SACNormalPi([d_obs, 64, 64, d_act])
        s.pi_opt = Adam(s.pi.parameters(), s.lr)
        s.alpha = nn.Parameter(tc.tensor(0.2))
        # - $\bar{\theta}_1 \gets \theta_1, \bar{\theta}_2 \gets \theta_2$  (Initialize target network weights)
        s.Q1Q2_tar = deepcopy(s.Q1Q2)
        s.Q1Q2_tar.requires_grad_(False)

        # - $D \gets \emptyset $. (Initialize an empty replay pool)
        s.D: List[np.ndarray] = []
        s.D_cnt = 0

        s.s_t = s.env.reset()[0]
        s.eps_ret, s.eps_len = 0, 0

    def run(s):
        for _ in range(s.n_iteration):
            for _ in range(s.n_env_step):
                s.step_env()
            for _ in range(s.n_grad_step):
                s.update_params()

    def step_env(s):
        # - $a_t \sim \pi_\phi(a_t|s_t)$. (Sample action from the policy)
        if s.D_cnt < s.n_rand_step:
            a_t = s.env.action_space.sample()
        else:
            with tc.no_grad():
                a_t = to_np(s.pi(tensor(s.s_t))[0])
        # - $s_{t+1} \sim p(s_{t+1}|s_t, a_t)$. (Sample transition from the environment)
        s_next, r_t, term, trunc, _ = s.env.step(a_t)
        # - $D \gets D \cup \{(s_t, a_t, r(s_t, a_t), s_{t+1})\}$. (Store the transition in the replay pool)
        s.store([s.s_t, a_t, r_t, s_next, term, trunc])
        s.eps_ret += r_t
        s.eps_len += 1
        if term or trunc:
            s.s_t = s.env.reset()[0]
            print(f"step: {s.D_cnt}, episode len: {s.eps_len}, return: {s.eps_ret}")
            s.eps_ret, s.eps_len = 0, 0
        else:
            s.s_t = s_next

    def store(s, values: List[np.ndarray]):
        if not len(s.D):
            for v in values:
                shape = (s.D_size, *np.shape(v))
                s.D.append(np.zeros(shape, dtype=np.float32))
            s.D_cnt = 0
        ptr = s.D_cnt % s.D_size
        for k, v in enumerate(values):
            s.D[k][ptr] = v
        s.D_cnt += 1

    def sample(s, size):
        if size > s.D_cnt:
            raise Exception(f"requested samples: {size}, available samples: {s.D_cnt}")
        high = min(s.D_cnt, s.D_size)
        indices = np.random.choice(high, size, replace=False)
        return [v[indices] for v in s.D]

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
        s.Q_opt.zero_grad()
        Q_loss.backward()
        s.Q_opt.step()

        # - $\phi \gets \phi - \lambda_\pi \nabla_\phi J_\pi(\phi)$. (Update policy weights)
        s.Q1Q2.requires_grad_(False)
        a_t_new, logp_new = s.pi(s_t)
        Q_t_new = tc.min(*s.Q1Q2(s_t, a_t_new))
        pi_loss = tc.mean(s.alpha * logp_new - Q_t_new)
        s.pi_opt.zero_grad()
        pi_loss.backward()
        s.pi_opt.step()
        s.Q1Q2.requires_grad_(True)

        # - $\alpha \gets \alpha - \lambda \nabla_\alpha J(\alpha)$. (Adjust temperature)
        pass

        # - $\bar{\theta}_i \gets \tau \theta_i + (1 - \tau) \bar{\theta}_i \quad i \in \{1, 2\}$. (Update target network weights)
        with tc.no_grad():
            for p, p_tar in zip(s.Q1Q2.parameters(), s.Q1Q2_tar.parameters()):
                p_tar.copy_(s.tau * p + (1 - s.tau) * p_tar)


if __name__ == "__main__":
    sac = SAC(env=gym.make("HalfCheetah-v5"))
    sac.run()
