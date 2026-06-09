import itertools
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam

from neu_ai.nn_utils import mlp


class SquashedNormalActor(nn.Module):
    def __init__(s, d_obs, d_act, hidden, Act, act_lim):
        super().__init__()
        s.net = mlp([d_obs, *hidden], Act, [Act()])
        s.mu = nn.Linear(hidden[-1], d_act)
        s.log_std = nn.Linear(hidden[-1], d_act)
        s.act_lim = act_lim

    def forward(s, obs, deterministic=False, with_logp=True):
        x = s.net(obs)
        mu = s.mu(x)
        std = tc.clamp(s.log_std(x), -20, 2).exp()
        pi = Normal(mu, std)
        act = mu if deterministic else pi.rsample()
        if with_logp:
            logp = pi.log_prob(act).sum(-1)
            logp -= (2 * (np.log(2) - act - F.softplus(-2 * act))).sum(1)
        else:
            logp = None
        act = s.act_lim * tc.tanh(act)
        return act, logp


class QCritic(nn.Module):
    def __init__(s, d_obs, d_act, hidden, Act):
        super().__init__()
        s.q = mlp([d_obs + d_act, *hidden, 1], Act)

    def forward(s, obs, act):
        q = s.q(tc.cat([obs, act], dim=-1))
        return tc.squeeze(q, -1)


class ActorCritic(nn.Module):
    def __init__(s, obs_sp, act_sp, hidden=(64, 64), Act=nn.ReLU):
        super().__init__()
        d_obs = obs_sp.shape[0]
        d_act = act_sp.shape[0]
        act_lim = act_sp.high[0]
        s.pi = SquashedNormalActor(d_obs, d_act, hidden, Act, act_lim)
        s.q1 = QCritic(d_obs, d_act, hidden, Act)
        s.q2 = QCritic(d_obs, d_act, hidden, Act)

    @tc.no_grad()
    def act(s, obs, deterministic=False):
        obs = tc.from_numpy(obs).float()
        act, logp = s.pi(obs, deterministic, False)
        return act.numpy()

    def q1q2(s, o, a):
        return s.q1(o, a), s.q2(o, a)


class ReplayBuffer:
    def __init__(s, size):
        s.mem = []
        s.ptr, s.size, s.max_size = 0, 0, size

    def store(s, obs, act, rew, next_obs, done):
        values = [obs, act, rew, next_obs, done]
        if not len(s.mem):
            for v in values:
                shape = (s.max_size, *np.shape(v))
                s.mem.append(np.zeros(shape, dtype=np.float32))
        for k, v in enumerate(values):
            s.mem[k][s.ptr] = v
        s.ptr = (s.ptr + 1) % s.max_size
        s.size = min(s.size + 1, s.max_size)

    def sample(s, batch_size=32):
        idx = np.random.randint(0, s.size, batch_size)
        return [tc.from_numpy(v[idx]) for v in s.mem]


# ====================================


def sac(
    env_fn=lambda: gym.make("HalfCheetah-v5"),
    seed=0,
    steps_per_epoch=4000,
    epochs=100,
    replay_size=int(1e6),
    gamma=0.99,
    polyak=0.995,
    lr=1e-3,
    alpha=0.2,
    batch_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    max_ep_len=1000,
):
    env: gym.Env = env_fn()

    tc.manual_seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    ac = ActorCritic(env.observation_space, env.action_space)
    ac_targ = deepcopy(ac)
    # Freeze target networks (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    actor_opt = Adam(ac.pi.parameters(), lr)
    critic_opt = Adam(q_params, lr)

    buf = ReplayBuffer(replay_size)

    def update():
        # obs, act, rew, next_obs, done
        o, a, r, o2, d = buf.sample(batch_size)
        q1, q2 = ac.q1q2(o, a)
        with tc.no_grad():
            a2, logp2 = ac.pi(o2)
            q_targ = tc.min(*ac_targ.q1q2(o2, a2))
            backup = r + gamma * (1 - d) * (q_targ - alpha * logp2)
        loss_q = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        critic_opt.zero_grad()
        loss_q.backward()
        critic_opt.step()

        for p in q_params:
            p.requires_grad = False

        act, logp = ac.pi(o)
        q = tc.min(*ac.q1q2(o, act))
        loss_pi = (alpha * logp - q).mean()

        actor_opt.zero_grad()
        loss_pi.backward()
        actor_opt.step()

        for p in q_params:
            p.requires_grad = True

        with tc.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        # r1 = (2.037222385406494, -0.4769050180912018)
        # r2 = (loss_q.item(), loss_pi.item())
        # assert 0, (r1 == r2, r1, r2)

    o, ep_ret, ep_len = env.reset()[0], 0, 0
    for t in range(steps_per_epoch * epochs):
        a = ac.act(o) if t > start_steps else env.action_space.sample()
        o2, r, term, trunc, _ = env.step(a)
        d = term or trunc
        ep_ret += r
        ep_len += 1
        d = False if ep_len == max_ep_len else d
        buf.store(o, a, r, o2, d)
        o = o2
        if d or ep_len == max_ep_len:
            print(ep_ret)
            o, ep_ret, ep_len = env.reset()[0], 0, 0
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                update()


if __name__ == "__main__":
    sac()
