import gymnasium as gym
import numpy as np
import torch as tc
import torch.nn as nn
from gymnasium.spaces import Box, Discrete
from scipy.signal import lfilter
from torch.distributions import Categorical, Normal
from torch.optim import Adam

from neu_ai.nn_utils import mlp
from neu_ai.utils import Memory


def discount_cumsum(x, discount):
    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    log_std: tc.Tensor

    def __init__(s, net: nn.Module, Dist, d_act=None):
        super().__init__()
        s.net = net
        s.Dist = Dist
        if Dist is Normal:
            log_std = nn.Parameter(-0.5 * tc.ones(d_act))
            s.register_parameter("log_std", log_std)

    def forward(s, obs, act=None):
        pi = s._pi(obs)
        logp = None if act is None else s._logp(pi, act)
        return pi, logp

    def _pi(s, obs):
        if s.Dist is Categorical:
            return Categorical(logits=s.net(obs))
        elif s.Dist is Normal:
            return Normal(s.net(obs), s.log_std.exp())

    def _logp(s, pi, act):
        if isinstance(pi, Categorical):
            return pi.log_prob(act)
        elif isinstance(pi, Normal):
            return pi.log_prob(act).sum(-1)


class ActorCritic(nn.Module):
    def __init__(s, obs_space, act_space, hidden=(64, 64), Act=nn.Tanh):
        super().__init__()
        d_obs = obs_space.shape[0]
        if isinstance(act_space, Box):
            d_act = act_space.shape[0]
            net = mlp([d_obs, *hidden, d_act], Act)
            s.pi = Actor(net, Normal, d_act)
        elif isinstance(act_space, Discrete):
            net = mlp([d_obs, *hidden, act_space.n], Act)
            s.pi = Actor(net, Categorical)
        s.v = mlp([d_obs, *hidden, 1], Act)

    @tc.no_grad()
    def step(s, obs):
        pi = s.pi._pi(obs)
        a = pi.sample()
        logp = s.pi._logp(pi, a)
        v = s.v(obs).squeeze(-1)
        return a.numpy(), v.numpy(), logp.numpy()

    def act(s, obs):
        return s.step(obs)[0]


class PPOBuffer(Memory):
    def __init__(s, size, gamma=0.99, lam=0.95):
        super().__init__(size)
        s.gamma, s.lam = gamma, lam
        s.ptr, s.path_start = 0, 0
        s.adv = np.zeros(size, dtype=np.float32)
        s.ret = np.zeros(size, dtype=np.float32)

    def store(s, obs, act, rew, val, logp):
        s.save([obs, act, rew, val, logp])
        s.ptr += 1

    def finish_path(s, last_V=0):
        slc = slice(s.path_start, s.ptr)
        obs, act, rew, val, logp = s.mem
        rews = np.append(rew[slc], last_V)
        vals = np.append(val[slc], last_V)
        td_err = rews[:-1] + s.gamma * vals[1:] - vals[:-1]
        s.adv[slc] = discount_cumsum(td_err, s.gamma * s.lam)
        s.ret[slc] = discount_cumsum(rews, s.gamma)[:-1]
        s.path_start = s.ptr

    def get(s):
        assert s.ptr == s.size
        s.ptr, s.path_start = 0, 0
        s.adv = (s.adv - np.mean(s.adv)) / np.std(s.adv)
        obs, act, rew, val, logp = s.mem
        data = dict(obs=obs, act=act, ret=s.ret, adv=s.adv, logp=logp)
        return {
            k: tc.as_tensor(v, dtype=tc.float32).squeeze(-1) for k, v in data.items()
        }


def ppo(
    env=gym.make("HalfCheetah-v5"),
    seed=10000,
    buf_size=4000,
    epochs=250,
    gamma=0.99,
    clip_ratio=0.2,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_pi_iters=80,
    train_v_iters=80,
    lam=0.97,
    max_ep_len=1000,
    target_kl=0.01,
):
    tc.manual_seed(seed)
    np.random.seed(seed)

    ac = ActorCritic(env.observation_space, env.action_space, hidden=[64, 64])
    pi_opt = Adam(ac.pi.parameters(), lr=pi_lr)
    v_opt = Adam(ac.v.parameters(), lr=vf_lr)

    buf = PPOBuffer(buf_size, gamma, lam)

    def update():
        d = buf.get()
        obs, act, logp_old = d["obs"], d["act"], d["logp"]
        adv, ret = d["adv"], d["ret"]

        for i in range(train_pi_iters):
            pi, logp = ac.pi(obs, act)
            kl = (logp_old - logp).mean().item()
            if kl > 1.5 * target_kl:
                break
            ratio = tc.exp(logp - logp_old)
            clip_r = tc.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            loss_pi = -(tc.min(ratio * adv, clip_r * adv)).mean()
            pi_opt.zero_grad()
            loss_pi.backward()
            pi_opt.step()

        for i in range(train_v_iters):
            loss_v = ((ac.v(obs).squeeze(-1) - ret) ** 2).mean()
            v_opt.zero_grad()
            loss_v.backward()
            v_opt.step()

        # r1 = (loss_pi.item(), loss_v.item())
        # r2 = (-0.056847117841243744, 1171.0224609375)
        # assert 0, (r1 == r2, r1, r2)

    o, ep_ret, ep_len = env.reset(seed=seed)[0], 0, 0
    for _ in range(epochs):
        for t in range(buf_size):
            a, v, logp = ac.step(tc.as_tensor(o, dtype=tc.float32))

            next_o, r, term, trunc, _ = env.step(a)
            d = term or trunc
            ep_ret += r
            ep_len += 1
            buf.store(o, a, r, v, logp)
            o = next_o
            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == buf_size - 1

            if terminal or epoch_ended:
                if timeout or epoch_ended:
                    _, v, _ = ac.step(tc.as_tensor(o, dtype=tc.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    print(ep_ret)
                o, ep_ret, ep_len = env.reset()[0], 0, 0
        update()


if __name__ == "__main__":
    ppo()
