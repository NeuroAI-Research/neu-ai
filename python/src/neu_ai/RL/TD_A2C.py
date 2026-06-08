import gymnasium as gym
import torch as tc
import torch.nn as nn
from torch.distributions import Categorical

from neu_ai.nn_utils import mlp
from neu_ai.utils import Memory


class RLBase:
    env: gym.Env

    def run(s, n_episode=200, f_log=10, mem_size=2000):
        s.memory = Memory(mem_size)

        for e in range(n_episode):
            obs, _ = s.env.reset()
            eps_rew = 0
            done = False
            while not done:
                act = s.sample_act(obs)
                obs2, rew, term, trunc, _ = s.env.step(act)
                s.memory.save([obs, act, obs2, rew, term])
                s.step()

                obs = obs2
                eps_rew += rew
                done = term or trunc
            if e % f_log == 0:
                print(f"episode: {e}, eps_rew: {eps_rew}")
        s.env.close()

    def sample_act(s, obs):
        return s.env.action_space.sample()

    def step(s):
        print(s.memory.cnt)


# ==================================


class TD_A2C(RLBase):
    batch_size = 64
    gamma = 0.99

    def __init__(s, env: gym.Env, hidden=[64, 64], lr=1e-3):
        s.env = env
        d_obs = env.observation_space.shape[0]
        d_act = env.action_space.n

        # BasalGanglia.Striatum.Striosome 基底核.纹状体.小体:
        s.actor = mlp([d_obs, *hidden, d_act])
        # BasalGanglia.Striatum.Matrix 基底核.纹状体.基质:
        s.critic = mlp([d_obs, *hidden, 1])

        s.nets = nn.ModuleList([s.actor, s.critic])
        s.opt = tc.optim.Adam(s.nets.parameters(), lr)

    @tc.no_grad()
    def sample_act(s, obs):
        obs = tc.from_numpy(obs)
        dist = Categorical(logits=s.actor(obs))
        return dist.sample().item()

    def step(s):
        # Hippocampus 海马体:
        mem = s.memory.sample(s.batch_size)
        if mem is not None:
            obs, act, obs2, rew, term = map(tc.from_numpy, mem)

            v, v2 = s.critic(obs), s.critic(obs2)
            dist = Categorical(logits=s.actor(obs))
            log_p: tc.Tensor = dist.log_prob(act[:, 0])[:, None]

            # Midbrain.VTA.DopaminergicNeurons 中脑.腹侧被盖区.多巴胺神经元:
            td_err: tc.Tensor = rew + s.gamma * v2 * (1 - term) - v

            critic_loss = td_err.pow(2).mean()
            actor_loss = -(log_p * td_err.detach()).mean()
            loss = critic_loss + actor_loss

            s.opt.zero_grad()
            loss.backward()
            s.opt.step()
