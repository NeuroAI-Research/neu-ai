from typing import List

import gymnasium as gym
import numpy as np
import torch as tc
from gymnasium.spaces import Box, Discrete


class RLBase:
    env: gym.Env
    on_policy: bool
    mem_size: int
    gam = 0.99

    def set_env(s, env, seed):
        s.env = env
        tc.manual_seed(seed)
        np.random.seed(seed)
        s.env.reset(seed=seed)
        s.env.action_space.seed(seed)

        d_obs = s.env.observation_space.shape[0]
        act_sp = s.env.action_space
        if isinstance(act_sp, Box):
            d_act, act_type = act_sp.shape[0], float
        elif isinstance(act_sp, Discrete):
            d_act, act_type = act_sp.n, int
        return d_obs, d_act, act_type

    def run(
        s,
        n_step=1e6,
        n_warmup_step=1e4,
    ):
        s.memory: List[np.ndarray] = []
        obs, _ = s.env.reset()
        eps_ret = 0
        for i in range(int(n_step)):
            if not s.on_policy and i < n_warmup_step:
                act = s.env.action_space.sample()
            else:
                with tc.no_grad():
                    obs_t = tc.from_numpy(obs).float()
                    act = s.get_act(obs_t).numpy()
            next_obs, rew, term, trunc, _ = s.env.step(act)
            s.memorize(i, [obs, act, next_obs, rew, term, trunc])
            eps_ret += rew
            if term or trunc:
                obs, _ = s.env.reset()
                print(f"step: {i + 1}, eps_ret: {eps_ret}")
                eps_ret = 0
            else:
                obs = next_obs

            if s.on_policy:
                if (i + 1) % s.mem_size == 0:
                    s.learn_from_memory()  # brand new memory
            else:
                if i >= n_warmup_step:
                    s.learn_from_memory()

    def memorize(s, i, values):
        if not len(s.memory):
            for v in values:
                shape = (s.mem_size, *np.shape(v))
                s.memory.append(np.zeros(shape, dtype=np.float32))
        i = i % s.mem_size
        for k, v in enumerate(values):
            s.memory[k][i] = v
        s.step_idx = i

    def sample_mem(s, size):
        high = min(s.step_idx + 1, s.mem_size)
        indices = np.random.randint(0, high, size)
        return [v[indices] for v in s.memory]

    def get_act(s, obs: tc.Tensor) -> tc.Tensor:
        raise NotImplementedError

    def learn_from_memory(s):
        raise NotImplementedError
