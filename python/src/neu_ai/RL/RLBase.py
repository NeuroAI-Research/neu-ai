from typing import Dict, List

import gymnasium as gym
import numpy as np
import torch as tc
from gymnasium.spaces import Box, Discrete

from neu_ai.nn_utils import tensor, to_np
from neu_ai.plot import plot1


class RLBase:
    n_rand_step: int
    D_size: int

    def init(s, env: gym.Env, seed):
        s.env = env
        tc.manual_seed(seed)
        np.random.seed(seed)
        env.action_space.seed(seed)

        s.D: List[np.ndarray] = []
        s.D_cnt = 0
        s.records = []

        s.s_t = s.env.reset(seed=seed)[0]
        s.reset_agent()
        s.eps_ret, s.eps_len = 0, 0

        d_obs = env.observation_space.shape[0]
        sp = env.action_space
        if isinstance(sp, Box):
            d_act, act_type = sp.shape[0], float
        elif isinstance(sp, Discrete):
            d_act, act_type = sp.n, int
        return d_obs, d_act, act_type

    def reset_agent(s):
        pass

    def get_act(s, s_t: tc.Tensor) -> tc.Tensor:
        raise NotImplementedError

    def step_env(s):
        if s.D_cnt < s.n_rand_step:
            a_t = s.env.action_space.sample()
        else:
            with tc.no_grad():
                a_t = to_np(s.get_act(tensor(s.s_t)))
        s_next, r_t, term, trunc, _ = s.env.step(a_t)
        s.store([s.s_t, a_t, r_t, s_next, term, trunc])
        s.eps_ret += r_t
        s.eps_len += 1
        if term or trunc:
            s.s_t = s.env.reset()[0]
            s.reset_agent()
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

    def sample_BT(s, B, T):
        if B + T > s.D_cnt:
            raise Exception(f"requested (B, T): {(B, T)}, available: {s.D_cnt}")
        high = min(s.D_cnt, s.D_size) - T
        starts = np.random.choice(high, B, replace=False)
        return [np.array([v[i : i + T] for i in starts]) for v in s.D]

    def record(s, dic: Dict[str, tc.Tensor]):
        s.records.append({k: v.item() for k, v in dic.items()})

    def plot_records(s, id):
        data = {k: np.array([r[k] for r in s.records]) for k in s.records[0]}
        plot1(data, id)
