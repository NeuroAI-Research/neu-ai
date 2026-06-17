from typing import List

import gymnasium as gym
import numpy as np
import torch as tc
import torch.nn.functional as F
from torch.optim import Adam

from neu_ai.nn_utils import opt_step, tensor, to_np
from neu_ai.plot import plot1
from neu_ai.rl.DreamerV3_2023 import (
    DreamerV3Actor,
    DreamerV3Critic,
    RSSMWorldModel,
    dreamerV3_R,
)
from neu_ai.rl.RLBase import RLBase
from neu_ai.utils import shape


class DreamerV3_2023(RLBase):
    n_rand_step = 10
    D_size = 10

    loss_w = dict(xp=1, rp=1, cp=1, dyn=1, rep=0.1, vp=1, vp_mse=0, pi=1, H=3e-4)
    gamma = 0.99
    lam = 0.95

    def __init__(s, env, seed=0):
        d_obs, d_act, act_type = s.init(env, seed)
        dims_XA_ZH = [d_obs, d_act, 16 * 16, 64]
        d_s = sum(dims_XA_ZH[-2:])

        s.wm = RSSMWorldModel(dims_XA_ZH)
        s.v = DreamerV3Critic(d_s)
        s.pi = DreamerV3Actor([d_s, 64, 64, d_act], act_type)
        s.opt = Adam([*s.wm.parameters(), *s.v.parameters(), *s.pi.parameters()])

    def test(s):
        for i in range(s.D_size):
            s.step_env()
        D = tensor([v[None, ...] for v in s.D])
        x_raw, a, r, x_raw_next, term, trunc = D
        res: List[tc.Tensor] = s.wm(x_raw, a)
        z_lg, zp_lg, hz = res
        v = s.v(hz)
        R_tar = dreamerV3_R(r, term, v, s.gamma, s.lam)
        s.pi.update_S(R_tar)
        print(shape(D))
        print(shape(res))
        hz = hz.detach()

        rows = []
        for i in range(1000):
            losses = s.wm.loss(x_raw, a, r, term)
            losses["vp"] = s.v.loss(hz, R_tar)
            losses["vp_mse"] = F.mse_loss(s.v(hz), R_tar)
            losses.update(s.pi.loss(hz, a, R_tar, v))
            opt_step(s.opt, losses, s.loss_w)
            rows.append(losses)
        data = {k: np.array([r[k].item() for r in rows]) for k in rows[0]}
        data["z.img"] = to_np(hz)[0, 0, 64:].reshape((16, 16))
        plot1(data, "temp")


if __name__ == "__main__":
    dreamer = DreamerV3_2023(env=gym.make("HalfCheetah-v5"))
    dreamer.test()
