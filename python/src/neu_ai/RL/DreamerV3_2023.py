from typing import List

import gymnasium as gym
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, kl_divergence
from torch.optim import Adam
from tqdm import tqdm

from neu_ai.nn_utils import (
    TwoHot,
    cat,
    mlp,
    onehot_softmax,
    opt_step,
    stack_rows,
    symlog,
    tensor,
    to_np,
)
from neu_ai.plot import plot1
from neu_ai.rl.RLBase import RLBase
from neu_ai.utils import shape


class RSSMWorldModel(nn.Module):
    def __init__(s, dims_XA_ZH: List[int]):
        super().__init__()
        X, A, Z, H = s.dims_XA_ZH = dims_XA_ZH
        s.twohot = TwoHot(64)
        s.z_n_classes = int(np.sqrt(Z))
        assert s.z_n_classes**2 == Z

        s.rnn_zah_2_h = nn.GRUCell(Z + A, H)
        s.enc_hx_2_z = nn.Linear(H + X, Z)
        s.dyn_h_2_zp = nn.Linear(H, Z)
        s.rew_hz_2_rp = nn.Linear(H + Z, len(s.twohot.bins))
        s.con_hz_2_cp = nn.Linear(H + Z, 1)
        s.dec_hz_2_xp = nn.Linear(H + Z, X)

        s.twohot.init(s.rew_hz_2_rp)

    def get_z(s, logits):
        noisy = s.training
        return onehot_softmax(logits, noisy, s.z_n_classes)

    def forward(s, x_raw: tc.Tensor, a):
        x = symlog(x_raw)
        B, T, X = x.shape
        X, A, Z, H = s.dims_XA_ZH
        h_t = tc.zeros(B, H)
        z_t, _ = s.get_z(s.dyn_h_2_zp(h_t))
        rows = []
        for t in range(T):
            a_prev = a[:, t - 1] if t > 0 else tc.zeros(B, A)
            h_t = s.rnn_zah_2_h(cat(z_t, a_prev), h_t)
            # dynamics predictor (prior)
            zp_t, zp_lg = s.get_z(s.dyn_h_2_zp(h_t))
            # encoder (posterior)
            z_t, z_lg = s.get_z(s.enc_hx_2_z(cat(h_t, x[:, t])))
            hz_t = cat(h_t, z_t)
            rows.append([z_lg, zp_lg, hz_t])
        return stack_rows(rows, dim=1)

    def loss(s, x_raw, a, r, term):
        assert x_raw.shape[:-1] == a.shape[:-1] == r.shape == term.shape
        c = tc.unsqueeze(1 - term, -1)
        z_lg, zp_lg, hz = s(x_raw, a)
        rp_lg = s.rew_hz_2_rp(hz)
        cp_lg = s.con_hz_2_cp(hz)
        xp = s.dec_hz_2_xp(hz)
        return dict(
            rp=s.twohot.loss(rp_lg, r),
            cp=F.binary_cross_entropy_with_logits(cp_lg, c),
            xp=F.mse_loss(xp, symlog(x_raw)),
            dyn=RSSM_KL(tc.detach(z_lg), zp_lg),
            rep=RSSM_KL(z_lg, tc.detach(zp_lg)),
        )


def RSSM_KL(logits_P, logits_Q, min=1):
    P = Categorical(logits=logits_P)
    Q = Categorical(logits=logits_Q)
    kl = kl_divergence(P, Q)
    return tc.clip(kl.sum(-1), min=min).mean()


# =======================================


class DreamerV3Critic(nn.Module):
    def __init__(s, d_s):
        super().__init__()
        s.twohot = TwoHot(64)
        s.net = mlp([d_s, 64, 64, len(s.twohot.bins)])
        s.twohot.init(s.net[-1])

    def loss(s, s_t, r, term, gamma, lam):
        logits = s.net(s_t)
        with tc.no_grad():
            v = s.twohot.decode_logits(logits)
            R_tar = dreamerV3_R(r, term, v, gamma, lam)
        loss = s.twohot.loss(logits, R_tar)
        return loss, v, R_tar


def dreamerV3_R(r: tc.Tensor, term, v, gamma, lam):
    c = 1 - term
    B, T = r.shape
    R = tc.zeros_like(r)
    R[:, T - 1] = v[:, T - 1]
    for t in range(T - 2, -1, -1):
        mix = (1 - lam) * v[:, t] + lam * R[:, t + 1]
        R[:, t] = r[:, t] + gamma * c[:, t] * mix
    return R


# ===================================


class DreamerV3Actor(nn.Module):
    S = 1.0
    r_EMA = 0.99

    def __init__(s, sizes, act_type):
        super().__init__()
        s.act_type = act_type
        if act_type is float:
            sizes[-1] *= 2
        s.net = mlp(sizes)

    def get_dist(s, s_t: tc.Tensor):
        logits = s.net(s_t)
        if s.act_type is float:
            mu, std = tc.chunk(logits, 2, dim=-1)
            std = F.softplus(std) + 1e-4
            return Normal(mu, std)
        return Categorical(logits=logits)

    def update_S(s, R_t):
        R_t = to_np(R_t)
        delta_R = np.percentile(R_t, 95) - np.percentile(R_t, 5)
        s.S = s.r_EMA * s.S + (1 - s.r_EMA) * delta_R

    def loss(s, s_t, a_t, R_t, v_t):
        A = tc.detach((R_t - v_t) / max(1, s.S))
        dist = s.get_dist(s_t)
        logp = dist.log_prob(a_t)
        H = dist.entropy()
        if s.act_type is float:
            logp, H = logp.sum(-1), H.sum(-1)
        return dict(pi=-(A * logp).mean(), H=-H.mean())


# =================================


class DreamerV3_2023(RLBase):
    n_rand_step = int(1e6)
    D_size = 1000

    T = 20
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
        s.wm_opt = Adam(s.wm.parameters())
        s.v_opt = Adam(s.v.parameters())
        s.pi_opt = Adam(s.pi.parameters())

    def test(s):
        # 1. collect experience
        for i in range(s.D_size):
            s.step_env()
        B = int(s.D_size / s.T)
        D = [tensor(v).view(B, s.T, *v.shape[1:]) for v in s.D]
        print(shape(D))
        x_raw, a, r, x_raw_next, term, trunc = D

        # 2. train world model
        rows = []
        for i in tqdm(range(1000)):
            losses = s.wm.loss(x_raw, a, r, term)
            opt_step(s.wm_opt, losses, s.loss_w)
            rows.append(losses)
        data = {k: np.array([r[k].item() for r in rows]) for k in rows[0]}

        # 3. evaluate world model
        with tc.no_grad():
            z_lg, zp_lg, hz = s.wm(x_raw, a)
        print(shape([z_lg, zp_lg, hz]))
        data["z.img"] = to_np(hz)[0, 0, 64:].reshape((16, 16))
        plot1(data, "temp")

        # 4. train critic
        rows = []
        for i in tqdm(range(1000)):
            loss, v, R_tar = s.v.loss(hz, r, term, s.gamma, s.lam)
            opt_step(s.v_opt, loss)
            rows.append(loss.item())
        data["vp"] = np.array(rows)
        plot1(data, "temp")

        # 5. train actor
        rows = []
        for i in tqdm(range(1000)):
            losses = s.pi.loss(hz, a, R_tar, v)
            opt_step(s.pi_opt, losses, s.loss_w)
            rows.append(losses)
        data3 = {k: np.array([r[k].item() for r in rows]) for k in rows[0]}
        data.update(data3)
        plot1(data, "temp")


if __name__ == "__main__":
    dreamer = DreamerV3_2023(env=gym.make("HalfCheetah-v5"))
    dreamer.test()
