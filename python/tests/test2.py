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
    BCE_logits,
    TwoHot,
    cat,
    onehot_softmax,
    opt_step,
    stack_rows,
    symlog,
    tensor,
)
from neu_ai.plot import plot1
from neu_ai.rl.RLBase import RLBase
from neu_ai.utils import shape


def RSSM_KL(logits_P, logits_Q, min=1):
    P = Categorical(logits=logits_P)
    Q = Categorical(logits=logits_Q)
    kl = kl_divergence(P, Q)
    return tc.clip(kl.sum(-1), min=min).mean()


class DreamerV3(nn.Module, RLBase):
    n_rand_step = int(1e8)
    D_size = 1000

    T = 20
    n_opt_step = 1000
    loss_w = dict(z_enc=0.1, zp_dyn=1, rp=1, cp=1, xp=1, vp=1, pi=1, H=3e-4)
    gamma = 0.99
    lam = 0.95
    S_A = 1

    def __init__(s, env, seed=0):
        super().__init__()
        X, A, s.act_type = s.init(env, seed)
        s.z_nc = 16
        s.H = H = 64
        Z = s.z_nc**2
        s.twohot = TwoHot(64)

        s.rnn_zah_2_h = nn.GRUCell(Z + A, H)
        s.enc_hx_2_z = nn.Linear(H + X, Z)
        s.dyn_h_2_zp = nn.Linear(H, Z)
        s.rew_hz_2_rp = nn.Linear(H + Z, len(s.twohot.bins))
        s.con_hz_2_cp = nn.Linear(H + Z, 1)
        s.dec_hz_2_xp = nn.Linear(H + Z, X)

        s.cri_hz_2_vp = nn.Linear(H + Z, len(s.twohot.bins))
        a_out = A if s.act_type is int else A * 2
        s.act_hz_2_a = nn.Linear(H + Z, a_out)

        s.twohot.init(s.rew_hz_2_rp)
        s.twohot.init(s.cri_hz_2_vp)
        s.wm_opt = Adam(s.parameters())
        s.ac_opt = Adam([*s.cri_hz_2_vp.parameters(), *s.act_hz_2_a.parameters()])

    def learn(s):
        for i in range(s.D_size):
            s.step_env()
        B = int(s.D_size / s.T)
        D = [tensor(v).view(B, s.T, *v.shape[1:]) for v in s.D]
        print(shape(D))
        rows = []
        for i in tqdm(range(s.n_opt_step)):
            wm_losses = s.wm_losses(D)
            opt_step(s.wm_opt, wm_losses, s.loss_w)

            hz, a, rp, cp = s.dream()
            ac_losses = s.ac_losses(hz, a, rp, cp)
            opt_step(s.ac_opt, ac_losses, s.loss_w)

            rows.append({**wm_losses, **ac_losses})
        data = {k: np.array([r[k].item() for r in rows]) for k in rows[0]}
        plot1(data, "temp")

    def get_z(s, logits):
        noisy = s.training
        return onehot_softmax(logits, noisy, s.z_nc)

    def wm_losses(s, D: List[tc.Tensor]):
        x_raw, a, r, x_raw_next, term, trunc = D
        x = symlog(x_raw)
        c = 1 - term.unsqueeze(-1)
        not_end = (1 - term) * (1 - trunc)
        B, T, _ = x.shape
        rows = []
        z_t = None
        for t in range(T):
            if t == 0:
                h_t = tc.zeros(B, s.H)
            else:
                h_t = s.rnn_zah_2_h(cat(z_t, a[:, t - 1]), h_t)
                h_t = h_t * not_end[:, t - 1, None]
            z_t, z_lg = s.get_z(s.enc_hx_2_z(cat(h_t, x[:, t])))
            rows.append([h_t, z_t, z_lg])
        h, z, z_lg = stack_rows(rows, dim=1)
        hz = cat(h, z)
        zp, zp_lg = s.get_z(s.dyn_h_2_zp(h))
        return dict(
            z_enc=RSSM_KL(z_lg, tc.detach(zp_lg)),
            zp_dyn=RSSM_KL(tc.detach(z_lg), zp_lg),
            rp=s.twohot.loss(s.rew_hz_2_rp(hz), r),
            cp=BCE_logits(s.con_hz_2_cp(hz), c),
            xp=F.mse_loss(s.dec_hz_2_xp(hz), x),
        )

    @tc.no_grad()
    def dream(s, B=50, T=20):
        x_raw = [s.env.reset()[0] for _ in range(B)]
        x_0 = symlog(tensor(np.array(x_raw)))
        rows = []
        a_t = None
        for t in range(T):
            if t == 0:
                h_t = tc.zeros(B, s.H)
                zp_t, zp_lg = s.get_z(s.enc_hx_2_z(cat(h_t, x_0)))
            else:
                h_t = s.rnn_zah_2_h(cat(zp_t, a_t), h_t)
                zp_t, zp_lg = s.get_z(s.dyn_h_2_zp(h_t))
            a_t = s.get_pi(cat(h_t, zp_t)).sample()
            rows.append([h_t, zp_t, a_t])
        h, zp, a = stack_rows(rows, dim=1)
        hz = cat(h, zp)
        rp = s.twohot.decode_logits(s.rew_hz_2_rp(hz))
        cp = tc.sigmoid(s.con_hz_2_cp(hz)).squeeze(-1)
        return hz, a, rp, cp

    def ac_losses(s, hz, a, rp, cp):
        # critic:
        vp_lg = s.cri_hz_2_vp(hz)
        vp = s.twohot.decode_logits(vp_lg)
        Rp = s.get_R(rp, cp, vp)
        # actor:
        A = tc.detach((Rp - vp) / max(1, s.S_A))
        pi = s.get_pi(hz)
        logp, H = pi.log_prob(a), pi.entropy()
        if s.act_type is float:
            logp, H = logp.sum(-1), H.sum(-1)
        return dict(vp=s.twohot.loss(vp_lg, Rp), pi=-(A * logp).mean(), H=-H.mean())

    @tc.no_grad()
    def get_R(s, rp: tc.Tensor, cp, vp):
        B, T = rp.shape
        R = [None] * T
        R[T - 1] = vp[:, T - 1]
        for t in range(T - 2, -1, -1):
            mix = (1 - s.lam) * vp[:, t] + s.lam * R[t + 1]
            R[t] = rp[:, t] + s.gamma * cp[:, t] * mix
        return tc.stack(R, dim=1)

    def get_pi(s, hz):
        a_lg = s.act_hz_2_a(hz)
        if s.act_type is float:
            mu, std = tc.chunk(a_lg, 2, dim=-1)
            std = F.softplus(std) + 1e-4
            return Normal(mu, std)
        return Categorical(logits=a_lg)


if __name__ == "__main__":
    wm = DreamerV3(env=gym.make("HalfCheetah-v5"))
    wm.learn()
