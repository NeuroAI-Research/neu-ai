from typing import List

import numba
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, kl_divergence

from neu_ai.nn_utils import (
    TwoHot,
    cat,
    mlp,
    onehot_softmax,
    stack_rows,
    symlog,
    tensor,
    to_np,
)


class RSSMWorldModel(nn.Module):
    def __init__(s, dims_XA_ZH: List[int]):
        super().__init__()
        s.dims_XA_ZH = dims_XA_ZH
        s.twohot = TwoHot(64)
        X, A, Z, H = s.dims_XA_ZH
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
        z_t = tc.zeros(B, Z)
        rows = []
        for t in range(T):
            x_t = x[:, t]
            a_prev = a[:, t - 1] if t > 0 else tc.zeros(B, A)
            h_t = s.rnn_zah_2_h(cat(z_t, a_prev), h_t)
            # dynamics predictor (prior)
            zp_t, zp_lg = s.get_z(s.dyn_h_2_zp(h_t))
            # encoder (posterior)
            z_t, z_lg = s.get_z(s.enc_hx_2_z(cat(h_t, x_t)))
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
    kl = kl_divergence(P, Q).sum(-1)
    return tc.clip(kl, min=min).mean()


# =======================================


class DreamerV3Critic(nn.Module):
    def __init__(s, d_s):
        super().__init__()
        s.twohot = TwoHot(64)
        s.net = mlp([d_s, 64, 64, len(s.twohot.bins)])
        s.twohot.init(s.net[-1])

    def forward(s, s_t):
        return s.twohot.decode_logits(s.net(s_t))

    def loss(s, s_t, R_tar):
        logits = s.net(s_t)
        return s.twohot.loss(logits, R_tar)


@tensor
@numba.njit
def dreamerV3_R(r: np.ndarray, term, v, gamma, lam):
    c = 1 - term
    B, T = r.shape
    R = np.zeros_like(r)
    for t in range(T - 1, -1, -1):
        if t == T - 1:
            R[:, t] = v[:, t]
        else:
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
