from typing import List

import matplotlib.pyplot as plt
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
    one_hot_softmax,
    opt_step,
    stack_rows,
    tensor,
    to_np,
)
from neu_ai.utils import shape


class RSSMWorldModel(nn.Module):
    z_n_classes = 32
    dims_XA_ZH = [16, 4, 32 * 32, 64]
    w_losses = [1, 1, 0.1]

    def __init__(s):
        super().__init__()
        X, A, Z, H = s.dims_XA_ZH
        s.rnn_zah_2_h = nn.GRUCell(Z + A, H)
        s.enc_hx_2_z = nn.Linear(H + X, Z)
        s.dyn_h_2_zp = nn.Linear(H, Z)
        s.rew_hz_2_rp = nn.Linear(H + Z, 1)
        s.con_hz_2_cp = nn.Linear(H + Z, 1)
        s.dec_hz_2_xp = nn.Linear(H + Z, X)

    def get_z(s, logits):
        noisy = s.training
        return one_hot_softmax(logits, noisy, s.z_n_classes)

    def forward(s, x: tc.Tensor, a):
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
            zp_t, zp_logits = s.get_z(s.dyn_h_2_zp(h_t))
            # encoder (posterior)
            z_t, z_logits = s.get_z(s.enc_hx_2_z(cat(h_t, x_t)))

            hz_t = cat(h_t, z_t)
            rp_t = s.rew_hz_2_rp(hz_t)
            cp_t = s.con_hz_2_cp(hz_t)
            xp_t = s.dec_hz_2_xp(hz_t)

            rows.append([z_logits, zp_logits, rp_t, cp_t, xp_t])
        return stack_rows(rows, dim=1)

    def loss(s, x, a, r, c):
        res: List[tc.Tensor] = s(x, a)
        z_logits, zp_logits, rp, cp, xp = res
        bce = F.binary_cross_entropy_with_logits

        def KL(logits_P, logits_Q):
            P = Categorical(logits=logits_P)
            Q = Categorical(logits=logits_Q)
            kl = kl_divergence(P, Q).sum(-1)
            return tc.clip(kl, min=1).mean()

        # todo: replace with symlog
        x_loss = F.mse_loss(xp, x, reduction="none").sum(-1)
        r_loss = F.mse_loss(rp, r, reduction="none").squeeze(-1)
        c_loss = bce(cp, c, reduction="none").squeeze(-1)
        pred_loss = (x_loss + r_loss + c_loss).mean()
        dyn_loss = KL(z_logits.detach(), zp_logits)
        rep_loss = KL(z_logits, zp_logits.detach())
        losses = [pred_loss, dyn_loss, rep_loss]
        loss: tc.Tensor = sum(w * l for w, l in zip(s.w_losses, losses))
        return loss, losses


# =======================================


class DreamerV3Critic(nn.Module):
    def __init__(s, d_s):
        super().__init__()
        s.twohot = TwoHot()

        s.net = mlp([d_s, 64, 64, len(s.twohot.bins)])
        nn.init.zeros_(s.net[-1].weight)
        nn.init.zeros_(s.net[-1].bias)

    def forward(s, s_t):
        # v_t := E[v_\psi(R_t|s_t)]
        logits = s.net(s_t)
        return s.twohot.decode(probs=F.softmax(logits, dim=-1))

    def loss(s, s_t, R_tar):
        logits = s.net(s_t)
        return s.twohot.loss(logits, R_tar)


@numba.njit
def dreamerV3_R(r: np.ndarray, c, v, gamma, lam):
    B, T, _ = r.shape
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
    eta = 3e-4
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
        pi_loss = -(A * logp).mean()
        H_loss = -s.eta * H.mean()
        return pi_loss, H_loss


# =======================================


def test_RSSMWorldModel():
    wm = RSSMWorldModel()
    opt = tc.optim.Adam(wm.parameters())
    B, T = 1, 1
    X, A, Z, H = wm.dims_XA_ZH
    x = tc.rand(B, T, X)
    a = tc.rand(B, T, A)
    r = tc.rand(B, T, 1)
    c = tc.ones(B, T, 1)
    data = wm(x, a)
    print(shape(data))
    plt.imshow(to_np(data[0][0, 0]))
    plt.savefig("temp")

    for _ in range(1000):
        loss, losses = wm.loss(x, a, r, c)
        print(loss.item(), [l.item() for l in losses])
        opt_step(opt, loss)


def test_DreamerV3Critic():
    twohot = TwoHot()
    print(twohot.decode(twohot.encode(tc.tensor([-666, 888]))))

    d_s = 10
    s_t = tc.rand((1, 1, d_s))
    R_tar = tc.ones((1, 1)) * 1e3
    critic = DreamerV3Critic(d_s)
    opt = tc.optim.Adam(critic.parameters())
    for i in range(1000):
        loss_CE = critic.loss(s_t, R_tar)
        loss_MSE = F.mse_loss(critic(s_t), R_tar)
        opt_step(opt, loss_CE)
        if i % 100 == 0:
            print(f"step: {i}, CE: {loss_CE.item()}, MSE: {loss_MSE.item()}")


def test_DreamerV3Actor():
    B, T, d_s, d_a = 2, 3, 10, 4
    s_t = tc.rand((B, T, d_s))
    R_t_np = np.random.normal(500, 150, (B, T))
    v_t_np = R_t_np - np.random.exponential(10, (B, T))
    R_t, v_t = tensor([R_t_np, v_t_np])

    for act_type in [float, int]:
        pi = DreamerV3Actor([d_s, 64, 64, d_a], act_type)
        opt = tc.optim.Adam(pi.parameters())
        pi.update_S(R_t)
        with tc.no_grad():
            a_t = pi.get_dist(s_t).sample()
        for i in range(100):
            losses = pi.loss(s_t, a_t, R_t, v_t)
            loss = sum(losses)
            opt_step(opt, loss)
            if i % 10 == 0:
                print(
                    f"{act_type} {i}, [pi, H] losses: {[v.item() for v in losses]}, S: {pi.S:.2f}"
                )


if __name__ == "__main__":
    test_DreamerV3Critic()
