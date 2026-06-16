from typing import List

import matplotlib.pyplot as plt
import numba
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence

from neu_ai.nn_utils import (
    cat,
    cross_entropy,
    mlp,
    one_hot_softmax,
    opt_step,
    stack_rows,
    tensor,
    to_np,
)
from neu_ai.utils import exp_bins, shape, two_hot_decode, two_hot_encode


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


# =======================================


class DreamerV3Critic(nn.Module):
    def __init__(s, d_s):
        super().__init__()
        s.R_bins_np = exp_bins(-10, 10, 128)
        s.R_bins = tensor(s.R_bins_np)

        s.net = mlp([d_s, 64, 64, len(s.R_bins)])
        nn.init.zeros_(s.net[-1].weight)
        nn.init.zeros_(s.net[-1].bias)

    def forward(s, s_t):
        # v_t := E[v_\psi(R_t|s_t)]
        logits = s.net(s_t)
        probs = F.softmax(logits, dim=-1)
        return tc.sum(probs * s.R_bins, dim=-1)

    def loss(s, s_t: tc.Tensor, R_tar: np.ndarray, type="CE"):
        if type == "CE":
            logits = s.net(s_t)
            probs_tar = tensor(two_hot_encode(R_tar, s.R_bins_np))
            return cross_entropy(logits, probs_tar)
        elif type == "MSE":
            R_pred = s(s_t)
            return F.mse_loss(R_pred, tensor(R_tar))


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


def test_DreamerV3Critic():
    bins = exp_bins(-10, 10, 128)
    x = np.array([-666, 888])
    probs = two_hot_encode(x, bins)
    print(two_hot_decode(probs, bins))

    d_s = 10
    s_t = tc.rand((1, 1, d_s))
    R_tar = np.ones((1, 1)) * 1e3
    critic = DreamerV3Critic(d_s)
    opt = tc.optim.Adam(critic.parameters())
    for i in range(1000):
        loss_CE = critic.loss(s_t, R_tar, "CE")
        loss_MSE = critic.loss(s_t, R_tar, "MSE")
        opt_step(opt, loss_CE)
        if i % 100 == 0:
            print(f"step: {i}, CE: {loss_CE.item()}, MSE: {loss_MSE.item()}")


if __name__ == "__main__":
    test_DreamerV3Critic()
