import matplotlib.pyplot as plt
import torch as tc
import torch.nn as nn

from neu_ai.nn_utils import cat, one_hot_softmax, stack_rows, to_np
from neu_ai.utils import shape


class RSSMWorldModel(nn.Module):
    z_n_classes = 32
    dims_XA_ZH = [16, 4, 32 * 32, 64]

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


def main():
    wm = RSSMWorldModel()
    B, T = 1, 1
    X, A, Z, H = wm.dims_XA_ZH
    x = tc.rand(B, T, X)
    a = tc.rand(B, T, A)
    data = wm(x, a)
    print(shape(data))
    plt.imshow(to_np(data[0][0, 0]))
    plt.savefig("temp")


if __name__ == "__main__":
    main()
