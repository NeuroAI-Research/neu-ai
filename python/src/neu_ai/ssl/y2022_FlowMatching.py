import torch as tc
import torch.nn as nn
import torch.nn.functional as F

from neu_ai.nn_utils import cat, mlp, opt_step, tensor, to_np


class FlowMatching2022:
    def __init__(s, net: nn.Module, lr=1e-2):
        s.net = net
        s.opt = tc.optim.Adam(net.parameters(), lr)

    def get_v(s, x_t, t, context):
        return s.net(cat(x_t, t, context))

    def opt_step(s, x1, context):
        x0 = tc.randn_like(x1)
        t = tc.rand(len(x1), 1)
        x_t = (1 - t) * x0 + t * x1
        v = s.get_v(x_t, t, context)
        loss = F.mse_loss(v, x1 - x0)
        opt_step(s.opt, loss)

    def int_step(s, x_t: tc.Tensor, t1: tc.Tensor, t2, context):
        t1 = t1.view(1, 1).expand(x_t.shape[0], 1)
        dt = t2 - t1
        t_mid = t1 + dt / 2
        v1 = s.get_v(x_t, t1, context)
        x_mid = x_t + v1 * dt / 2
        v_mid = s.get_v(x_mid, t_mid, context)
        return x_t + v_mid * dt

    @tc.no_grad()
    def integrate(s, x0, context, n_step=10):
        t = tc.linspace(0, 1, n_step + 1)
        for i in range(n_step):
            x0 = s.int_step(x0, t[i], t[i + 1], context)
        return x0


def test_FlowMatching2022():
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons

    d_x, d_c = 2, 1
    net = mlp([d_x + 1 + d_c, 64, 64, d_x])
    fm = FlowMatching2022(net)

    for _ in range(1000):
        x1, context = tensor(make_moons(256, noise=0.15))
        fm.opt_step(x1, context.view(-1, 1))

    x0 = tc.randn(300, 2)
    context = tc.randint(0, 2, (len(x0), 1)).float()
    colors = to_np(context.squeeze())
    x1 = to_np(fm.integrate(x0, context))
    plt.scatter(x1[:, 0], x1[:, 1], s=5, c=colors)
    plt.savefig("temp")


if __name__ == "__main__":
    test_FlowMatching2022()
