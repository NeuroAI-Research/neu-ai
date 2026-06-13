import numpy as np
import torch as tc
import torch.nn as nn


def mlp(sizes, Act=nn.ReLU, end=[]):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), Act()]
    return nn.Sequential(*(layers[:-1] + end))


def to_np(x: tc.Tensor):
    return x.detach().numpy()


def tensor(x):
    if isinstance(x, np.ndarray):
        return tc.from_numpy(x).float()
    if isinstance(x, (list, tuple)):
        return [tensor(v) for v in x]


def opt_step(opt: tc.optim.Adam, loss: tc.Tensor):
    opt.zero_grad()
    loss.backward()
    opt.step()
