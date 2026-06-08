import torch as tc
import torch.nn as nn


def mlp(sizes, Act=nn.ReLU, end=[]):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), Act()]
    return nn.Sequential(*(layers[:-1] + end))


def to_np(x: tc.Tensor):
    return x.detach().numpy()
