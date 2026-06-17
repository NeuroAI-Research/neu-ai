from typing import List

import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F


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


def cat(*args):
    return tc.cat(args, dim=-1)


def stack_rows(rows: List[List[tc.Tensor]], dim):
    return [tc.stack([r[i] for r in rows], dim) for i in range(len(rows[0]))]


def cross_entropy(logits: tc.Tensor, probs: tc.Tensor):
    n = logits.shape[-1]
    return F.cross_entropy(logits.view(-1, n), probs.view(-1, n))


def one_hot_softmax(logits: tc.Tensor, noisy: bool, n_classes: int = None):
    """returns one-hot, but differentiable as softmax"""
    new_shape = old_shape = logits.shape
    n_classes = old_shape[-1] if n_classes is None else n_classes
    if n_classes != old_shape[-1]:
        new_shape = (*old_shape[:-1], -1, n_classes)
    logits = logits.view(new_shape)
    # z = z_hard + z_soft - detach(z_soft)
    if noisy:
        # hard=True, returns one-hot, but differentiable as softmax
        probs = F.gumbel_softmax(logits, hard=True)
    else:
        indices = logits.argmax(dim=-1)
        hard_one_hot = F.one_hot(indices, num_classes=n_classes).float()
        softmax = logits.softmax(dim=-1)
        probs = hard_one_hot + softmax - softmax.detach()
    return probs.view(old_shape), logits


def symlog(x):
    # \text{symlog}(x) := \text{sign}(x) \ln(|x|+1)
    return tc.sign(x) * tc.log(tc.abs(x) + 1)


def symexp(x):
    # \text{symexp}(x) := \text{sign}(x) (\exp(|x|) - 1)
    return tc.sign(x) * (tc.exp(tc.abs(x)) - 1)


class TwoHot:
    def __init__(s, n_bins=32):
        s.bins = symexp(tc.linspace(-20, 20, n_bins))

    def decode(s, probs):
        return tc.sum(probs * s.bins, dim=-1)

    def encode(s, x: tc.Tensor):
        assert s.bins[0] < x.min() <= x.max() < s.bins[-1]
        i2 = tc.bucketize(x, s.bins)
        i1 = i2 - 1
        b1, b2 = s.bins[i1], s.bins[i2]
        # p1 * b1 + (1 - p1) * b2 = x
        p1 = (x - b2) / (b1 - b2)
        p2 = 1 - p1
        probs = tc.zeros((*x.shape, len(s.bins)))
        probs.scatter_(-1, i1.unsqueeze(-1), p1.unsqueeze(-1))
        probs.scatter_(-1, i2.unsqueeze(-1), p2.unsqueeze(-1))
        return probs

    def loss(s, logits, x_tar):
        return cross_entropy(logits, probs=s.encode(x_tar))
