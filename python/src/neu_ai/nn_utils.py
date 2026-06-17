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


def to_np(x):
    if isinstance(x, tc.Tensor):
        return x.detach().numpy()
    if isinstance(x, (list, tuple)):
        return [to_np(v) for v in x]
    return x


def tensor(x):
    if isinstance(x, np.ndarray):
        return tc.from_numpy(x).float()
    if isinstance(x, (list, tuple)):
        return [tensor(v) for v in x]
    if callable(x):

        def func(*args):
            return tensor(x(*to_np(args)))

        return func


def opt_step(opt: tc.optim.Adam, loss: tc.Tensor, w={}):
    if isinstance(loss, dict):
        loss = sum(w[k] * loss[k] for k in loss)
    opt.zero_grad()
    loss.backward()
    opt.step()


def cat(*args):
    return tc.cat(args, dim=-1)


def stack_rows(rows: List[List[tc.Tensor]], dim):
    return [tc.stack([r[i] for r in rows], dim) for i in range(len(rows[0]))]


def cross_entropy(logits: tc.Tensor, probs: tc.Tensor):
    assert logits.shape == probs.shape
    logp = F.log_softmax(logits, dim=-1)
    return -tc.sum(probs * logp, dim=-1).mean()


def onehot_softmax(logits: tc.Tensor, noisy: bool, n_classes: int = None):
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
    def __init__(s, n_bins):
        s.bins = symexp(tc.linspace(-20, 20, n_bins))

    def decode(s, probs):
        return tc.sum(probs * s.bins, dim=-1)

    def decode_logits(s, logits):
        return s.decode(probs=F.softmax(logits, dim=-1))

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

    def loss(s, logits: tc.Tensor, x_tar: tc.Tensor):
        assert logits.shape == (*x_tar.shape, len(s.bins))
        return cross_entropy(logits, probs=s.encode(x_tar))

    def init(s, m: nn.Linear):
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)
