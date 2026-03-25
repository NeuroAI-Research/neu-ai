import jax.numpy as jnp
from jax import random
from jax.nn import relu
from jax.random import normal


def mlp_params(key, sizes):
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, k2 = random.split(key)
        W = normal(k2, (a, b)) * jnp.sqrt(2 / a)
        B = jnp.zeros((b,))
        params.append((W, B))
    return params


def mlp_forward(params, x):
    for w, b in params[:-1]:
        x = relu(jnp.dot(x, w) + b)
    w, b = params[-1]
    return jnp.dot(x, w) + b


def mse_loss(pred, targ):
    return jnp.mean((pred - targ) ** 2)
