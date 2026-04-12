from functools import partial

import jax
import jax.numpy as jnp
from jax import random

from neu_ai.plot import plot1


def gaussian_tuning_curve(s, amp=50, mu=0, std=10):
    return amp * jnp.exp(-0.5 * ((s - mu) / std) ** 2)


def causal_kernel(alpha, t):
    kernel = (alpha**2) * t * jnp.exp(-alpha * t)
    return kernel / jnp.sum(kernel)


@partial(jax.vmap, in_axes=(0, None, None))
def poisson_spikes(key, rates: jnp.ndarray, dt: float):
    probs = rates * dt
    spikes = random.uniform(key, rates.shape) < probs
    return spikes.astype(jnp.float32)


@partial(jax.vmap, in_axes=(0, None))
def spikes_to_rates(spikes, kernel):
    conv = jnp.convolve(spikes, kernel, mode="full")
    return conv[: len(spikes)]


def demo_spikes_to_rates():
    key = random.PRNGKey(42)
    dt, duration = 1e-3, 10
    t = jnp.arange(0, duration, dt)
    alpha = 100
    n_trials = 100

    stimulus_path = jnp.linspace(-20, 20, len(t))
    true_rates = gaussian_tuning_curve(stimulus_path)
    kernel = causal_kernel(alpha, t=jnp.arange(0, 0.1, dt))

    keys = random.split(key, n_trials)
    spikes = poisson_spikes(keys, true_rates, dt)
    rates = spikes_to_rates(spikes, kernel) / dt  # scale by 1/dt to get Hz

    spike_times = t[spikes[0] > 0]
    ISI = jnp.diff(spike_times)

    plots = {
        "true_rates": true_rates,
        "kernel": kernel,
        "spikes[0]": spikes[0],
        "rates.mean(0)": rates.mean(0),
        "Inter_Spike_Interval.hist": ISI,
    }
    plot1(plots, "demo_spikes_to_rates")


# ======================================================


@partial(jax.vmap, in_axes=(0, None, None))
def find_STA_window(t_i, s_t, tau):
    # equivalent to s_t[t_i - tau : t_i]
    return jax.lax.dynamic_slice_in_dim(s_t, t_i - tau, tau)


def find_STA(s_t, t_i, tau):
    valid_t_i = t_i[t_i >= tau]
    windows = find_STA_window(valid_t_i, s_t, tau)
    return windows.mean(0)


def find_Q_rs(r, s):
    return jnp.convolve(r, s[::-1], mode="same")


def demo_find_STA():
    key = random.PRNGKey(42)
    key2 = random.PRNGKey(1)
    duration, dt = 10, 1e-3
    T = int(duration / dt)
    tau = 50

    s = random.normal(key, (T,))  # white noise

    x = jnp.linspace(0, tau, tau)
    true_feature = jnp.exp(-((x - 20) ** 2) / (2 * 5**2))
    r = jnp.convolve(s, true_feature, mode="full")[:T]
    r = jax.nn.relu(r)
    spikes = random.bernoulli(key2, r * dt * 100)
    t_i = jnp.where(spikes)[0]

    STA = find_STA(s, t_i, tau)
    plots = {
        "stimulus": s,
        "true_features": true_feature,
        "firing rate": r,
        "spikes": spikes,
        "STA": STA,
    }
    plot1(plots, "demo_find_STA")
