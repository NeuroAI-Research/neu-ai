import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, lax, random, vmap
from jax.nn import relu
from jax.numpy import fft
from jax.random import PRNGKey
from jax.scipy.stats import norm

from neu_ai.plot import plot1


def c3p2_discrimination():
    mu_minus, mu_plus = 25, 35
    sigma = 5
    r = jnp.linspace(0, 60, 200)

    # discriminability d'
    # d_prime = (mu_plus - mu_minus) / sigma

    # P[r|s]
    p_r_plus = norm.pdf(r, mu_plus, sigma)
    p_r_minus = norm.pdf(r, mu_minus, sigma)
    p_ratio = p_r_plus / p_r_minus
    # ROC analysis
    # alpha = P[r >= z | -] (false alarm)
    # beta  = P[r >= z | +] (hit rate)
    beta = 1 - norm.cdf(r, mu_plus, sigma)
    alpha = 1 - norm.cdf(r, mu_minus, sigma)

    R, C = 2, 2
    plt.figure(figsize=(4 * C, 3 * R))

    plt.subplot(R, C, 1)
    plt.plot(r, p_r_plus, label="p(r|+)")
    plt.plot(r, p_r_minus, label="p(r|-)")
    plt.legend()

    plt.subplot(R, C, 2)
    plt.ylabel("log scale")
    plt.plot(r, p_ratio, label="p(r|+) / p(r|-)")
    plt.axhline(1, label="ratio = 1", c="gray")
    plt.yscale("log")
    plt.legend()

    plt.subplot(R, C, 3)
    plt.plot(r, beta, label="beta: P[r >= z | +]")
    plt.plot(r, alpha, label="alpha: P[r >= z | -]")
    plt.legend()

    plt.subplot(R, C, 4)
    plt.title("P[correct] = area under ROC")
    plt.xlabel("alpha (false alarm)")
    plt.ylabel("beta (hit rate)")
    plt.plot(alpha, beta, label="ROC curve")
    plt.plot([0, 1], [0, 1], label="P[correct] = 0.5")
    plt.plot([0, 0, 1], [0, 1, 1], label="P[correct] = 1")
    plt.legend()

    plt.tight_layout()
    plt.savefig("c3p2_discrimination")


# =========================


def gaussian_r(s, mu, std, r_max):
    return r_max * jnp.exp(-0.5 * ((s - mu) / std) ** 2)


def gaussian_r_pop_sample(key, s, mu_pop, std, r_max, T):
    r = vmap(lambda mu: gaussian_r(s, mu, std, r_max))(mu_pop)
    cnt = r * T
    cnt_sampled = random.poisson(key, cnt)
    r_sampled = cnt_sampled / T
    return r_sampled


def decode_r_pop_vec(pref_angles, r_sampled, r_max):
    """
    Eqn 3.24: vector method, best for periodic / directional data
    pref_angles = populations' preferred angles
    """
    cos_sin = jnp.stack([jnp.cos(pref_angles), jnp.sin(pref_angles)], axis=1)
    weights = r_sampled / r_max
    cos, sin = jnp.dot(weights, cos_sin)
    return jnp.arctan2(sin, cos)


def decode_max_like_gau(pref_angles, r_sampled):
    """
    Eqn 3.34: Maximum Likelihood for Gaussian tuning + Poisson noise
    firing rate weighted average
    """
    return jnp.sum(r_sampled * pref_angles) / jnp.sum(r_sampled)


def decode_max_a_posteriori(pref_angles, r_sampled, std, T, s_prior, std_prior):
    """
    Eqn 3.37: Maximum A Posteriori
    combines neural evidence with a 'prior' belief
    """
    precision_data = T / (std**2)
    precision_prior = 1 / (std_prior**2)
    A = precision_data * jnp.sum(r_sampled * pref_angles) + s_prior * precision_prior
    B = precision_data * jnp.sum(r_sampled) + precision_prior
    return A / B


def c3p3_population_decoding():
    n_neuron = 50
    pref_angles = jnp.linspace(-jnp.pi, jnp.pi, n_neuron)
    s = 0.5  # true_angle
    key, std, r_max, T = PRNGKey(42), 0.5, 50, 0.2

    r_sampled = gaussian_r_pop_sample(key, s, pref_angles, std, r_max, T)

    s_guess1 = decode_r_pop_vec(pref_angles, r_sampled, r_max)
    s_guess2 = decode_max_like_gau(pref_angles, r_sampled)
    print(f"s: {s}")
    print(f"s_guess1 (vector method): {s_guess1:.3f}")
    print(f"s_guess2 (max likelihood): {s_guess2:.3f}")


# =============================


def rand_smooth_curve(key, n, dt, f_cutoff=3):
    raw = random.normal(key, (n,))
    # low pass filter
    freq = fft.rfftfreq(n, dt)
    LPF = jnp.exp(-0.5 * (freq / f_cutoff) ** 2)
    return fft.irfft(fft.rfft(raw) * LPF, n)


def exp_sin_kernel(dt):
    t = jnp.arange(0, 0.2, dt)
    return 15 * jnp.exp(-t / 0.03) * jnp.sin(2 * jnp.pi * 5 * t)


@jit
def get_spikes(key, s, kernel, dt):
    # linear filter -> relu nonlinearity -> poisson spikes
    L = jnp.convolve(s, kernel, mode="full")[: len(s)]
    r = relu(L + 5)
    return random.poisson(key, r * dt)


def get_STA(s, spikes, win=200):
    spike_idx = jnp.where(spikes > 0)[0]
    spike_idx = spike_idx[spike_idx > win]

    def get_window(i):
        return lax.dynamic_slice_in_dim(s, i - win, win)

    windows = vmap(get_window)(spike_idx)
    spikes = spikes[spike_idx, None]
    return jnp.sum(windows * spikes, axis=0) / jnp.sum(spikes)


def decode_spikes(spikes, sta_kernel):
    return jnp.convolve(spikes, sta_kernel[::-1], mode="full")[: len(spikes)]


def c3p4_spike_train_decoding():
    key, n, dt = PRNGKey(42), 2000, 1e-3

    s = 50 * rand_smooth_curve(key, n, dt)  # stimulus
    kernel = exp_sin_kernel(dt)
    k1, k2 = random.split(key)
    spikes_h1 = get_spikes(k1, s, kernel, dt)
    spikes_anti = get_spikes(k2, -s, kernel, dt)
    sta_kernel = (get_STA(s, spikes_h1) + get_STA(-s, spikes_anti)) / 2
    s_dec1 = decode_spikes(spikes_h1, sta_kernel)
    s_dec2 = decode_spikes(spikes_anti, sta_kernel)
    s_dec = s_dec1 - s_dec2

    plots = {
        "stimulus": s,
        "kernel": kernel[::-1],
        "spikes_h1": spikes_h1,
        "decoded_stimulus": s_dec,
        "sta_kernel": sta_kernel,
        "spikes_anti": spikes_anti,
    }
    plot1(plots, "c3p4_spike_train_decoding", C=3)


if __name__ == "__main__":
    c3p4_spike_train_decoding()
