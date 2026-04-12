import jax.numpy as jnp
import jax.numpy.fft as F
import matplotlib.pyplot as plt
from jax.random import PRNGKey

from neu_ai.plot import plot1
from neu_ai.utils import frame_to_jax, read_video

from .m1_neural_encoding import poisson_spikes


def surprise(prob):
    return jnp.where(prob > 0, -jnp.log2(prob), 0)


def entropy(prob_dist):
    p = prob_dist
    return jnp.sum(p * surprise(p))


def cont_entropy(p_r, dr):
    return jnp.sum(dr * p_r * surprise(p_r))


def kl_divergence(prob_dist1, prob_dist2):
    p1, p2 = prob_dist1, prob_dist2
    mask = (p1 > 0) & (p2 > 0)
    p1, p2 = p1[mask], p2[mask]
    return jnp.sum(p1 * jnp.log2(p1 / p2))


def c4p1_entropy_and_mutual_information():
    prob = jnp.linspace(0, 1, 200)
    sur = surprise(prob)
    ent = entropy(prob / prob.sum())
    print(f"ent: {ent}")
    plots = {"surprise": [prob, sur]}
    plot1(plots, "c4p1_entropy_and_mutual_information")


# =======================


def image_fft(img: jnp.ndarray, filter="band_pass"):
    freq = F.fftshift(F.fft2(img))  # shift zero freq to center

    w, h = img.shape
    u, v = jnp.meshgrid(jnp.linspace(-1, 1, w), jnp.linspace(-1, 1, h))
    kappa = jnp.sqrt(u**2 + v**2)
    if filter == "band_pass":
        filter = kappa * jnp.exp(-(kappa**2) / 0.5)

    img2 = jnp.abs(F.ifft2(F.ifftshift(freq * filter)))
    return img2, filter


def c4p2_information_and_entropy_maximization():
    # optimal tuning curve
    s = jnp.linspace(-5, 5, 1000)
    p_s = 0.7 * jnp.exp(-((s - 1) ** 2) / 0.5) + 0.3 * jnp.exp(-((s + 2) ** 2) / 2)
    p_s /= jnp.trapezoid(p_s, s)  # normalize
    r_max = 100
    f_s_opt = r_max * jnp.cumsum(p_s) * (s[1] - s[0])

    # fft: band pass filtering
    for img in read_video("./data/cat_dance.mp4"):
        img = frame_to_jax(img)
        img2, filter = image_fft(img)
        break

    plots = {
        "p(s)": [s, p_s],
        "optimal f(s)": [s, f_s_opt],
        "img.img": img,
        "FFT_band_pass.img": filter,
        "new_img.img": img2,
    }
    plot1(plots, "c4p2_information_and_entropy_maximization")


# ======================================


def reshape_seq(long_seq: jnp.ndarray, seq_len):
    n_seq = len(long_seq) // seq_len
    return long_seq[: n_seq * seq_len].reshape((n_seq, seq_len))


def binary_to_num(binaries: jnp.ndarray):
    n, bits = binaries.shape
    powers = 2.0 ** jnp.arange(bits)[::-1]
    return jnp.dot(binaries, powers)


def estimate_probs(values):
    _, counts = jnp.unique(values, return_counts=True)
    return counts / len(values)


def find_entropy_rate(spikes: jnp.ndarray, seq_len, dt):
    words = reshape_seq(spikes, seq_len)
    values = binary_to_num(words)
    ent = entropy(estimate_probs(values))
    return ent / (seq_len * dt)


def poisson_entropy_rate(rates, dt):
    avg_rate = jnp.mean(rates)
    return (avg_rate / jnp.log(2)) * (1 - jnp.log(avg_rate * dt))


def c4p3_entropy_and_information_for_spike_trains():
    key = PRNGKey(42).reshape(1, -1)
    T, dt = 30_000, 0.003
    rates = jnp.full(T, 20)
    seq_lens = jnp.arange(4, 11)

    spikes = poisson_spikes(key, rates, dt)[0]

    ent_rates = []
    for seq_len in seq_lens:
        ent_rates.append(find_entropy_rate(spikes, seq_len, dt))

    theory_ent_rate = poisson_entropy_rate(rates, dt)

    plt.plot(seq_lens, ent_rates, ".", label="entropy rate")
    plt.axhline(theory_ent_rate, label="theory")
    plt.xlabel("seq_len")
    plt.ylabel("entropy rate (bits/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("c4p3_entropy_and_information_for_spike_trains")


if __name__ == "__main__":
    c4p3_entropy_and_information_for_spike_trains()
