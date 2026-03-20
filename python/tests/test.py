import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.stats import norm


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


if __name__ == "__main__":
    c3p2_discrimination()
