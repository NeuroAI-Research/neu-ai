import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import lax
from jax.random import PRNGKey, multivariate_normal


def c8p2_synaptic_plasticity_rules():
    steps = 1000
    lr = 0.01  # = 1/tau_w
    alpha = 1
    key = PRNGKey(42)

    mean = jnp.zeros(2)
    cov = jnp.array([[2.0, 1.8], [1.8, 2.0]])
    u = multivariate_normal(key, mean, cov, (steps,))

    def oja_step(w, u):
        v = jnp.dot(w, u)
        dw = lr * (v * u - alpha * (v**2) * w)
        new_w = w + dw
        return new_w, (new_w, v)

    w0 = jnp.array([1.0, 0.0])
    w, (w_hist, v_hist) = lax.scan(oja_step, w0, u)
    w_norm = jnp.linalg.norm(w_hist, axis=1)

    plt.figure(figsize=(4 * 2, 3))
    plt.subplot(1, 2, 1)
    plt.scatter(u[:, 0], u[:, 1], alpha=0.1, label="input: u")
    plt.quiver(0, 0, w0[0], w0[1], scale=3, label="initial weights")
    plt.quiver(0, 0, w[0], w[1], scale=3, color="red", label="final weights")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(w_norm, c="gray", label="w_norm")
    plt.axhline(1 / jnp.sqrt(alpha), label="target norm")
    plt.legend()

    plt.tight_layout()
    plt.savefig("c8p2_synaptic_plasticity_rules")


if __name__ == "__main__":
    c8p2_synaptic_plasticity_rules()
