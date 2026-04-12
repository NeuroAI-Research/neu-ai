import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import lax, random
from jax.random import PRNGKey, multivariate_normal, normal


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


# ======================


def hebbian_dw(w, u, v_targ):
    return jnp.outer(v_targ, u)


def grad_dw(w, u, v_targ):
    # d( (v_targ - w u)**2 ) / dw = 2 * (w u - v_targ) * u
    return jnp.outer(v_targ - jnp.dot(w, u), u)


def wake_sleep_dw(w, u, v_targ):
    # wake phase: clamped to reality (u, v_targ)
    # sleep phase: "dreams" an output
    return jnp.outer(v_targ, u) - jnp.outer(jnp.dot(w, u), u)


def c8p4_supervised_learning():
    k1, k2 = random.split(PRNGKey(42))
    d_in, d_out = 10, 2
    n_sample = 100

    U = normal(k1, (n_sample, d_in))
    W_target = normal(k2, (d_out, d_in))
    V_target = jnp.dot(U, W_target.T)

    def train(dw_fn):
        w = jnp.zeros((d_out, d_in))
        lr = 1e-3
        mse = []
        for _ in range(50):
            for i in range(n_sample):
                w += lr * dw_fn(w, U[i], V_target[i])
            V = jnp.dot(U, w.T)
            mse.append(jnp.mean((V - V_target) ** 2))
        plt.plot(mse, label=dw_fn.__name__)

    for dw_fn in [hebbian_dw, grad_dw, wake_sleep_dw]:
        train(dw_fn)
    plt.legend()
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    c8p4_supervised_learning()
