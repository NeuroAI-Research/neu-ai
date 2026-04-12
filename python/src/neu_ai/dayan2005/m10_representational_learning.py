import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, random
from jax.nn import softmax
from jax.random import PRNGKey


def make_points(key, n_cluster, n_dim, n_sample=300):
    k1, k2, k3 = random.split(key, 3)
    mu = random.uniform(k1, (n_cluster, n_dim), minval=-10.0, maxval=10.0)
    cluster_ids = random.randint(k2, (n_sample,), 0, n_cluster)
    noise = random.normal(k3, (n_sample, n_dim))
    u = mu[cluster_ids] + noise
    return u, mu


@jit
def expectation_maximization(u, params):
    """
    u: (N, D) - data
    params: (mu, p_v)
        mu: (K, D) - means of K causes
        p_v: (K,) - The p_v Distribution: $P[v; G]$
    """
    mu, p_v = params

    # E-PHASE: Calculate P[v|u]
    # (N, 1, D) - (1, K, D) -> (N, K, D)
    diff = u[:, None, :] - mu[None, :, :]
    log_p_u_v = -0.5 * jnp.sum(diff**2, axis=2) / (0.2**2)
    log_bayes_top = log_p_u_v + jnp.log(p_v)
    p_v_u = softmax(log_bayes_top, axis=1)

    # M-PHASE: Update Generative Parameters
    new_p_v = jnp.mean(p_v_u, axis=0)
    A = jnp.dot(p_v_u.T, u)  # (K, N) @ (N, D) -> (K, D)
    B = jnp.sum(p_v_u, axis=0)[:, None]
    new_mu = A / B

    return (new_mu, new_p_v), p_v_u


def c10p1_representational_learning():
    key, n_cluster, n_dim = PRNGKey(0), 3, 2
    u, true_mu = make_points(key, n_cluster, n_dim)

    k1, k2 = random.split(key)
    mu = random.normal(k1, (n_cluster, n_dim))
    p_v = jnp.ones(n_cluster) / n_cluster

    for i in range(1):
        (mu, p_v), p_v_u = expectation_maximization(u, (mu, p_v))
    colors = jnp.argmax(p_v_u, axis=1)

    plt.scatter(u[:, 0], u[:, 1], c=colors, cmap="viridis", alpha=0.5, label="data")
    plt.scatter(mu[:, 0], mu[:, 1], marker="X", s=200, c="red", label="learned centers")
    plt.legend()
    plt.savefig("c10p1_representational_learning")


if __name__ == "__main__":
    c10p1_representational_learning()
