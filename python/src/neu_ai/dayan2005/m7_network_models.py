import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import jit, lax, random, vmap
from jax.nn import relu, sigmoid
from jax.random import PRNGKey, normal, uniform

from neu_ai.plot import plot1
from neu_ai.utils import gaussian

from .ann import mlp_forward, mlp_params, mse_loss


def c7p2_firing_rate_models():
    tau_r = 0.01  # time constant: 10ms
    dt = 0.001  # simulation time step: 1ms
    tot_t = 0.1
    n_steps = int(tot_t / dt)
    n_in, n_out = 5, 4  # network size
    k1, k2, k3 = random.split(PRNGKey(42), 3)

    W = normal(k1, (n_out, n_in)) * 0.5  # feedforward weights
    M = normal(k2, (n_out, n_out)) * 0.5  # recurrent weights
    u = uniform(k3, (n_in,))

    def F(I):
        return relu(I - 0.1)

    def rate_dynamics(v, _):
        I = jnp.dot(W, u) + jnp.dot(M, v)
        dv_dt = (-v + F(I)) / tau_r
        v_next = v + dv_dt * dt
        return v_next, v_next

    v_init = jnp.zeros(n_out)
    xs = jnp.arange(n_steps)
    _, v_hist = lax.scan(rate_dynamics, v_init, xs)

    # ANN: dv_dt = 0, M = 0
    v_steady_M0 = F(jnp.dot(W, u))

    R, C = 2, 2
    for i in range(n_out):
        plt.subplot(R, C, i + 1)
        plt.plot(xs * dt, v_hist[:, i], ".", label="dynamic & M != 0")
        plt.axhline(v_steady_M0[i], label="steady & M = 0", c="red", linestyle="--")
        plt.title(f"neuron {i}")
        plt.legend()
    plt.tight_layout()
    plt.savefig("c7p2_firing_rate_models")


# =====================================


def c7p3_feedforward_networks_ANN():
    n_samples = 5000
    k1, k2 = random.split(PRNGKey(42))
    s = uniform(k1, (n_samples, 1), minval=-40, maxval=40)
    g = uniform(k2, (n_samples, 1), minval=-20, maxval=20)
    x = jnp.concat([s, g], axis=1)
    y = s + g

    params = mlp_params(PRNGKey(42), [2, 32, 32, 1])
    opt = optax.adam(learning_rate=1e-3)
    opt_state = opt.init(params)

    @jit
    def loss_fn(params, x, y):
        return mse_loss(mlp_forward(params, x), y)

    @jit
    def update(params, opt_state, x, y):
        grads = jax.grad(loss_fn)(params, x, y)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state

    for step in range(1001):
        params, opt_state = update(params, opt_state, x, y)
        if step % 200 == 0:
            print(f"step: {step}, loss: {loss_fn(params, x, y):.4f}")


def c7p3_feedforward_networks_BNN():
    # input:
    s = jnp.linspace(-60, 60, 200)
    g = jnp.array([0, 10, -20])

    # parameters:
    s_pref = jnp.linspace(-120, 120, 100)
    g_pref = jnp.linspace(-120, 120, 100)
    s_pref, g_pref = jnp.meshgrid(s_pref, g_pref)
    mu = 0
    weights = gaussian(s_pref + g_pref, mu, 20)

    @jit
    def forward(s, g):
        # retinal_term * gaze_dependent_gain_modulation
        r1 = gaussian(s, s_pref, 10) * gaussian(g, g_pref, 15)
        r2 = jnp.sum(r1 * weights)
        return r2

    f1 = vmap(forward, in_axes=(0, None))
    f2 = vmap(f1, in_axes=(None, 0))
    r2 = f2(s, g)

    styles = ["-", "--", ":"]
    colors = ["black", "blue", "red"]
    for i, g_i in enumerate(g):
        r_i = r2[i]
        s_max = s[jnp.argmax(r_i)]
        label = f"g: {g_i}, s_max: {s_max:.1f}, s+g: {s_max + g_i:.1f}"
        plt.plot(s, r_i, styles[i], c=colors[i], label=label)
    plt.xlabel("s")
    plt.ylabel("response")
    plt.title(f"response from A SINGLE NEURON that prefers s+g={mu}")
    plt.legend()
    plt.savefig("c7p3_feedforward_networks_BNN")


# =============================


def bio_rnn(
    lambda_max=0.8,
    N=100,  # number of neurons
    tau=10,
    dt=0.1,
    steps=1000,
):
    x = jnp.linspace(0, 2 * jnp.pi, N)
    e1 = jnp.sin(x)
    e1 = e1 / jnp.linalg.norm(e1)
    M = lambda_max * jnp.outer(e1, e1)

    h = sum([jnp.sin(n * x) for n in range(3)])

    def scan(v, _):
        RHS = -v + h + jnp.dot(M, v)
        v_next = v + RHS / tau * dt
        return v_next, v_next

    _, v_hist = lax.scan(scan, jnp.zeros(N), jnp.arange(steps))
    return {"input: h": h, "output: final v": v_hist[-1]}


def c7p4_recurrent_networks():
    r1 = bio_rnn()
    plots = {
        "selective_amplification": r1,
    }
    plot1(plots, "c7p4_recurrent_networks")


# =================


def c7p5_excitatory_inhibitory_networks():
    # M_EE, M_EI, M_IE, M_II
    M = jnp.array([[1.25, -1.0], [1.0, 0.0]])
    gamma = jnp.array([-10.0, 10.0])
    tau_E = 0.01
    tau_I_bif = tau_E * (1 - M[1, 1]) / (M[0, 0] - 1)

    def simulate(tau_I, steps=2000, dt=1e-3):
        tau = jnp.array([tau_E, tau_I])

        @jit
        def step_fn(v, _):
            next_v = v + dt / tau * (-v + relu(M @ v - gamma))
            return next_v, next_v

        v0 = jnp.array([20.0, 20.0])
        _, v_hist = lax.scan(step_fn, v0, jnp.arange(steps))
        return v_hist

    tau_Is = jnp.arange(0.01, 0.1, 0.002)
    min_vE, max_vE = [], []

    tau_I_plt = [0.03, 0.09]
    R, C, i = len(tau_I_plt) + 1, 2, 0
    plt.figure(figsize=(4 * C, 3 * R))

    for tau_I in tau_Is:
        tau_I = round(float(tau_I), 3)
        vE, vI = simulate(tau_I).T
        vE_stable = vE[-500:]
        min_vE.append(jnp.min(vE_stable))
        max_vE.append(jnp.max(vE_stable))
        if tau_I in tau_I_plt:
            i += 1
            plt.subplot(R, C, i)
            plt.title(f"tau_I: {tau_I}")
            plt.plot(vE, label="vE")
            plt.plot(vI, label="vI")
            plt.legend()
            i += 1
            plt.subplot(R, C, i)
            plt.title(f"tau_I: {tau_I}")
            plt.plot(vE, vI, label="(vE, vI)")
            plt.legend()

    i += 1
    plt.subplot(R, C, i)
    plt.plot(tau_Is, max_vE, label="max_vE")
    plt.plot(tau_Is, min_vE, label="min_vE")
    plt.axvline(tau_I_bif, linestyle="--", label="tau_I for bifurcation")
    plt.xlabel("tau_I")
    plt.legend()

    plt.tight_layout()
    plt.savefig("c7p5_excitatory_inhibitory_networks")


# ========================


def c7p6_stochastic_networks():
    N = 10  # number of neurons
    key = PRNGKey(0)
    M_key, h_key, key0 = random.split(key, 3)
    steps = 5000

    M = normal(M_key, (N, N))
    M = (M + M.T) / 2
    M = M.at[jnp.diag_indices(N)].set(0)
    h = normal(h_key, (N,))

    def energy(v):
        return -jnp.dot(h, v) - 0.5 * jnp.dot(v, jnp.dot(M, v))

    def gibbs_step(state, _):
        v, key = state
        key, idx_key, flip_key = random.split(key, 3)
        # pick a random neuron to update (Glauber dynamics)
        idx = random.randint(idx_key, (), 0, N)
        I_a = h[idx] + jnp.dot(M[idx], v)
        prob = sigmoid(I_a)
        new_bit = random.bernoulli(flip_key, prob).astype(jnp.float32)
        v = v.at[idx].set(new_bit)
        return (v, key), energy(v)

    v0 = jnp.zeros(N)
    _, gibbs_energies = lax.scan(gibbs_step, (v0, key0), jnp.arange(steps))

    plots = {"gibbs_energies": gibbs_energies, "gibbs_energies.hist": gibbs_energies}
    plot1(plots, "c7p6_stochastic_networks")


if __name__ == "__main__":
    c7p6_stochastic_networks()
