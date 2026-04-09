import jax.numpy as jnp
from jax import jit, lax, random
from jax.nn import softmax
from jax.random import PRNGKey

from theoretical_neuroscience.plot import plot1
from theoretical_neuroscience.utils import SMA, postfix


def c9p2_classical_conditioning():
    n_trials = 600
    T = 300
    t_stimulus = 100
    t_reward = 200
    lr = 0.2
    max_lag = T - t_stimulus

    @jit
    def td_learning_step(carry, _):
        w, u, r = carry
        # The Prediction (Expected Future Reward):
        # $ v(t) = \sum_{\tau=0} w(\tau) u(t-\tau) $
        # special case: pulse stimulus, u = 1 when t = t_stimulus+\tau >= t_stimulus
        v = jnp.zeros(T)
        v = v.at[t_stimulus:].set(w)

        # The TD Error (Dopamine Signal):
        # $ \delta(t) = r(t) + v(t+1) - v(t) $
        v_next = jnp.roll(v, -1).at[-1].set(0.0)
        delta = r + v_next - v

        # The Weight Update (Synaptic Change):
        # $ w(\tau) \rightarrow w(\tau) + \epsilon \delta(t) u(t-\tau) $
        w_new = w + lr * delta[t_stimulus : t_stimulus + max_lag]

        return (w_new, u, r), (v, delta)

    w_init = jnp.zeros(max_lag) + 1e-5
    u = jnp.zeros(T).at[t_stimulus].set(1.0)
    r = jnp.zeros(T).at[t_reward].set(1.0)
    (w, _, _), (v, delta) = lax.scan(
        td_learning_step, (w_init, u, r), jnp.arange(n_trials)
    )

    trials = [0, 300, 599]
    plots = {}
    for i in trials:
        plots[f"v (t_rew={t_reward}, trial={i})"] = v[i]
        plots[f"delta (t_rew={t_reward}, trial={i})"] = delta[i]

    # t_reward = 150
    # r2 = jnp.zeros(T).at[t_reward].set(1.0)
    # _, (v, delta) = td_learning_step((w, u, r2), None)
    # plots[f"v (t_rew={t_reward})"] = v
    # plots[f"delta (t_rew={t_reward})"] = delta

    plot1(plots, "c9p2_classical_conditioning")


# =============================


def c9p3_static_action_choice():
    def update_indirect(m: jnp.ndarray, act_idx, reward, epsilon):  # Model-Based
        # m_a -> m_a + epsilon * (reward - m_a)
        delta = reward - m[act_idx]
        return m.at[act_idx].add(epsilon * delta)

    def update_direct(
        m: jnp.ndarray, act_idx, reward, epsilon, probs, r_bar
    ):  # Model-Free
        # m_a' -> m_a' + epsilon * (delta_aa' - P_a') * (reward - r_bar)
        indices = jnp.arange(len(m))
        kronecker = (indices == act_idx).astype(jnp.float32)
        return m + epsilon * (kronecker - probs) * (reward - r_bar)

    def sim(type, beta=1.0, epsilon=0.1):
        n_trials = 200
        m = jnp.zeros(2)
        r_bar = 1.5
        key = PRNGKey(42)
        M, R = [], []

        for t in range(n_trials):
            probs = softmax(beta * m)
            key, k2 = random.split(key)
            act_idx = random.choice(k2, 2, p=probs)

            means = jnp.array([1.0, 2.0]) if t < n_trials / 2 else jnp.array([2.0, 1.0])
            reward = means[act_idx]
            if type == "indirect":
                m = update_indirect(m, act_idx, reward, epsilon)
            else:
                m = update_direct(m, act_idx, reward, epsilon, probs, r_bar)
            M.append(m)
            R.append(reward)
        M, R = jnp.array(M).T, jnp.array(R)
        plots = {"action value m": {"m0": M[0], "m1": M[1]}, "SMA(reward)": SMA(R, 10)}
        return postfix(plots, f" {type}")

    plots = {**sim("indirect"), **sim("direct")}
    plot1(plots, "c9p3_static_action_choice")


# ===================


def c9p4_sequential_action_choice():
    n_states = 11
    tar_state = 5
    gamma = 0.95  # exponential discounting
    lr = 0.2  # learning rate
    beta = 2.0  # softmax inverse temperature
    n_episode = 501

    @jit
    def ac_update(s, s2, r, v: jnp.ndarray, m: jnp.ndarray, act, probs):
        # CRITIC: Prediction Error delta = r + gamma * V(s') - V(s)
        delta = r + gamma * v[s2] - v[s]
        # CRITIC UPDATE: v -> v + epsilon * delta
        v = v.at[s].add(lr * delta)
        # ACTOR UPDATE: m -> m + epsilon * (Indicator - Probability) * delta
        kronecker = jnp.eye(2)[act]
        m = m.at[s].add(lr * (kronecker - probs) * delta)
        return v, m

    v = jnp.zeros(n_states)
    m = jnp.zeros((n_states, 2))
    key = PRNGKey(42)
    v_hist = {}

    for ep in range(n_episode):
        key, k2 = random.split(key)
        s = random.randint(k2, (), 0, n_states)
        while s != tar_state:
            key, k3 = random.split(key)
            probs = softmax(beta * m[s])
            act = random.choice(k3, 2, p=probs)

            ds = [-1, 1][act]
            s2 = jnp.clip(s + ds, 0, n_states - 1)
            r = 1.0 if s2 == tar_state else 0.0

            v, m = ac_update(s, s2, r, v, m, act, probs)
            s = s2
        if ep % 50 == 0:
            v_hist[ep] = v.copy()

    probs = softmax(beta * m, axis=1).T
    plots = {
        "critic v": {f"ep{i}": v_hist[i] for i in [0, 50, 500]},
        "final probs": {"p_left": probs[0], "p_right": probs[1]},
    }
    plot1(plots, "c9p4_sequential_action_choice")


if __name__ == "__main__":
    c9p4_sequential_action_choice()
