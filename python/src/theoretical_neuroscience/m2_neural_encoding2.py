import jax.numpy as jnp
from jax import lax, random, vmap
from jax.nn import relu, sigmoid
from jax.random import PRNGKey

from theoretical_neuroscience.m1_neural_encoding import poisson_spikes
from theoretical_neuroscience.plot import VideoMaker, plot1
from theoretical_neuroscience.utils import frame_to_jax, read_video


def predict_firing_rate(stimulus, kernel, r0, act="relu"):
    # L: linear filter output
    L = jnp.convolve(stimulus, kernel, mode="full")[: len(stimulus)]
    if act == "relu":
        F_L = relu(L)
    elif act == "sigmoid":
        r_max, g1, L_half = 100, 2, 3
        F_L = r_max * sigmoid(g1 * (L - L_half))
    return r0 + F_L


def demo_predict_firing_rate():
    key1 = PRNGKey(42)
    key2 = random.split(PRNGKey(42), 3)
    dt = 1e-3
    r0 = 100

    stimulus = random.normal(key1, (500,))
    kernel = jnp.exp(-jnp.linspace(0, 10, 50))

    r = predict_firing_rate(stimulus, kernel, r0)
    spikes = poisson_spikes(key2, r, dt)

    plots = {
        "stimulus": stimulus,
        "kernel": kernel,
        "firing rate": r,
        "spikes[0]": spikes[0],
    }
    plot1(plots, "3_demo_predict_firing_rate")


# =================================


def conv_2d(stimulus: jnp.ndarray, kernel_2d: jnp.ndarray):
    s = stimulus[None, None, :, :]
    k = kernel_2d[None, None, :, :]
    s2 = lax.conv_general_dilated(s, k, window_strides=(1, 1), padding="SAME")
    return s2.squeeze()


def center_surround_kernel(sig_cen=1, sig_sur=3, B=1):
    x = jnp.linspace(-10, 10, 21)
    x, y = jnp.meshgrid(x, x)
    r2 = x**2 + y**2

    def gau_2d(s):
        return 1 / (2 * jnp.pi * s**2) * jnp.exp(-r2 / (2 * s**2))

    return gau_2d(sig_cen) - B * gau_2d(sig_sur)


def retinal_step(state, frame, kernel_2d, r0=0):
    key = state

    # 1. photoreceptor: normalize the current frame relative to its own mean
    # bio: rapid adaption to local light levels
    avg_lum = jnp.mean(frame)
    s = (frame - avg_lum) / (avg_lum + 1e-5)

    # 2. bipolar cells: center-surround filtering (linear filter)
    # math: L = ∫ dτ D s (2D spatial convolution)
    graded_potential = conv_2d(s, kernel_2d)

    # 3. ganglion cells: static nonlinearity & Poisson spiking
    # math: r_est = r0 + F(L)
    firing_rate = relu(r0 + graded_potential)
    this_key, next_key = random.split(key)
    spikes = random.poisson(this_key, firing_rate)

    data = {
        "frame": frame,
        # "kernel_2d": kernel_2d,
        "stimulus": s,
        "graded_potential": graded_potential,
        "firing_rate": firing_rate,
        "spikes": spikes,
    }
    return next_key, data


def demo_retinal_step():
    path = "./data/cat_dance.mp4"
    key = PRNGKey(42)
    # kernel_2d = jnp.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=jnp.float32)
    kernel_2d = center_surround_kernel()
    vm = VideoMaker("4_demo_retinal_step.mp4")

    for frame in read_video(path):
        frame = frame_to_jax(frame)
        key, data = retinal_step(key, frame, kernel_2d)
        # plot1(postfix(data, ".img"), "4_demo_retinal_step")
        vm.add(data)
    vm.release()


# ==========================


def gabor_kernel_2d(theta=0, phi=0, k=0.5, sigma=3, size=21):
    x = jnp.linspace(-size // 2, size // 2, size)
    x, y = jnp.meshgrid(x, x)
    sin, cos = jnp.sin(theta), jnp.cos(theta)
    # rotate coordinates for orientation selectivity
    xr = x * cos + y * sin
    yr = -x * sin + y * cos
    envelope = jnp.exp(-(xr**2 + yr**2) / (2 * sigma**2))
    ker = envelope * jnp.cos(k * xr - phi)
    return ker / jnp.sum(jnp.abs(ker))


def temporal_kernel(T=50, dt=0.01, alpha=1 / 0.015):
    tau = jnp.arange(T) * dt
    at = alpha * tau
    A, B = (at**5) / 120, (at**7) / 5040
    return alpha * jnp.exp(-at) * (A - B)


def v1_simple_step(key, prev_data, retina_out, kernel_2d, r_fast=0.6, r_slow=0.9):
    spatial_res = conv_2d(retina_out, kernel_2d)
    L_fast = r_fast * prev_data["L_fast"] + (1 - r_fast) * spatial_res
    L_slow = r_slow * prev_data["L_slow"] + (1 - r_slow) * spatial_res
    L = L_fast - L_slow
    r = relu(L) ** 2
    this_key, next_key = random.split(key)
    spikes = random.poisson(this_key, r)
    data = {
        "retina_out": retina_out,
        # "spatial_res": spatial_res,
        "L_fast": L_fast,
        "L_slow": L_slow,
        # "L": L,
        "firing_rate": r,
        "spikes": spikes,
    }
    return next_key, data


def v1_simple_batch(retina_out, kernel_2d, kernel_time):
    spatial_res = vmap(conv_2d, in_axes=(0, None))(retina_out, kernel_2d)

    def conv_time(pixel_hist):
        return jnp.convolve(pixel_hist, kernel_time, mode="same")

    conv_time = vmap(vmap(conv_time, 1), 1)
    L = conv_time(spatial_res).transpose(2, 0, 1)
    r = relu(L) ** 2
    return r


def v1_complex_batch(retina_out, kernel_2d, kernel_time):
    simp_r = vmap(v1_simple_batch, in_axes=(None, 0, None))(
        retina_out, kernel_2d, kernel_time
    )
    comp_r = jnp.sum(simp_r, axis=0)
    return simp_r, comp_r


def demo_v1_step():
    retina_kernel_2d = jnp.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], jnp.float32)
    kernel_2d = gabor_kernel_2d(theta=jnp.pi / 4)  # 45-degree detector
    vm = VideoMaker("5_demo_v1_step.mp4")
    size = (224, 224)

    key = PRNGKey(42)
    d2 = {"L_fast": jnp.zeros(size), "L_slow": jnp.zeros(size)}

    for frame in read_video("./data/cat_dance.mp4"):
        frame = frame_to_jax(frame, size)
        key, d1 = retinal_step(key, frame, retina_kernel_2d)
        key, d2 = v1_simple_step(key, d2, frame, kernel_2d)
        vm.add(d2)
    vm.release()


def demo_v1_batch():
    size = (224, 224)
    frames = []
    for frame in read_video("./data/cat_dance.mp4"):
        frames.append(frame_to_jax(frame, size))
    frames = jnp.array(frames)

    phases = jnp.arange(4) * jnp.pi / 2
    kernel_2d = vmap(lambda p: gabor_kernel_2d(phi=p))(phases)
    kernel_time = temporal_kernel()

    simp_r, comp_r = v1_complex_batch(frames, kernel_2d, kernel_time)

    vm = VideoMaker("5_demo_v1_batch.mp4")
    for t in range(len(frames)):
        plots1 = {f"simple_cell_{k}": simp_r[k, t] for k in range(len(phases))}
        vm.add({"stimulus": frames[t], **plots1, "complex_cell": comp_r[t]})
    vm.release()


if __name__ == "__main__":
    demo_v1_batch()
