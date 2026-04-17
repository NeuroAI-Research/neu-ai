import jax.numpy as jnp
from flax import nnx
from jax import Array

from neu_ai.plot import plot1, plot_img_patches
from neu_ai.utils import read_pdf


class PatchEmbed(nnx.Module):
    def __init__(s, patch_size, d_embed, rngs, n_channel=3):
        s.conv = nnx.Conv(
            in_features=n_channel,
            out_features=d_embed,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            rngs=rngs,
            padding="VALID",
        )

    def __call__(s, x):
        # x: (batch, height, width, n_channel)
        return s.conv(x)


def mid_split(x: Array):
    mid = x.shape[-1] // 2
    return x[..., :mid], x[..., mid:]


class RoPE1D(nnx.Module):
    def __init__(s, d, T, base=1e4):
        wt = s.wt(T, s.omega(base, d))
        # We repeat the frequencies so they match the shape of the features
        wt = jnp.concat([wt, wt], axis=-1)
        s.cos, s.sin = jnp.cos(wt), jnp.sin(wt)

    def omega(s, base, d):
        # We only need d/2 frequencies because they apply to pairs
        return 1.0 / (base ** (jnp.arange(0, d, 2) / d))

    def wt(s, T, omega):
        return jnp.outer(jnp.arange(T), omega)

    def rotate_x(s, x):
        x1, x2 = mid_split(x)
        return jnp.concat([-x2, x1], axis=-1)

    def __call__(s, x: Array):
        # x: (batch, seq_len, n_head, d_head)
        _, T, _, _ = x.shape
        # Reshape for broadcasting over heads
        cos = s.cos[None, :T, None, :]
        sin = s.sin[None, :T, None, :]
        # Apply the "Rotary" transformation:
        # (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
        # We use a trick: [x1, x2] -> [-x2, x1]
        return x * cos + s.rotate_x(x) * sin


class RoPE2D(RoPE1D):
    def __init__(s, d, H, W, base=1e4):
        # We split 'dim' into two parts: half for H, half for W
        # Each axis needs its own d/2 dimensions (so d/4 pairs each)
        omega = s.omega(base, d // 2)
        wt_H = s.wt(H, omega)
        wt_W = s.wt(W, omega)
        # Broadcast to 2D grid: (H, W, d/4)
        wt_H = jnp.tile(wt_H[:, None, :], (1, W, 1))
        wt_W = jnp.tile(wt_W[None, :, :], (H, 1, 1))
        # Concatenate H and W frequencies along the feature dimension
        # Total feature dim: (H, W, d/4) -> (H, W, d)
        wt = jnp.concat([wt_H, wt_H, wt_W, wt_W], axis=-1)
        s.cos, s.sin = jnp.cos(wt), jnp.sin(wt)

    def __call__(s, x: Array):
        # x: (batch, H, W, n_head, d_head)
        _, H, W, _, _ = x.shape
        cos = s.cos[None, :H, :W, None, :]
        sin = s.sin[None, :H, :W, None, :]
        x1, x2 = mid_split(x)
        x_rotated = jnp.concat([s.rotate_x(x1), s.rotate_x(x2)], axis=-1)
        return x * cos + x_rotated * sin


def main():
    path = "../papers/2025-03-12_Gemma_3_Technical_Report.pdf"
    pages = read_pdf(path, range(3))
    img, txt = pages[0]

    patch_size, d_embed, rngs = 16, 8, nnx.Rngs(0)
    patch_embed = PatchEmbed(patch_size, d_embed, rngs)
    rope2d = RoPE2D(d_embed, H=200, W=200)

    x0 = jnp.array([img])
    x1 = patch_embed(x0)
    x2 = rope2d(jnp.ones_like(x1[:, :, :, None, :]))
    x3 = x2.squeeze()

    print(f"""
img: {x0.shape}
after PatchEmbed: {x1.shape}
after RoPE2D: {x2.shape}
    """)
    plot_img_patches(img, patch_size, "PatchEmbed")

    plots = {}
    for i in range(d_embed):
        plots[f"x3[..., {i}].img"] = x3[..., [i]]
    plot1(plots, "RoPE2D", C=d_embed // 2)


if __name__ == "__main__":
    main()
