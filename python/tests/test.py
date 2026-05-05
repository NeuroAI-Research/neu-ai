import jax.numpy as jnp
import optax
import tiktoken
from flax import nnx
from jax import Array, lax, random
from jax.nn import silu, softmax
from jax.random import PRNGKey, categorical


class RoPE:
    def __init__(s, T, d_head, base=1e4):
        omega = 1.0 / (base ** (jnp.arange(0, d_head, 2) / d_head))
        s.e_iwt = jnp.exp(1j * jnp.outer(jnp.arange(T), omega))

    def __call__(s, q: Array):
        B, n_head, T, d_head = q.shape
        z = lax.complex(q[..., ::2], q[..., 1::2]) * s.e_iwt[:T]
        return jnp.stack([jnp.real(z), jnp.imag(z)], axis=-1).reshape(q.shape)


class RMSNorm(nnx.Module):
    def __init__(s, dim, eps=1e-6):
        s.gain = nnx.Param(jnp.ones(dim))
        s.eps = eps

    def __call__(s, x):
        MS = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * lax.rsqrt(MS + s.eps) * s.gain


# =========================================


class LLMConf:
    n_vocab: int
    mask: Array
    roPE: RoPE

    max_T: int = 1000
    n_head: int = 8  # n_query_head
    n_kv_head: int = 2
    n_layer: int = 3
    d_model: int = 64
    d_ff: int = 64
    dropout = 0.1
    rngs = nnx.Rngs(0)

    @property
    def d_head(s):
        return s.d_model // s.n_head

    def linear(s, a, b):
        return nnx.Linear(a, b, use_bias=False, rngs=s.rngs)


def to_batch(seq, T):
    B = len(seq) // T
    return jnp.array(seq[: B * T]).reshape(B, T)


def causal_mask(T):
    """
    causal_mask(3):
        [[0. 1. 1.]
        [0. 0. 1.]
        [0. 0. 0.]]
    """
    return jnp.triu(jnp.ones((T, T)), k=1)


class TextEmbed(nnx.Module):
    def __init__(s, c: LLMConf):
        s.c = c
        s.token_embed = nnx.Embed(c.n_vocab, c.d_model, rngs=c.rngs)

    def __call__(s, x: Array):
        # x: (B, T)
        return s.token_embed(x) * jnp.sqrt(s.c.d_model)

    def undo(s, x: Array):
        W: Array = s.token_embed.embedding
        return x @ W.T


class GroupedQueryAttention(nnx.Module):
    def __init__(s, c: LLMConf):
        s.c = c
        assert c.d_model % c.n_head == 0
        s.d_head = c.d_model // c.n_head
        s.WQ = c.linear(c.d_model, c.d_model)
        s.WK = c.linear(c.d_model, c.n_kv_head * s.d_head)
        s.WV = c.linear(c.d_model, c.n_kv_head * s.d_head)
        s.WO = c.linear(c.d_model, c.d_model)
        s.q_norm = RMSNorm((c.n_head, 1, s.d_head))
        s.k_norm = RMSNorm((c.n_kv_head, 1, s.d_head))

    def split_heads(s, x: Array):
        B, T, d_model = x.shape
        x = x.reshape(B, T, -1, s.d_head)
        return x.transpose(0, 2, 1, 3)  # (B, n_head, T, d_head)

    def merge_heads(s, x: Array):
        B, n_head, T, d_head = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, n_head * d_head)

    def repeat_kv(s, x: Array):
        return jnp.repeat(x, s.c.n_head // s.c.n_kv_head, axis=1)

    def __call__(s, x: Array):
        B, T, d_model = x.shape
        q, k, v = s.WQ(x), s.WK(x), s.WV(x)
        q, k, v = map(s.split_heads, (q, k, v))
        q, k = s.q_norm(q), s.k_norm(k)
        q, k = map(s.c.roPE, (q, k))
        k, v = map(s.repeat_kv, (k, v))
        # (B, n_head, T, d_head) @ (B, n_head, d_head, T) -> (B, n_head, T, T)
        qk = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(s.d_head)
        if s.c.mask is not None:
            qk = jnp.where(s.c.mask[:T, :T], -jnp.inf, qk)
        context = jnp.matmul(softmax(qk, axis=-1), v)
        return s.WO(s.merge_heads(context))


class SwiGLU(nnx.Module):
    def __init__(s, c: LLMConf):
        s.W1 = c.linear(c.d_model, c.d_ff)
        s.W3 = c.linear(c.d_model, c.d_ff)
        s.W2 = c.linear(c.d_ff, c.d_model)

    def __call__(s, x):
        return s.W2(silu(s.W1(x)) * s.W3(x))


class TransformerBlock(nnx.Module):
    def __init__(s, c: LLMConf):
        s.attention = GroupedQueryAttention(c)
        s.swiGLU = SwiGLU(c)
        s.norm1 = RMSNorm(c.d_model)
        s.norm2 = RMSNorm(c.d_model)

    def __call__(s, x: Array):
        # Pre-LayerNorm: more stable gradients than the original Post-LayerNorm
        x = x + s.attention(s.norm1(x))
        x = x + s.swiGLU(s.norm2(x))
        return x


class LLM(nnx.Module):
    def __init__(s, c: LLMConf):
        s.c = c
        s.embed = TextEmbed(c)
        s.blocks = nnx.Sequential(*[TransformerBlock(c) for _ in range(c.n_layer)])
        s.norm = RMSNorm(c.d_model)

    def __call__(s, x: Array):
        return s.embed.undo(s.norm(s.blocks(s.embed(x))))


# ======================


def cross_entropy(model: nnx.Module, x, y):
    logits = model(x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return loss.mean()


@nnx.jit
def opt_step(opt: nnx.Optimizer, model, x, y):
    grad_fn = nnx.value_and_grad(cross_entropy)
    loss, grad = grad_fn(model, x, y)
    opt.update(model, grad)
    return loss


def generate(s: LLM, idx: Array, max_num=10, temperature=1.0, key=PRNGKey(0)):
    s.eval()
    for _ in range(max_num):
        logits = s(idx[:, -s.c.max_T :])
        logits = logits[:, -1, :] / temperature
        key, k2 = random.split(key)
        next_idx = categorical(k2, logits)
        idx = jnp.concat([idx, next_idx[:, None]], axis=1)
    return idx


def main():
    path = "../docs/docs/2026_SOTA_LLM/papers/01_Attention_Is_All_You_Need.md"
    with open(path) as f:
        txt = f.read()

    bpe = tiktoken.encoding_for_model("gpt2")
    tokens = to_batch(bpe.encode(txt), T=10)[:3]
    x = tokens[:, :-1]
    y = tokens[:, 1:]

    c = LLMConf()
    c.n_vocab = bpe.n_vocab
    c.mask = causal_mask(c.max_T)
    c.roPE = RoPE(c.max_T, c.d_head)

    llm = LLM(c)
    logits = llm(x)
    print(logits.shape)

    lr_sch = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=50,
        decay_steps=500,
        end_value=1e-5,
    )
    opt = nnx.Optimizer(llm, optax.adamw(lr_sch), wrt=nnx.Param)

    for i in range(501):
        loss = opt_step(opt, llm, x, y)
        if i % 10 == 0:
            print(f"step {i}: {loss}")

    idx1 = x[:1, :3]
    idx2 = generate(llm, idx1)
    print({"in": bpe.decode(idx1[0].tolist()), "out": bpe.decode(idx2[0].tolist())})


if __name__ == "__main__":
    main()
