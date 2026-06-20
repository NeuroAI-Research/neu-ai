import jax.numpy as jnp
import optax
import tiktoken
from flax import nnx
from jax import Array, random
from jax.nn import gelu, softmax
from jax.random import PRNGKey, categorical


class LLMConf:
    n_vocab: int
    mask: Array

    max_T: int = 1000
    n_head: int = 8
    n_layer: int = 3
    d_model: int = 64
    d_ff: int = 64
    dropout = 0.1
    rngs = nnx.Rngs(0)


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
        s.pos_embed = nnx.Embed(c.max_T, c.d_model, rngs=c.rngs)

    def __call__(s, x: Array):
        # x: (B, T)
        T = x.shape[1]
        token_vec = s.token_embed(x) * jnp.sqrt(s.c.d_model)
        pos_idx = jnp.arange(T)[None, :]
        pos_vec = s.pos_embed(pos_idx)
        return token_vec + pos_vec

    def undo(s, x: Array):
        W: Array = s.token_embed.embedding
        return x @ W.T


class MultiHeadAttention(nnx.Module):
    def __init__(s, c: LLMConf):
        s.c = c
        assert c.d_model % c.n_head == 0
        s.d_head = c.d_model // c.n_head
        s.WQ = nnx.Linear(c.d_model, c.d_model, rngs=c.rngs)
        s.WK = nnx.Linear(c.d_model, c.d_model, rngs=c.rngs)
        s.WV = nnx.Linear(c.d_model, c.d_model, rngs=c.rngs)
        s.WO = nnx.Linear(c.d_model, c.d_model, rngs=c.rngs)

    def split_heads(s, x: Array):
        B, T, _ = x.shape
        x = x.reshape(B, T, s.c.n_head, s.d_head)
        return x.transpose(0, 2, 1, 3)  # (B, n_head, T, d_head)

    def unsplit_heads(s, x: Array):
        B, n_head, T, d_head = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, n_head * d_head)

    def __call__(s, x: Array):
        T = x.shape[1]
        q, k, v = s.WQ(x), s.WK(x), s.WV(x)
        q, k, v = map(s.split_heads, [q, k, v])
        # (B, n_head, T, d_head) @ (B, n_head, d_head, T) -> (B, n_head, T, T)
        qk = jnp.matmul(q, k.transpose(0, 1, 3, 2))
        qk = qk / jnp.sqrt(s.d_head)
        if s.c.mask is not None:
            qk = jnp.where(s.c.mask[:T, :T], -jnp.inf, qk)
        context = jnp.matmul(softmax(qk, axis=-1), v)
        return s.WO(s.unsplit_heads(context))


class TransformerBlock(nnx.Module):
    def __init__(s, c: LLMConf):
        s.attention = MultiHeadAttention(c)
        s.W1 = nnx.Linear(c.d_model, c.d_ff, rngs=c.rngs)
        s.W2 = nnx.Linear(c.d_ff, c.d_model, rngs=c.rngs)
        s.norm1 = nnx.LayerNorm(c.d_model, rngs=c.rngs)
        s.norm2 = nnx.LayerNorm(c.d_model, rngs=c.rngs)
        s.dropout = nnx.Dropout(c.dropout, rngs=c.rngs)

    def __call__(s, x: Array):
        # Pre-LayerNorm: more stable gradients than the original Post-LayerNorm
        x = x + s.dropout(s.attention(s.norm1(x)))
        x = x + s.dropout(s.W2(gelu(s.W1(s.norm2(x)))))
        return x


class LLM(nnx.Module):
    def __init__(s, c: LLMConf):
        s.c = c
        s.embed = TextEmbed(c)
        s.blocks = nnx.Sequential(*[TransformerBlock(c) for _ in range(c.n_layer)])
        s.norm = nnx.LayerNorm(c.d_model, rngs=c.rngs)

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
