import json
from typing import Dict, List

import cv2
import jax.numpy as jnp
import numba
import numpy as np
import pypdfium2

D_TYPE = Dict[str, np.ndarray]


def postfix(x: Dict, txt):
    return {k + txt: v for k, v in x.items()}


def frame_to_jax(frame, size=(112, 112), c=cv2.COLOR_BGR2GRAY):
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return jnp.array(cv2.cvtColor(frame, c), dtype=jnp.float32)


def read_video(path):
    vid = cv2.VideoCapture(path)
    try:
        while vid.isOpened():
            success, frame = vid.read()
            if not success:
                break
            yield frame
    finally:
        vid.release()


def read_pdf(path, indices=None, scale=2):
    pdf = pypdfium2.PdfDocument(path)
    if indices is None:
        indices = range(len(pdf))
    pages = []
    for i in indices:
        page = pdf[i]
        img = page.render(scale).to_numpy() / 255.0
        txt = page.get_textpage().get_text_bounded()
        pages.append((img, txt))
    return pages


def shape(x):
    if isinstance(x, dict):
        return {k: shape(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [shape(v) for v in x]
    try:
        return x.shape
    except Exception:
        return ""


def gaussian(x, mu, sig):
    return jnp.exp(-((x - mu) ** 2) / (2 * sig**2))


def SMA(x, T):
    return jnp.convolve(x, jnp.ones(T), "valid") / T


def load_json(path):
    with open(path) as f:
        return json.load(f)


@numba.njit
def discount_cumsum(x: np.ndarray, d):
    # [x0, x1, x2] -> [x0 + d * x1 + d^2 * x2, x1 + d * x2, x2]
    y = x.copy()
    for i in range(len(x) - 2, -1, -1):
        y[i] += d * y[i + 1]
    return y


class Tree:
    def get_id(s, x: Dict):
        pass

    def __init__(s, data: List[Dict]):
        s.map: Dict[str, Dict] = {}
        for x in data:
            id, pid, name = s.get_id(x)
            s.map[id] = {"c": {}, "x": x, "id": id, "pid": pid, "name": name}
        assert len(data) == len(s.map), "id must be unique"
        s.root = {"c": {}}
        for id, x in s.map.items():
            p = s.map.get(x["pid"], s.root)
            p["c"][id] = x

    def save(s, path, lim=None, max_lvl=None):
        lines = []

        def add(root: Dict[str, Dict], level):
            if max_lvl and level > max_lvl:
                return
            space = " " * 4 * level
            for id, c in root["c"].items():
                if lim and len(lines) >= lim:
                    return
                lines.append(f"{space}{c['name']}{c.get('info', '')}")
                add(c, level + 1)

        add(s.root, 0)
        with open(path, "w+") as f:
            f.write("\n".join(lines))


class Memory:
    def __init__(s, size):
        s.size = size

    @property
    def full(s):
        return s.cnt % s.size == 0

    def save(s, values: List[np.ndarray]):
        if not hasattr(s, "mem"):
            s.mem: List[np.ndarray] = []
            s.cnt = 0
            for v in values:
                v_shape = (1,) if np.ndim(v) == 0 else v.shape
                s.mem.append(np.zeros((s.size, *v_shape), dtype=np.float32))
        ptr = s.cnt % s.size
        for i, v in enumerate(values):
            s.mem[i][ptr] = v
        s.cnt += 1

    def sample(s, num):
        if num > s.cnt:
            return
        max_idx = min(s.cnt, s.size)
        idx = np.random.choice(max_idx, size=num, replace=False)
        return [v[idx] for v in s.mem]
