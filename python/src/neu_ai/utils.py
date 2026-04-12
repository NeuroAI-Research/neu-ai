from typing import Dict

import cv2
import jax.numpy as jnp
import numpy as np

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
            ret, frame = vid.read()
            if not ret:
                break
            yield frame
    finally:
        vid.release()


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
