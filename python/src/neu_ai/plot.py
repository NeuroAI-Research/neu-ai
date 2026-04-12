import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .utils import D_TYPE


def plot1(data: D_TYPE, id, sort=False, C=2, bins=200, n_std=3):
    if sort:
        data = dict(sorted(data.items()))
    C = min(C, len(data))
    R = math.ceil(len(data) / C)
    plt.figure(figsize=(4 * C, 3 * R))
    i = 0
    for k, v in data.items():
        i += 1
        plt.subplot(R, C, i)
        shape = v.shape if hasattr(v, "shape") else ""
        title = f"{k} {shape}"
        if ".hist" in k:
            v = v.flatten()
            mu, std = v.mean(), v.std()
            bins_k = min(len(v), bins)
            range = [mu - std * n_std, mu + std * n_std]
            plt.hist(v, bins_k, range=range)
            title += f"\n mu={mu:.2g}, std={std:.2g}"
        elif ".img" in k:
            plt.imshow(v, cmap="gray")
        elif isinstance(v, list):
            plt.plot(v[0], v[1])
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                plt.plot(v2, label=k2)
            plt.legend()
        else:
            plt.plot(v)
        plt.title(title)
    plt.tight_layout()
    plt.savefig(id)
    plt.close()


class VideoMaker:
    def __init__(s, path, cols=3, fps=30, gap=10):
        s.path, s.cols, s.fps, s.gap = path, cols, fps, gap
        s.writer = None

    def to_uint8(s, x: np.ndarray):
        min, max = x.min(), x.max()
        norm = (x - min) / (max - min + 1e-5)
        return (norm * 255).astype(np.uint8)

    def _stich_plots(s, plots: D_TYPE):
        imgs = []
        for k, v in plots.items():
            img = s.to_uint8(np.array(v))
            cv2.putText(
                img, k, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,), 1, cv2.LINE_AA
            )
            img = cv2.copyMakeBorder(
                img, 0, s.gap, 0, s.gap, cv2.BORDER_CONSTANT, value=50
            )
            imgs.append(img)
        n = len(imgs)
        C = min(n, s.cols)
        R = np.ceil(n / C)
        h, w = img.shape
        while len(imgs) < (R * C):
            imgs.append(np.full((h, w), 50, dtype=np.uint8))
        rows = [np.hstack(imgs[i : i + C]) for i in range(0, len(imgs), C)]
        return np.vstack(rows)

    def add(s, plots: D_TYPE):
        frame = s._stich_plots(plots)
        if s.writer is None:
            h, w = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            s.writer = cv2.VideoWriter(s.path, fourcc, s.fps, (w, h), isColor=False)
        s.writer.write(frame)

    def release(s):
        s.writer.release()
