#!/usr/bin/env python
"""Generate a synthetic point cloud in the explorer's data format, for testing the viewer
without running a model.

    python scripts/synthetic.py --n 5000 --out ./public/data

Each category is a cluster in a 64-dim embedding space; cosine similarity, PCA and the
secondary layout are computed from those vectors so neighbors and layouts are consistent.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CATEGORIES = ["dog", "cat", "car", "boat", "flower", "building", "bird", "food"]
COLORS = [(214, 120, 90), (120, 160, 214), (110, 190, 150), (220, 190, 110), (190, 140, 220), (120, 200, 220), (230, 130, 170), (170, 200, 120)]


def fit_to_cube(xyz: np.ndarray, extent: float) -> np.ndarray:
    xyz = xyz - (xyz.min(axis=0) + xyz.max(axis=0)) / 2.0
    span = float(np.abs(xyz).max())
    return xyz / span * (extent / 2.0) if span > 0 else xyz


def pca3(E: np.ndarray) -> np.ndarray:
    X = E - E.mean(axis=0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return X @ vt[:3].T


def pca_basis(E: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    mean = E.mean(axis=0)
    _, _, vt = np.linalg.svd(E - mean, full_matrices=False)
    return mean, vt[:k].T


def make_thumb(path: Path, cat: int, rng: random.Random, size: int = 128) -> None:
    base = COLORS[cat % len(COLORS)]
    img = Image.new("RGB", (size, size), tuple(max(0, c - 70) for c in base))
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(2, 5)):
        x, y = rng.randint(0, size), rng.randint(0, size)
        r = rng.randint(size // 8, size // 2)
        jitter = tuple(min(255, max(0, c + rng.randint(-40, 40))) for c in base)
        shape = cat % 3
        if shape == 0:
            d.ellipse([x - r, y - r, x + r, y + r], fill=jitter)
        elif shape == 1:
            d.rectangle([x - r, y - r, x + r, y + r], fill=jitter)
        else:
            d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=jitter)
    img.save(path, quality=85)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--categories", type=int, default=6)
    p.add_argument("--out", type=Path, default=Path("public/data"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sim-max", type=int, default=3000)
    p.add_argument("--patches", type=int, default=256, help="patch tokens per image (0 to skip)")
    p.add_argument("--patch-feats-max", type=int, default=2000)
    args = p.parse_args()

    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    n, k, dim = args.n, min(args.categories, len(CATEGORIES)), 64

    cats = nrng.integers(0, k, size=n)
    centers = nrng.normal(size=(k, dim)) * 2.5
    E = centers[cats] + nrng.normal(size=(n, dim))
    # a few sub-clusters per category so there is structure to fly into
    sub = nrng.integers(0, 3, size=n)
    E += nrng.normal(size=(k, 3, dim))[cats, sub] * 1.2
    E /= np.linalg.norm(E, axis=1, keepdims=True)

    sim = (E @ E.T).astype(np.float32)
    layouts = {"pca": fit_to_cube(pca3(E), 100).round(3).tolist()}
    # a second layout: PCA of a random nonlinear feature map, stands in for UMAP
    F = np.tanh(E @ nrng.normal(size=(dim, dim)) * 1.5)
    layouts["umap"] = fit_to_cube(pca3(F), 100).round(3).tolist()

    patch_info = None
    if args.patches > 0:
        # each image's patches scatter around its own vector, with a few shared "part" directions
        P, K = args.patches, 16
        parts = nrng.normal(size=(k, 6, dim)) * 1.6
        raw = E * 4.0
        mean, basis = pca_basis(raw, K)
        feats = np.empty((n, P, K), dtype=np.float32)
        for i in range(n):
            part = nrng.integers(0, 6, size=P)
            toks = raw[i] + parts[cats[i], part] + nrng.normal(size=(P, dim)) * 0.8
            feats[i] = (toks - mean) @ basis
        flat = feats[:, :, :3].reshape(-1, 3)
        center = (flat.min(axis=0) + flat.max(axis=0)) / 2
        scale = 50.0 / float(np.abs(flat - center).max())
        xyz = ((feats[:, :, :3] - center) * scale).astype(np.float32)
        layouts["patch"] = xyz.mean(axis=1).round(3).tolist()
        side = int(round(P ** 0.5))
        patch_info = {"file": "patches.bin", "count": P, "grid": [side, side]}

    kk = min(10, n - 1)
    neighbors = []
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf
        idx = np.argpartition(-row, kk - 1)[:kk]
        idx = idx[np.argsort(-row[idx])]
        neighbors.append([{"id": int(j), "sim": round(float(sim[i, j]), 4)} for j in idx])

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "thumbs").mkdir(parents=True)
    items = []
    for i in range(n):
        c = int(cats[i])
        make_thumb(out / "thumbs" / f"{i}.jpg", c, rng)
        items.append({"id": i, "file": f"{CATEGORIES[c]}/{CATEGORIES[c]}_{i:04d}.jpg", "thumb": f"thumbs/{i}.jpg", "category": CATEGORIES[c]})

    data = {"model": "synthetic", "pooling": "mean", "dim": dim, "layouts": layouts, "items": items, "neighbors": neighbors}
    if n <= args.sim_max:
        sim.tofile(out / "similarity.bin")
        data["similarity"] = "similarity.bin"
    if patch_info:
        xyz.tofile(out / "patches.bin")
        if n <= args.patch_feats_max:
            feats.tofile(out / "patch_feats.bin")
            patch_info["feats"] = "patch_feats.bin"
            patch_info["featDims"] = feats.shape[2]
        data["patches"] = patch_info
    np.save(out / "embeddings.npy", E.astype(np.float32))
    with open(out / "points.json", "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {n} synthetic points ({k} categories) to {out}; cube side {math.ceil(100)}")


if __name__ == "__main__":
    main()
