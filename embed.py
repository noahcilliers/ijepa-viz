#!/usr/bin/env python
"""Embed a folder of images with a frozen I-JEPA encoder and write data for the explorer.

Usage:
    python embed.py --images ./images --out ./public/data [--umap] [--model facebook/ijepa_vith14_1k]

Outputs (in --out):
    points.json       3D layouts, items, nearest neighbors
    thumbs/{id}.jpg   square thumbnails for the point sprites
    images/{id}.jpg   resized copies for the detail panel
    embeddings.npy    L2-normalized embeddings, shape (N, D)
    similarity.bin    float32 N x N cosine similarity matrix (only when N <= --sim-max)
    patches.bin       float32 (N, P, 3): every patch token projected into a shared 3D PCA space
    patch_feats.bin   float32 (N, P, K): the same tokens in K PCA dims, for patch-level matching
    models.pkl        the fitted PCA / UMAP projections, so serve.py can add images later
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, type=Path, help="input folder (recursive; subfolder = category)")
    p.add_argument("--out", required=True, type=Path, help="output folder, e.g. ./public/data")
    p.add_argument("--model", default="facebook/ijepa_vith14_1k", help="Hugging Face checkpoint name")
    p.add_argument("--pool", default="mean", choices=["mean", "max"], help="how to pool patch tokens")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None, help="cuda | mps | cpu (auto-detected by default)")
    p.add_argument("--umap", action="store_true", help="also compute a UMAP layout")
    p.add_argument("--neighbors", type=int, default=10, help="nearest neighbors stored per image")
    p.add_argument("--thumb-size", type=int, default=128)
    p.add_argument("--image-size", type=int, default=1024, help="max side of the panel preview copy")
    p.add_argument("--sim-max", type=int, default=3000, help="write similarity.bin only when N <= this")
    p.add_argument("--extent", type=float, default=100.0, help="side of the cube the cloud is scaled into")
    p.add_argument("--no-patches", action="store_true", help="skip the per-patch outputs")
    p.add_argument("--patch-dims", type=int, default=16, help="PCA dims kept per patch for patch matching")
    p.add_argument("--patch-feats-max", type=int, default=2000, help="write patch_feats.bin only when N <= this")
    return p.parse_args()


def find_images(root: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            rel = path.relative_to(root)
            category = rel.parts[0] if len(rel.parts) > 1 else ""
            files.append((path, category))
    return files


def pick_device(requested: str | None):
    import torch

    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def embed_images(
    paths: list[Path], model_name: str, pool: str, batch_size: int, device, token_store: Path | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Returns (image embeddings (N, D), patch tokens memmap (N, P, D) float16 or None)."""
    import torch
    from tqdm import tqdm
    from transformers import AutoImageProcessor, AutoModel

    print(f"loading {model_name} on {device} ...", flush=True)
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    for param in model.parameters():  # frozen encoder
        param.requires_grad_(False)

    chunks: list[np.ndarray] = []
    tokens_mm: np.ndarray | None = None
    with torch.no_grad():
        for start in tqdm(range(0, len(paths), batch_size), desc="embedding", unit="batch"):
            batch = [load_rgb(p) for p in paths[start : start + batch_size]]
            inputs = proc(images=batch, return_tensors="pt").to(device)
            out = model(**inputs)
            tokens = out.last_hidden_state  # (B, num_patches, D); I-JEPA has no CLS token
            if token_store is not None:
                if tokens_mm is None:
                    shape = (len(paths), int(tokens.shape[1]), int(tokens.shape[2]))
                    tokens_mm = np.lib.format.open_memmap(token_store, mode="w+", dtype=np.float16, shape=shape)
                tokens_mm[start : start + tokens.shape[0]] = tokens.float().cpu().numpy().astype(np.float16)
            if pool == "mean":
                emb = tokens.mean(dim=1)
            else:
                emb = tokens.max(dim=1).values
            emb = torch.nn.functional.normalize(emb, dim=-1)
            chunks.append(emb.float().cpu().numpy())
    if tokens_mm is not None:
        tokens_mm.flush()
    return np.concatenate(chunks, axis=0), tokens_mm


def cube_transform(xyz: np.ndarray, extent: float) -> tuple[np.ndarray, float]:
    """Center and scale that fit `xyz` into a cube of side `extent` around the origin."""
    center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2.0
    span = float(np.abs(xyz - center).max())
    scale = (extent / 2.0) / span if span > 0 else 1.0
    return center, scale


def fit_to_cube(xyz: np.ndarray, extent: float) -> np.ndarray:
    center, scale = cube_transform(xyz, extent)
    return (xyz - center) * scale


def project_patches(
    tokens: np.ndarray, dims: int, extent: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """PCA over all patch tokens. Returns (xyz (N,P,3), feats (N,P,dims), centroid xyz (N,3), fits).

    PCA is linear, so the centroid of an image's projected patches equals the projection of its
    mean token: the image's position in patch space is the center of its own patch cloud.
    """
    from sklearn.decomposition import PCA

    n, p, d = tokens.shape
    dims = max(3, min(dims, d, n * p - 1))
    rng = np.random.default_rng(seed)
    sample_images = min(n, max(1, 60000 // p))
    idx = np.sort(rng.choice(n, size=sample_images, replace=False))
    sample = np.asarray(tokens[idx], dtype=np.float32).reshape(-1, d)
    pca = PCA(n_components=dims, random_state=seed).fit(sample)

    feats = np.empty((n, p, dims), dtype=np.float32)
    for start in range(0, n, 64):
        chunk = np.asarray(tokens[start : start + 64], dtype=np.float32)
        feats[start : start + 64] = pca.transform(chunk.reshape(-1, d)).reshape(-1, p, dims)

    center, scale = cube_transform(feats[:, :, :3].reshape(-1, 3), extent)
    xyz = ((feats[:, :, :3] - center) * scale).astype(np.float32)
    centroids = xyz.mean(axis=1)
    return xyz, feats, centroids, {"patch_pca": pca, "patch_center": center, "patch_scale": scale}


def layouts_from(E: np.ndarray, extent: float, use_umap: bool) -> tuple[dict[str, list], dict]:
    """Returns (layouts, fits). `fits` holds the fitted projections for projecting new images later."""
    from sklearn.decomposition import PCA

    n = E.shape[0]
    out: dict[str, list] = {}
    fits: dict = {}
    k = min(3, n, E.shape[1])
    pca = np.zeros((n, 3), dtype=np.float32)
    if k >= 1 and n >= 2:
        model = PCA(n_components=k, random_state=0).fit(E)
        pca[:, :k] = model.transform(E)
        fits["pca"] = model
    center, scale = cube_transform(pca, extent)
    fits["pca_center"], fits["pca_scale"] = center, scale
    out["pca"] = ((pca - center) * scale).round(4).tolist()

    if use_umap:
        if n < 15:
            print("warning: UMAP needs at least ~15 images, skipping", file=sys.stderr)
        else:
            import umap

            reducer = umap.UMAP(n_components=3, metric="cosine", n_neighbors=min(15, n - 1), random_state=0)
            um = np.asarray(reducer.fit_transform(E))
            center, scale = cube_transform(um, extent)
            fits["umap"], fits["umap_center"], fits["umap_scale"] = reducer, center, scale
            out["umap"] = ((um - center) * scale).round(4).tolist()
    return out, fits


def nearest_neighbors(sim: np.ndarray, k: int) -> list[list[dict]]:
    n = sim.shape[0]
    k = min(k, n - 1)
    result: list[list[dict]] = []
    if k <= 0:
        return [[] for _ in range(n)]
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf
        idx = np.argpartition(-row, k - 1)[:k]
        idx = idx[np.argsort(-row[idx])]
        result.append([{"id": int(j), "sim": round(float(sim[i, j]), 4)} for j in idx])
    return result


def write_thumbs(paths: list[Path], out: Path, thumb_size: int, image_size: int) -> None:
    from tqdm import tqdm

    thumbs = out / "thumbs"
    images = out / "images"
    thumbs.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(tqdm(paths, desc="thumbnails", unit="img")):
        img = load_rgb(path)
        ImageOps.fit(img, (thumb_size, thumb_size), Image.Resampling.LANCZOS).save(
            thumbs / f"{i}.jpg", quality=88, optimize=True
        )
        preview = img.copy()
        preview.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
        preview.save(images / f"{i}.jpg", quality=88, optimize=True)


def print_similarity_report(sim: np.ndarray, categories: list[str], names: list[str]) -> None:
    n = sim.shape[0]
    if n <= 40:
        np.set_printoptions(precision=2, linewidth=240, suppress=True)
        print("\ncosine similarity matrix:")
        print(sim)
    cats = sorted(set(c for c in categories if c))
    if len(cats) >= 2:
        labels = np.array(categories)
        mask = ~np.eye(n, dtype=bool)
        print("\nmean cosine similarity (rows: category, cols: category):")
        print(" " * 14 + "".join(f"{c[:10]:>12}" for c in cats))
        for a in cats:
            row = []
            for b in cats:
                block = sim[np.ix_(labels == a, labels == b)]
                m = mask[np.ix_(labels == a, labels == b)]
                row.append(float(block[m].mean()) if m.any() else float("nan"))
            print(f"{a[:12]:>14}" + "".join(f"{v:12.3f}" for v in row))
        within = [sim[i, j] for i in range(n) for j in range(n) if i != j and categories[i] == categories[j]]
        between = [sim[i, j] for i in range(n) for j in range(n) if i != j and categories[i] != categories[j]]
        if within and between:
            print(f"\nwithin-category mean {np.mean(within):.3f}  |  between-category mean {np.mean(between):.3f}")


def main() -> None:
    args = parse_args()
    t0 = time.time()

    found = find_images(args.images)
    if not found:
        sys.exit(f"no images found under {args.images}")
    paths = [p for p, _ in found]
    categories = [c for _, c in found]
    print(f"found {len(paths)} images, {len(set(c for c in categories if c))} categories")

    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    token_store = None if args.no_patches else args.out / "_tokens.npy"
    E, tokens = embed_images(paths, args.model, args.pool, args.batch_size, device, token_store)
    print(f"embeddings: {E.shape}")
    np.save(args.out / "embeddings.npy", E)

    sim = (E @ E.T).astype(np.float32)
    np.clip(sim, -1.0, 1.0, out=sim)
    layouts, fits = layouts_from(E, args.extent, args.umap)
    neighbors = nearest_neighbors(sim, args.neighbors)

    patch_info = None
    if tokens is not None:
        print(f"projecting {tokens.shape[0] * tokens.shape[1]} patch tokens ...", flush=True)
        xyz, feats, centroids, patch_fits = project_patches(tokens, args.patch_dims, args.extent)
        fits.update(patch_fits)
        layouts["patch"] = centroids.round(4).tolist()
        xyz.tofile(args.out / "patches.bin")
        side = int(round(tokens.shape[1] ** 0.5))
        patch_info = {"file": "patches.bin", "count": int(tokens.shape[1]), "grid": [side, side]}
        if len(paths) <= args.patch_feats_max:
            feats.tofile(args.out / "patch_feats.bin")
            patch_info["feats"] = "patch_feats.bin"
            patch_info["featDims"] = int(feats.shape[2])
        del tokens
        token_store.unlink(missing_ok=True)

    write_thumbs(paths, args.out, args.thumb_size, args.image_size)

    items = [
        {
            "id": i,
            "file": str(p.relative_to(args.images)),
            "thumb": f"thumbs/{i}.jpg",
            "image": f"images/{i}.jpg",
            "category": categories[i],
        }
        for i, p in enumerate(paths)
    ]
    data = {
        "model": args.model,
        "pooling": args.pool,
        "dim": int(E.shape[1]),
        "layouts": layouts,
        "items": items,
        "neighbors": neighbors,
    }
    if len(paths) <= args.sim_max:
        sim.tofile(args.out / "similarity.bin")
        data["similarity"] = "similarity.bin"
    if patch_info:
        data["patches"] = patch_info

    with open(args.out / "points.json", "w") as f:
        json.dump(data, f, separators=(",", ":"))

    import pickle

    fits.update({"model": args.model, "pool": args.pool, "extent": args.extent, "neighbors": args.neighbors,
                 "sim_max": args.sim_max, "patch_feats_max": args.patch_feats_max,
                 "thumb_size": args.thumb_size, "image_size": args.image_size})
    with open(args.out / "models.pkl", "wb") as f:
        pickle.dump(fits, f)

    print_similarity_report(sim, categories, [p.name for p in paths])
    print(f"\nwrote {args.out / 'points.json'} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
