#!/usr/bin/env python
"""Local embedding server: keeps the I-JEPA encoder loaded and adds uploaded images to a data set.

    python serve.py --data ./public/data [--port 8765]

The data folder must have been produced by embed.py (it needs models.pkl, the fitted
projections). New images are embedded, projected into the existing PCA / UMAP / patch
layouts, written into the data folder (thumbs, images, points.json, embeddings.npy,
patches.bin, patch_feats.bin) and returned to the viewer, which adds them in place.

Endpoints (the Vite dev server proxies /api to this server):
    GET  /api/status                      -> {"ok": true, "model": ..., "n": N, ...}
    POST /api/embed?name=<file>&category= -> body = image bytes; returns the new item
    POST /api/flush                       -> rewrites similarity.bin
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageOps

from embed import pick_device

SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. ()")


def safe_name(name: str, fallback: str) -> str:
    name = os.path.basename(name or "").strip()
    name = "".join(c if c in SAFE_CHARS else "_" for c in name)
    return name[:120] or fallback


class Engine:
    def __init__(self, data_dir: Path, device_name: str | None, batch_size: int = 8):
        self.dir = data_dir
        self.lock = threading.Lock()
        self.dirty_similarity = False

        with open(data_dir / "models.pkl", "rb") as f:
            self.fits = pickle.load(f)
        with open(data_dir / "points.json") as f:
            self.data = json.load(f)
        self.E = np.load(data_dir / "embeddings.npy").astype(np.float32)
        self.k = int(self.fits.get("neighbors", 10))
        self.extent = float(self.fits.get("extent", 100.0))
        self.sim_max = int(self.fits.get("sim_max", 3000))
        self.patch_feats_max = int(self.fits.get("patch_feats_max", 2000))
        self.thumb_size = int(self.fits.get("thumb_size", 128))
        self.image_size = int(self.fits.get("image_size", 1024))
        self.pool = self.fits.get("pool", "mean")
        self.model_name = self.fits.get("model", self.data.get("model"))
        self.batch_size = batch_size
        if len(self.E) != len(self.data["items"]):
            sys.exit("embeddings.npy and points.json disagree; re-run embed.py")

        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = pick_device(device_name)
        print(f"loading {self.model_name} on {self.device} ...", flush=True)
        self.proc = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        print(f"ready: {len(self.E)} images in {data_dir}", flush=True)

    # ------------------------------------------------------------------ status

    def status(self) -> dict:
        return {
            "ok": True,
            "model": self.model_name,
            "n": len(self.data["items"]),
            "layouts": list(self.data["layouts"].keys()),
            "patches": bool(self.data.get("patches")),
            "device": str(self.device),
        }

    # ------------------------------------------------------------------- embed

    def add_image(self, raw: bytes, name: str, category: str) -> dict:
        with self.lock:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            t0 = time.time()
            with self.torch.no_grad():
                inputs = self.proc(images=[img], return_tensors="pt").to(self.device)
                tokens = self.model(**inputs).last_hidden_state  # (1, P, D)
                emb = tokens.mean(dim=1) if self.pool == "mean" else tokens.max(dim=1).values
                emb = self.torch.nn.functional.normalize(emb, dim=-1)
            e = emb.float().cpu().numpy()[0]
            toks = tokens.float().cpu().numpy()[0]
            dt = time.time() - t0

            new_id = len(self.data["items"])
            name = safe_name(name, f"upload_{new_id}.jpg")
            file_name = f"uploads/{name}" if category == "" else f"{category}/{name}"

            # layouts
            layouts: dict[str, list[float]] = {}
            pca = self.fits.get("pca")
            if pca is not None:
                p3 = np.zeros(3, dtype=np.float32)
                p3[: pca.n_components_] = pca.transform(e[None])[0]
                layouts["pca"] = ((p3 - self.fits["pca_center"]) * self.fits["pca_scale"]).round(4).tolist()
            if "umap" in self.fits and "umap" in self.data["layouts"]:
                um = np.asarray(self.fits["umap"].transform(e[None]))[0]
                layouts["umap"] = ((um - self.fits["umap_center"]) * self.fits["umap_scale"]).round(4).tolist()

            patches = None
            if self.data.get("patches") and "patch_pca" in self.fits:
                feats = self.fits["patch_pca"].transform(toks).astype(np.float32)  # (P, K)
                xyz = ((feats[:, :3] - self.fits["patch_center"]) * self.fits["patch_scale"]).astype(np.float32)
                layouts["patch"] = xyz.mean(axis=0).round(4).tolist()
                patches = {"xyz": xyz.round(4).tolist()}
                with open(self.dir / "patches.bin", "ab") as f:
                    f.write(xyz.tobytes())
                info = self.data["patches"]
                if info.get("feats") and new_id + 1 <= self.patch_feats_max:
                    with open(self.dir / "patch_feats.bin", "ab") as f:
                        f.write(feats.tobytes())
                    patches["feats"] = feats.round(5).tolist()
                elif info.get("feats"):
                    # the set outgrew the feature file; drop it so the viewer falls back to 3D matching
                    (self.dir / "patch_feats.bin").unlink(missing_ok=True)
                    info.pop("feats", None)
                    info.pop("featDims", None)

            # similarities and neighbors
            sims = self.E @ e if len(self.E) else np.zeros(0, dtype=np.float32)
            self.E = np.vstack([self.E, e[None]]) if len(self.E) else e[None].copy()
            order = np.argsort(-sims)[: self.k]
            neighbors = [{"id": int(j), "sim": round(float(sims[j]), 4)} for j in order]
            updated: dict[str, list] = {}
            for j in range(len(sims)):
                lst = self.data["neighbors"][j]
                s = round(float(sims[j]), 4)
                if len(lst) < self.k or s > lst[-1]["sim"]:
                    lst.append({"id": new_id, "sim": s})
                    lst.sort(key=lambda d: -d["sim"])
                    del lst[self.k :]
                    updated[str(j)] = lst

            # files
            (self.dir / "thumbs").mkdir(exist_ok=True)
            (self.dir / "images").mkdir(exist_ok=True)
            ImageOps.fit(img, (self.thumb_size, self.thumb_size), Image.Resampling.LANCZOS).save(
                self.dir / "thumbs" / f"{new_id}.jpg", quality=88, optimize=True
            )
            preview = img.copy()
            preview.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            preview.save(self.dir / "images" / f"{new_id}.jpg", quality=88, optimize=True)

            item = {"id": new_id, "file": file_name, "thumb": f"thumbs/{new_id}.jpg", "image": f"images/{new_id}.jpg", "category": category}
            self.data["items"].append(item)
            for key, xyz in layouts.items():
                self.data["layouts"][key].append(xyz)
            self.data["neighbors"].append(neighbors)
            self.write_points()
            np.save(self.dir / "embeddings.npy", self.E)
            self.dirty_similarity = True

            return {
                "item": item,
                "layouts": layouts,
                "neighbors": neighbors,
                "updatedNeighbors": updated,
                "patches": patches,
                "sims": sims.round(4).tolist(),
                "seconds": round(dt, 2),
            }

    def write_points(self) -> None:
        tmp = self.dir / "points.json.tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, separators=(",", ":"))
        os.replace(tmp, self.dir / "points.json")

    def flush(self) -> dict:
        with self.lock:
            n = len(self.E)
            if n <= self.sim_max:
                sim = np.clip(self.E @ self.E.T, -1.0, 1.0).astype(np.float32)
                sim.tofile(self.dir / "similarity.bin")
                self.data["similarity"] = "similarity.bin"
            else:
                (self.dir / "similarity.bin").unlink(missing_ok=True)
                self.data.pop("similarity", None)
            self.write_points()
            self.dirty_similarity = False
            return {"ok": True, "n": n, "similarity": self.data.get("similarity")}


class Handler(BaseHTTPRequestHandler):
    engine: Engine

    def log_message(self, fmt: str, *args) -> None:  # quieter log
        sys.stderr.write("%s %s\n" % (self.command, fmt % args))

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send(200, {"ok": True})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self._send(200, self.engine.status())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        url = urlparse(self.path)
        if url.path == "/api/flush":
            self._send(200, self.engine.flush())
            return
        if url.path != "/api/embed":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 64 * 1024 * 1024:
            self._send(400, {"error": "empty or oversized body"})
            return
        q = parse_qs(url.query)
        name = q.get("name", [""])[0]
        category = safe_name(q.get("category", [""])[0], "").strip()
        raw = self.rfile.read(length)
        try:
            result = self.engine.add_image(raw, name, category)
        except Exception as exc:  # noqa: BLE001
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._send(200, result)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("public/data"))
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    if not (args.data / "models.pkl").exists():
        sys.exit(f"{args.data / 'models.pkl'} not found: run embed.py first (it writes the fitted projections)")

    Handler.engine = Engine(args.data, args.device)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"embedding server on http://{args.host}:{args.port}  (ctrl-c to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if Handler.engine.dirty_similarity:
            Handler.engine.flush()


if __name__ == "__main__":
    main()
