#!/usr/bin/env python
"""Draw a small set of synthetic test images in category subfolders, for smoke-testing embed.py.

    python scripts/make_test_images.py --out ./images --per-category 10
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SCENES = {
    "circles": lambda d, rng, s: [d.ellipse(_box(rng, s), fill=_c(rng, (220, 90, 70))) for _ in range(rng.randint(3, 7))],
    "stripes": lambda d, rng, s: [d.rectangle([0, y, s, y + rng.randint(8, 30)], fill=_c(rng, (70, 120, 220))) for y in range(0, s, 40)],
    "grid": lambda d, rng, s: [d.rectangle([x, y, x + 24, y + 24], fill=_c(rng, (90, 200, 120))) for x in range(10, s, 48) for y in range(10, s, 48)],
    "blobs": lambda d, rng, s: [d.polygon([(rng.randint(0, s), rng.randint(0, s)) for _ in range(6)], fill=_c(rng, (230, 200, 90))) for _ in range(rng.randint(2, 4))],
}


def _box(rng: random.Random, s: int) -> list[int]:
    x, y, r = rng.randint(0, s), rng.randint(0, s), rng.randint(s // 10, s // 3)
    return [x - r, y - r, x + r, y + r]


def _c(rng: random.Random, base: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(min(255, max(0, c + rng.randint(-50, 50))) for c in base)  # type: ignore[return-value]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("images"))
    p.add_argument("--per-category", type=int, default=10)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)
    for name, draw in SCENES.items():
        folder = args.out / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_category):
            bg = tuple(rng.randint(10, 60) for _ in range(3))
            img = Image.new("RGB", (args.size, args.size), bg)
            draw(ImageDraw.Draw(img), rng, args.size)
            if rng.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 2)))
            img.save(folder / f"{name}_{i:02d}.jpg", quality=90)
    print(f"wrote {len(SCENES) * args.per_category} images to {args.out}")


if __name__ == "__main__":
    main()
