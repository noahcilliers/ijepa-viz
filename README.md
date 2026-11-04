# JEPA Embedding Explorer

Run a folder of images through a frozen, pretrained [I-JEPA](https://huggingface.co/facebook/ijepa_vith14_1k) encoder and fly through the result as a 3D point cloud. Images with similar content sit close together. The viewer is a Three.js app with free orbit, pan, zoom and fly navigation, thumbnail sprites with level of detail, nearest-neighbor inspection and exact cosine similarity between any two images. A second view, **Patches**, shows every image as the cloud of its 256 patch-token embeddings, so two similar images appear as two overlapping clouds rather than two nearby dots.

```
embed.py                 # images -> embeddings, PCA/UMAP layouts, thumbnails, points.json
serve.py                 # local server that embeds images you drop into the viewer
scripts/synthetic.py     # fake data set for trying the viewer without a model
scripts/make_test_images.py
src/
  main.ts                # scene, data loading, selection state, render loop
  controls.ts            # orbit / pan / zoom-to-cursor / fly / animated focus
  points.ts              # instanced sprites, texture atlas, LOD, picking
  ui.ts                  # tooltip, side panel, search, images drawer, help
public/data/             # generated output (git-ignored)
```

## 1. Embed images

Python 3.10+ is required. A GPU is used when available (CUDA or Apple MPS), otherwise the CPU. The ViT-H/14 checkpoint is about 2.5 GB and is downloaded once by `transformers`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put images in `images/`, optionally in one subfolder per category. The subfolder name becomes the point's category and its color in the viewer.

```
images/
  dogs/   dog1.jpg dog2.jpg ...
  cars/   car1.jpg ...
  boats/  ...
```

Then run:

```bash
python embed.py --images ./images --out ./public/data --umap
```

Options:

| flag | default | meaning |
| --- | --- | --- |
| `--model` | `facebook/ijepa_vith14_1k` | any I-JEPA checkpoint on the Hub (`facebook/ijepa_vith14_22k`, `facebook/ijepa_vitg16_22k`, ...) |
| `--pool` | `mean` | how patch tokens are pooled into one vector per image (`mean` or `max`) |
| `--umap` | off | also compute a UMAP layout (needs a few dozen images or more) |
| `--batch-size` | 16 | inference batch size |
| `--device` | auto | `cuda`, `mps` or `cpu` |
| `--neighbors` | 10 | nearest neighbors stored per image |
| `--sim-max` | 3000 | write the full `similarity.bin` matrix only up to this many images |
| `--no-patches` | off | skip the per-patch outputs |
| `--patch-dims` | 16 | PCA dimensions kept per patch for patch-level matching |
| `--patch-feats-max` | 2000 | write `patch_feats.bin` only up to this many images |

What it does for each image: load, run the frozen encoder under `torch.no_grad()`, mean-pool the patch tokens into one 1280-dim vector, L2-normalize. Then it computes the full pairwise cosine similarity matrix, projects to 3D with PCA (and UMAP when asked), scales the cloud into a 100-unit cube, writes 128 px thumbnails and a resized copy of each image for the detail panel, and prints a per-category similarity summary so you can sanity-check that same-category images score higher.

Output in `public/data/`:

- `points.json`: layouts, items (file, thumbnail, category) and each image's 10 nearest neighbors with cosine similarity.
- `thumbs/{id}.jpg`, `images/{id}.jpg`
- `embeddings.npy`: the normalized embeddings, shape `(N, 1280)`, for reuse and analysis.
- `similarity.bin`: float32 `N x N` cosine matrix, written when `N <= --sim-max`.
- `patches.bin`: float32 `(N, 256, 3)`, every patch token of every image projected into one shared 3D PCA space. The image's position in this space (`layouts.patch`) is the centroid of its own patches, because PCA is linear.
- `patch_feats.bin`: float32 `(N, 256, 16)`, the same tokens in 16 PCA dimensions, used in the browser to match patches between two images.
- `models.pkl`: the fitted PCA and UMAP projections, so `serve.py` can place new images into the same layouts.

## 2. Explore

```bash
npm install
npm run dev
```

Open the printed URL. The viewer reads `public/data/points.json` on load.

### Navigation

| action | input |
| --- | --- |
| orbit | left drag, one finger |
| pan | right drag, shift + drag, two fingers |
| zoom toward cursor | scroll wheel, trackpad pinch, two-finger pinch |
| fly | `W A S D` forward/left/back/right, `Q E` down/up, hold `Shift` to go faster |
| look around | `F` toggles fly mode; then left drag turns the camera in place |
| focus an image | click it (the camera flies there in 600 ms and orbits around it from then on) |
| reset to overview | `R` or the Reset button |
| grid on/off | `G` |
| images drawer | `L` |
| deselect | `Esc` |
| filter | `/` focuses the search box |

There is no minimum zoom distance, so you can move inside a dense cluster, and the far plane is large enough that the cloud cannot be lost. All motion is damped: drags glide to a stop and nothing snaps.

### Inspecting images

- **Hover** shows the thumbnail, file name and category.
- **Click** opens the side panel with the image, its category and its 10 nearest neighbors ranked by cosine similarity. Lines connect the selected image to its neighbors and everything else dims. Click a neighbor to fly to it.
- **Shift + click** a second image to see the exact cosine similarity between the two, with a line connecting them. When `similarity.bin` was written the value comes from the full matrix; otherwise it is taken from the stored neighbor lists and shown as n/a for pairs outside them.
- **Search** filters by file name or category. Non-matching points dim but stay in place.
- **Images drawer** (left side, `L` or the layers button) removes images from the scene and brings them back. Each category row is a checkbox; `only` shows that category alone (click again to restore everything); the chevron opens a thumbnail grid where single images can be switched off (double-click one to focus it). Hidden images are not drawn, picked or linked, and a hidden neighbor is marked in the side panel. `All` / `None` act on the whole set. The Display section toggles the grid, neighbor lines and thumbnails.
- **PCA / UMAP / Patches** switches layouts with a 1 s animated transition. `P` toggles the Patches view.

### Patches view

The image-level views place each picture at one point: its 256 patch tokens averaged together. The Patches view skips the averaging. Every patch token of every image is projected into one shared 3D PCA space and drawn as a small dot colored by category, and each image's sprite sits at the centroid of its own patches.

- Select an image and its 256 patches light up in white. Hover one to see its row and column in the 16 x 16 patch grid, outlined on the picture in the panel.
- Shift + click a second image and its patches light up in blue. If the two images are alike, the two clouds occupy the same region and interleave. Each white patch is joined by a line to its most similar blue patch, colored from gray (weak match) to blue (strong match), and the panel shows the mean best-match cosine between the two patch sets alongside the pooled cosine similarity. For sets larger than `--patch-feats-max` the lines fall back to nearest 3D distance.
- Clicking a patch point only inspects it; images are still selected through their sprites, the neighbor list or search.

At 5,000 images this is 1.28 million points in one draw call, which still holds 60 fps.

### Comparing data sets

The viewer reads `public/data/` by default. Add `?data=<folder>` to the URL to read another folder under `public/`, for example `?data=synthetic`.

## 3. Add your own images while the viewer is open

Start the embedding server in a second terminal. It loads the encoder once and keeps it in memory.

```bash
python serve.py --data ./public/data
```

Then click **Add images** in the viewer, or drop image files anywhere on the page. Each image is embedded (well under a second on a GPU or Apple Silicon), projected into the existing PCA, UMAP and patch layouts with the projections saved by `embed.py`, and added to the scene in place: it fades in, the camera flies to it, and its nearest neighbors appear in the panel. Give the batch a category in the popover to color the new points; otherwise they are gray.

Uploads are permanent. The server writes the thumbnail, a preview copy, the embedding, the patch tokens and the updated `points.json` into the data folder, and refreshes `similarity.bin`, so the images are still there after a reload. Existing images' neighbor lists are updated when a new image is closer than their old neighbors.

Two things to know:

- The layouts are fixed when `embed.py` runs. New images are projected into them, which is exact for PCA and patch space and approximate for UMAP, but the layouts are not refitted. After adding many images, re-run `embed.py` on the full folder (the uploaded files are in `public/data/images/`) to get fresh layouts.
- The Vite dev server proxies `/api` to `127.0.0.1:8765` (see `vite.config.ts`). If the server is not running, the popover says so.

## 4. Try it without a model

```bash
python scripts/synthetic.py --n 5000 --out ./public/synthetic
```

generates clustered fake embeddings with matching thumbnails, neighbors, patch clouds and two layouts, which is how the viewer was checked at 5,000 points. Open `http://localhost:5173/?data=synthetic` to view it. `scripts/make_test_images.py` draws a few dozen simple shape images in four category folders for smoke-testing `embed.py`.

## How to read the picture

The 3D positions come from squashing 1280 dimensions into three, so the layout is a distorted sketch of the embedding space. Two points that look close in 3D may not be close in the embedding, and vice versa. The cosine similarity numbers in the panel are the ground truth; the plot is for intuition. PCA is the default because it is stable and reliable on small sets; UMAP shows tighter clusters but needs more images to be meaningful.

Expect semantic similarity rather than pixel similarity: two photos of different dogs should land closer than a dog and a car. Mean pooling blurs the spatial arrangement inside an image, so pictures with similar content laid out differently can land closer than you expect; `--pool` is there so alternatives can be tried.

## Rendering notes

All images are drawn in a single instanced draw call. Far away, a point is a small category-colored dot of constant pixel size. As the camera approaches, it blends into its thumbnail, read from one texture atlas. Thumbnails stream into the atlas with sub-image uploads, so loading never stalls a frame. Picking projects every point on the CPU each time the pointer moves, which costs well under a millisecond at 5,000 points.
