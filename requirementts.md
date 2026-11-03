# JEPA Embedding Explorer: Build Spec

## Goal

Build a local web app that runs a set of images through a frozen, pretrained I-JEPA encoder and shows them as points in a 3D space the user can fly through freely. Distance in the space means semantic similarity: images with similar content sit close together.

Deliverables:

1. A Python script that embeds a folder of images and writes a data file plus thumbnails.
2. A browser front end (Three.js) that loads that data and renders the navigable 3D space.
3. A README with run instructions.

No model training is involved. Use a pretrained checkpoint and keep it frozen.

---

## Core requirement: free-flowing 3D navigation

**This is the main feature. Everything else supports it.** The viewport must feel like moving through a space, not looking at a chart. The user should be able to move around the vector space continuously, in any direction, at any time, with no fixed camera, no stepping between preset views, and no page reloads.

Required behavior:

- **Orbit:** left-drag rotates the camera around the current focus point.
- **Pan:** right-drag (or shift + left-drag) translates the camera.
- **Zoom:** scroll wheel and trackpad pinch dolly the camera toward the cursor position. There must be no minimum distance that stops the user from getting inside a cluster, and no maximum so tight that the data can be lost.
- **Fly mode:** WASD moves forward/left/back/right relative to the camera, Q/E move down/up, mouse-look steers, and Shift speeds up. The user can fly through the point cloud and past individual points.
- **Damping and inertia:** all camera motion eases in and out. Releasing a drag lets the camera glide to a stop. No hard stops, no jerky snapping.
- **Click to fly to a point:** clicking an image smoothly animates the camera to it (about 600 ms, eased) and makes it the new orbit center. The user can immediately keep navigating from there.
- **Reset view:** the `R` key and a button animate the camera back to a full overview of the cloud.
- **Never blocked:** tooltips, selection, loading, and panels must never interrupt or lock navigation. Navigation works at all times.
- **Touch support:** one finger orbits, two fingers pan and pinch-zoom.
- **Performance:** hold 60 fps with up to 5,000 points on a normal laptop. Navigation must stay smooth while hovering and selecting.

Do **not** use Plotly, matplotlib, or any static or semi-static chart renderer for the front end. They feel like a chart, not a space. Use Three.js directly.

---

## How it works

For each image:

1. Load it and run it through the frozen I-JEPA encoder to get patch tokens, one vector per patch.
2. Average-pool the patch tokens into **one vector per image**. I-JEPA has no CLS token, and mean pooling matches how its linear-probe evaluation works.
3. L2-normalize the vector.
4. Compute the full pairwise **cosine similarity** matrix. This is the real similarity measure.
5. Project the normalized vectors down to 3D with PCA (default) and optionally UMAP. The 3D position is only for visualization.

The front end loads the 3D coordinates, the thumbnails, and the similarity data, and renders them.

---

## Embedding script (Python)

File: `embed.py`. Usage: `python embed.py --images ./images --out ./public/data`

Behavior:

- Recursively read common image formats (jpg, jpeg, png, webp) from the input folder.
- Use the subfolder name as the image's `category` label when images are organized in folders. Otherwise leave it empty.
- Load the pretrained I-JEPA model with `transformers`. The checkpoint used for reference is `facebook/ijepa_vith14_1k`. **Verify the exact name on the Hugging Face Hub before relying on it**, and make the checkpoint a command-line argument.
- Run inference in batches under `torch.no_grad()` with the model in eval mode. Use the GPU if available, otherwise CPU.
- Mean-pool, normalize, compute cosine similarity, and run PCA to 3 components. Also run UMAP to 3 components when `--umap` is passed (needs at least a few dozen images to behave well).
- Generate square thumbnails (128 px) into `thumbs/`.
- Write `points.json` (schema below).
- Do not write the full 1280-dim embeddings to the front end by default. Write them to `embeddings.npy` for reuse and analysis.

Reference snippet for the embedding step:

```python
import torch
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel

name = "facebook/ijepa_vith14_1k"  # verify on the Hub
proc = AutoImageProcessor.from_pretrained(name)
model = AutoModel.from_pretrained(name).eval().to(device)

with torch.no_grad():
    inputs = proc(batch_of_pil_images, return_tensors="pt").to(device)
    out = model(**inputs)
    emb = out.last_hidden_state.mean(dim=1)              # (B, D) mean-pooled
    emb = torch.nn.functional.normalize(emb, dim=-1)     # L2 normalize

# after collecting all batches into E, shape (N, D)
sim = E @ E.T                                            # cosine similarity
xyz = PCA(n_components=3).fit_transform(E)
```

### Output: `points.json`

```json
{
  "model": "facebook/ijepa_vith14_1k",
  "pooling": "mean",
  "layouts": {
    "pca": [[x, y, z], ...],
    "umap": [[x, y, z], ...]
  },
  "items": [
    { "id": 0, "file": "dog1.jpg", "thumb": "thumbs/0.jpg", "category": "dog" }
  ],
  "neighbors": [[{ "id": 12, "sim": 0.91 }, ...], ...]
}
```

- `layouts.umap` is present only if `--umap` was used.
- Coordinates are scaled so the whole cloud fits in a cube of roughly 100 units, centered on the origin.
- `neighbors[i]` holds the 10 most similar images to image `i`, sorted by cosine similarity. For the full matrix, the front end can also load `similarity.bin` (float32, N x N) when N is small enough, otherwise rely on `neighbors`.

---

## Front end requirements

Stack: Vite, Three.js, TypeScript or plain JavaScript (either is fine). A static site that reads `public/data/`. Run with `npm run dev`.

### Rendering

- Each image is a **camera-facing sprite** showing its thumbnail.
- **Level of detail:** far away, a point renders as a small colored dot. As the camera gets closer, it becomes the thumbnail. Nearby images must be clearly visible, and far ones must not clutter the view.
- Use a texture atlas and instanced rendering (or `THREE.Points` with a shader) so thousands of points stay at 60 fps. Do not create one mesh and one texture per image.
- Color dots by `category` when categories exist, with a legend.
- Dark background, subtle depth fog so distance reads naturally, and faint axes or a grid for orientation that can be toggled off.

### Interaction

- **Hover:** a tooltip shows the thumbnail, filename, and category. The tooltip follows the cursor and never blocks navigation.
- **Click:** flies to the point (see core requirement) and opens a side panel with the full-size image, its filename, and its 10 nearest neighbors with similarity scores. Clicking a neighbor in the panel flies to it.
- **Highlight neighbors:** when an image is selected, draw faint lines to its nearest neighbors and dim everything else.
- **Compare two images:** shift-click a second image to show the exact cosine similarity between the two, plus a line connecting them.
- **Search and filter:** a text box filters by filename or category. Non-matching points dim but stay in place, so spatial context is not lost.
- **Layout switch:** if UMAP data exists, a toggle switches between PCA and UMAP layouts. Points **animate smoothly** from one layout to the other (about 1 second, eased) rather than jumping.
- **Controls help:** a small collapsible overlay lists the controls (orbit, pan, zoom, fly keys, reset).

### UI principles

- The 3D view fills the whole window. Panels float over it and are collapsible.
- The interface should feel fluid. Prefer animated transitions over instant state changes.
- Responsive to window resize. Works on touch devices.

---

## Suggested project structure

```
jepa-explorer/
  embed.py                 # embedding + projection script
  requirements.txt         # torch, transformers, scikit-learn, umap-learn, pillow, numpy
  images/                  # input images, optionally in category subfolders
  public/data/             # generated: points.json, thumbs/, embeddings.npy
  src/
    main.ts                # scene setup, load data
    controls.ts            # orbit + pan + zoom + fly + animated focus
    points.ts              # sprites, atlas, LOD, color by category
    ui.ts                  # tooltip, side panel, search, toggles
  index.html
  package.json
  README.md
```

---

## Implementation order

1. `embed.py` working end to end on a small test folder (20 to 50 images across 3 to 4 categories). Print the cosine similarity matrix and check that same-category images score higher.
2. Front end shell: Three.js scene rendering colored dots at the PCA positions.
3. **Navigation controls first**, before any other UI: orbit, pan, zoom-to-cursor, fly mode, damping, animated focus, reset. Get this feeling right before adding features.
4. Thumbnail sprites with atlas and LOD.
5. Hover tooltip, click to focus, side panel with neighbors.
6. Neighbor lines, two-image comparison, search and filter.
7. UMAP layout and animated layout switch.
8. README, plus a performance check with several thousand synthetic points.

---

## Design notes and pitfalls

- **The 3D view is a distorted picture.** Squashing about 1280 dimensions into 3 loses information. The cosine similarity numbers are the ground truth. The plot is for intuition. Show similarity numbers wherever images are compared, and mention this in the README.
- **Use enough images.** A handful of points shows little structure. 50 to 200 images across a few categories shows clear clusters. PCA is more reliable than UMAP on small sets, which is why PCA is the default.
- **Mean pooling blurs spatial layout.** Two images with similar content arranged differently can land closer than expected. Keep pooling configurable (`--pool mean`) so alternatives can be tried later.
- **Expect semantic similarity, not pixel similarity.** Two photos of different dogs should land closer than a dog and a car even if they look very different. This is the main sanity test.
- **Normalize before comparing.** Always compare L2-normalized vectors so cosine similarity is meaningful and PCA is not dominated by embedding magnitude.
- **Keep the encoder frozen.** No gradients, no fine-tuning, `model.eval()` and `torch.no_grad()`.

---

## Acceptance criteria

- [ ] `python embed.py` runs on a folder of images and produces `points.json`, thumbnails, and `embeddings.npy`.
- [ ] `npm run dev` opens the app and shows all images as points in 3D.
- [ ] The user can orbit, pan, zoom, and fly through the cloud continuously, with smooth damping and no hard stops.
- [ ] Zoom reaches inside dense clusters, and the data cannot be lost by zooming out.
- [ ] Clicking a point smoothly flies the camera to it, and navigation continues from there.
- [ ] Reset returns smoothly to the overview.
- [ ] Hovering, selecting, and loading never block or stutter navigation.
- [ ] Nearby points show thumbnails and distant points show dots.
- [ ] The side panel shows the 10 nearest neighbors with cosine similarity scores, and clicking one flies to it.
- [ ] Two selected images show their exact cosine similarity.
- [ ] Similar images (same category) visibly cluster on the test set.
- [ ] Holds about 60 fps at 5,000 points.
- [ ] Works with touch input.

## Optional extensions

- Compare I-JEPA against DINOv2 on the same images with a model toggle.
- Compare embeddings from different encoder layers.
- Show patch-level PCA-to-RGB maps for the selected image in the side panel.
- Drag-and-drop new images into the browser, embedded by a small local Python server.
