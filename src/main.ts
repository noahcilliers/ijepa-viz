import * as THREE from 'three';
import { CameraRig } from './controls';
import { PointCloud } from './points';
import { UI } from './ui';
import { PatchCloud, matchPatches } from './patches';
import { DATA_BASE, buildCategories, loadFloat32, loadPoints, loadSimilarity, similarityBetween, type PointsData } from './data';

const BG = new THREE.Color('#0b0d12');

async function main(): Promise<void> {
  const canvas = document.getElementById('view') as HTMLCanvasElement;
  canvas.tabIndex = 0;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  renderer.setClearColor(BG, 1);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, (window.innerWidth || 1) / (window.innerHeight || 1), 0.02, 50000);
  const rig = new CameraRig(camera, canvas);

  let data: PointsData | null = null;
  try {
    data = await loadPoints();
  } catch {
    data = null;
  }

  const state = {
    selected: -1,
    compare: -1,
    hovered: -1,
    query: '',
    activeCategories: null as Set<string> | null,
    grid: true,
    sim: null as Float32Array | null,
    patchFeats: null as Float32Array | null,
  };

  const ui = new UI({
    onSearch: (q) => {
      state.query = q.toLowerCase();
      applyDimming();
    },
    onLayout: (name) => setLayout(name),
    onToggleGrid: () => setGrid(!state.grid),
    onToggleFly: () => setFly(!rig.flyMode),
    onReset: () => rig.reset(),
    onSelect: (id) => select(id),
    onClose: () => clearSelection(),
    onCategoryToggle: (name) => {
      if (!state.activeCategories) state.activeCategories = new Set([name]);
      else if (state.activeCategories.has(name)) {
        state.activeCategories.delete(name);
        if (state.activeCategories.size === 0) state.activeCategories = null;
      } else state.activeCategories.add(name);
      ui.setLegendState(state.activeCategories);
      applyDimming();
    },
  });

  if (!data || !data.items.length) {
    ui.showEmptyState();
    animateEmpty();
    return;
  }

  // ---------------------------------------------------------------- scene
  const n = data.items.length;
  const categories = buildCategories(data.items);
  const extent = cloudExtent(data);
  const radius = extent.radius;
  // thumbnails are sized from the typical spacing between neighboring points
  const thumbSize = THREE.MathUtils.clamp(medianNeighborDistance(data.layouts[Object.keys(data.layouts)[0]]) * 1.4, 0.5, 8);

  rig.setHome(extent.center, radius);
  rig.snapHome();
  const home = rig.home.distance;
  const fogNear = home * 0.7;
  const fogFar = home * 2.4;
  scene.fog = new THREE.Fog(BG, fogNear, fogFar);

  const cloud = new PointCloud(renderer, data, categories, { background: BG, thumbSize, fogNear, fogFar });
  scene.add(cloud.mesh);

  // orientation grid + axes
  const gridGroup = new THREE.Group();
  const gridSize = extent.size * 1.4;
  const grid = new THREE.GridHelper(gridSize, 14, 0x1d2230, 0x161a24);
  grid.position.set(extent.center.x, extent.min.y - extent.size * 0.05, extent.center.z);
  (grid.material as THREE.Material).transparent = true;
  (grid.material as THREE.Material).opacity = 0.7;
  gridGroup.add(grid);
  const axes = new THREE.AxesHelper(extent.size * 0.5);
  axes.position.copy(extent.center);
  axes.setColors(new THREE.Color('#3a3f52'), new THREE.Color('#3a3f52'), new THREE.Color('#3a3f52'));
  (axes.material as THREE.Material).transparent = true;
  (axes.material as THREE.Material).opacity = 0.5;
  gridGroup.add(axes);
  scene.add(gridGroup);

  // neighbor + compare lines
  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
  const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.22, fog: true, depthWrite: false });
  const neighborLines = new THREE.LineSegments(lineGeo, lineMat);
  neighborLines.frustumCulled = false;
  scene.add(neighborLines);

  const compareGeo = new THREE.BufferGeometry();
  compareGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
  const compareLine = new THREE.Line(
    compareGeo,
    new THREE.LineBasicMaterial({ color: 0x7aa2ff, transparent: true, opacity: 0.9, fog: true, depthWrite: false }),
  );
  compareLine.visible = false;
  compareLine.frustumCulled = false;
  scene.add(compareLine);

  // patch token cloud (optional)
  let patches: PatchCloud | null = null;
  const matchGeo = new THREE.BufferGeometry();
  matchGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
  matchGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(0), 3));
  const matchLines = new THREE.LineSegments(
    matchGeo,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55, fog: true, depthWrite: false }),
  );
  matchLines.visible = false;
  matchLines.frustumCulled = false;
  scene.add(matchLines);
  if (data.patches) {
    const info = data.patches;
    loadFloat32(DATA_BASE + info.file, n * info.count * 3).then((xyz) => {
      if (!xyz) return;
      patches = new PatchCloud(data!, categories, xyz, {
        background: BG, fogNear, fogFar, homeDistance: home, pixelRatio: renderer.getPixelRatio(),
      });
      scene.add(patches.points);
      ui.setLayouts(Object.keys(data!.layouts), cloud.layout);
      if (cloud.layout === 'patch') patches.setVisible(true);
      refreshPatchHighlights();
      if (info.feats && info.featDims) {
        loadFloat32(DATA_BASE + info.feats, n * info.count * info.featDims).then((f) => {
          state.patchFeats = f;
          updateMatchLines();
        });
      }
    });
  }

  ui.setMeta(`${n.toLocaleString()} images · ${data.model.split('/').pop()} · ${data.pooling} pool`);
  ui.setLayouts(Object.keys(data.layouts).filter((l) => l !== 'patch'), cloud.layout);
  ui.buildLegend(categories);
  ui.setGrid(true);

  // thumbnails stream in without blocking
  cloud.loadThumbnails(
    data.items.map((it) => DATA_BASE + it.thumb),
    (done, total) => ui.setLoading(done, total),
  );
  loadSimilarity(data).then((s) => (state.sim = s));

  if (import.meta.env.DEV) (window as any).__explorer = { camera, rig, cloud, scene, renderer, state, get patches() { return patches; } };

  // -------------------------------------------------------------- helpers
  const _p = new THREE.Vector3();
  const pointAt = (id: number, out = _p) => out.set(cloud.positions[id * 3], cloud.positions[id * 3 + 1], cloud.positions[id * 3 + 2]);

  function matchesFilter(id: number): boolean {
    const it = data!.items[id];
    if (state.activeCategories && !state.activeCategories.has(it.category)) return false;
    if (!state.query) return true;
    return it.file.toLowerCase().includes(state.query) || it.category.toLowerCase().includes(state.query);
  }

  function applyDimming(): void {
    const t = cloud.alphaTarget;
    const filterActive = !!state.query || !!state.activeCategories;
    let matches = 0;
    const neighborSet = new Set<number>();
    if (state.selected >= 0) {
      neighborSet.add(state.selected);
      for (const nb of data!.neighbors[state.selected] ?? []) neighborSet.add(nb.id);
      if (state.compare >= 0) neighborSet.add(state.compare);
    }
    for (let i = 0; i < n; i++) {
      const m = matchesFilter(i);
      if (m) matches++;
      let a = m ? 1 : 0;
      if (state.selected >= 0) a = neighborSet.has(i) ? 1 : Math.min(a, 0.08);
      t[i] = a;
    }
    cloud.touchAlpha();
    ui.setSearchCount(matches, n, filterActive);
  }

  function updateLines(): void {
    if (state.selected < 0) {
      lineGeo.setDrawRange(0, 0);
      neighborLines.visible = false;
    } else {
      const nbs = data!.neighbors[state.selected] ?? [];
      const arr = new Float32Array(nbs.length * 6);
      const a = pointAt(state.selected).clone();
      nbs.forEach((nb, k) => {
        const b = pointAt(nb.id);
        arr.set([a.x, a.y, a.z, b.x, b.y, b.z], k * 6);
      });
      lineGeo.setAttribute('position', new THREE.BufferAttribute(arr, 3));
      lineGeo.setDrawRange(0, nbs.length * 2);
      neighborLines.visible = nbs.length > 0;
    }
    if (state.selected >= 0 && state.compare >= 0) {
      const arr = compareGeo.getAttribute('position') as THREE.BufferAttribute;
      const a = pointAt(state.selected).clone();
      const b = pointAt(state.compare);
      (arr.array as Float32Array).set([a.x, a.y, a.z, b.x, b.y, b.z]);
      arr.needsUpdate = true;
      compareLine.visible = true;
    } else compareLine.visible = false;
  }

  const inPatchMode = () => cloud.layout === 'patch';

  function setLayout(name: string): void {
    if (!data!.layouts[name]) return;
    cloud.setLayout(data!, name);
    ui.setLayouts(Object.keys(data!.layouts).filter((l) => l !== 'patch' || patches), name);
    patches?.setVisible(name === 'patch');
    refreshPatchHighlights();
    updateMatchLines();
    ui.setPatchHint(
      name === 'patch' && state.selected >= 0
        ? `Showing all ${data!.patches!.count} patch embeddings of this image. Hover a point to see where it sits in the picture.`
        : null,
    );
  }

  function refreshPatchHighlights(): void {
    if (!patches) return;
    patches.setHighlight(inPatchMode() ? state.selected : -1, inPatchMode() ? state.compare : -1);
  }

  function updateMatchLines(): void {
    const show = !!patches && inPatchMode() && state.selected >= 0 && state.compare >= 0;
    matchLines.visible = show;
    if (!show) {
      ui.setPatchCompare(null, false);
      return;
    }
    const p = patches!;
    const info = data!.patches!;
    const res = matchPatches(state.selected, state.compare, p.perImage, state.patchFeats, info.featDims ?? 0, p.positions);
    const pos = new Float32Array(p.perImage * 6);
    const col = new Float32Array(p.perImage * 6);
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const lo = new THREE.Color('#3a3f52');
    const hi = new THREE.Color('#7aa2ff');
    const c = new THREE.Color();
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < p.perImage; i++) {
      min = Math.min(min, res.sim[i]);
      max = Math.max(max, res.sim[i]);
    }
    for (let i = 0; i < p.perImage; i++) {
      p.patchCenter(state.selected, i, a);
      p.patchCenter(state.compare, res.match[i], b);
      pos.set([a.x, a.y, a.z, b.x, b.y, b.z], i * 6);
      let t = max > min ? (res.sim[i] - min) / (max - min) : 1;
      if (!res.exact) t = 1 - t; // distance: smaller is better
      c.copy(lo).lerp(hi, t);
      col.set([c.r, c.g, c.b, c.r, c.g, c.b], i * 6);
    }
    matchGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    matchGeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    ui.setPatchCompare(res.mean, res.exact);
  }

  function refreshHighlights(): void {
    cloud.clearHighlights();
    if (state.selected >= 0) {
      for (const nb of data!.neighbors[state.selected] ?? []) cloud.setHighlight(nb.id, 0.5);
      cloud.setHighlight(state.selected, 1);
      if (state.compare >= 0) cloud.setHighlight(state.compare, 2);
    }
  }

  function select(id: number, fly = true): void {
    state.selected = id;
    state.compare = -1;
    ui.hideCompare();
    ui.showPanel(data!.items[id], data!.neighbors[id] ?? [], data!.items);
    ui.setPatchHint(
      inPatchMode() && data!.patches
        ? `Showing all ${data!.patches.count} patch embeddings of this image. Hover a point to see where it sits in the picture.`
        : null,
    );
    refreshHighlights();
    refreshPatchHighlights();
    applyDimming();
    updateLines();
    updateMatchLines();
    if (fly) rig.flyTo(pointAt(id).clone(), inPatchMode() ? patchFocusDistance(id) : thumbSize * 3.6);
  }

  /** Distance that frames an image's whole patch cloud. */
  function patchFocusDistance(id: number): number {
    if (!patches) return thumbSize * 3.6;
    const c = pointAt(id).clone();
    const v = new THREE.Vector3();
    let r = 0;
    for (let k = 0; k < patches.perImage; k++) r = Math.max(r, patches.patchCenter(id, k, v).distanceTo(c));
    const fov = THREE.MathUtils.degToRad(camera.fov);
    return Math.max(thumbSize * 3.6, (r * 1.25) / Math.sin(fov / 2));
  }

  function compareWith(id: number): void {
    if (state.selected < 0 || id === state.selected) {
      select(id);
      return;
    }
    state.compare = id;
    const sim = similarityBetween(data!, state.sim, state.selected, id);
    ui.showCompare(data!.items[state.selected], data!.items[id], sim);
    refreshHighlights();
    refreshPatchHighlights();
    applyDimming();
    updateLines();
    updateMatchLines();
  }

  function clearSelection(): void {
    state.selected = -1;
    state.compare = -1;
    ui.hidePanel();
    refreshHighlights();
    refreshPatchHighlights();
    applyDimming();
    updateLines();
    updateMatchLines();
  }

  function setGrid(on: boolean): void {
    state.grid = on;
    gridGroup.visible = on;
    ui.setGrid(on);
  }

  function setFly(on: boolean): void {
    rig.setFlyMode(on);
    ui.setFly(on);
  }

  // --------------------------------------------------------------- picking
  let pointerX = -1;
  let pointerY = -1;
  let pickDirty = false;
  canvas.addEventListener('pointermove', (e) => {
    if (e.pointerType !== 'mouse') return;
    pointerX = e.clientX;
    pointerY = e.clientY;
    pickDirty = true;
  });
  canvas.addEventListener('pointerleave', () => {
    pointerX = pointerY = -1;
    state.hovered = -1;
    ui.hideTooltip();
  });
  canvas.addEventListener('pointerup', (e) => {
    if (e.button !== 0 || rig.wasDrag()) return;
    if (patches?.pick(e.clientX, e.clientY, camera, window.innerWidth, window.innerHeight)) return; // patch points are inspect-only
    const id = cloud.pick(e.clientX, e.clientY, camera, window.innerWidth, window.innerHeight);
    if (id < 0) return;
    if (e.shiftKey) compareWith(id);
    else select(id);
  });

  function updateHover(): void {
    if (pointerX < 0) return;
    const hit = patches?.pick(pointerX, pointerY, camera, window.innerWidth, window.innerHeight) ?? null;
    if (hit) {
      const [, cols] = patches!.grid;
      const row = Math.floor(hit.patch / cols);
      const col = hit.patch % cols;
      state.hovered = hit.image;
      canvas.style.cursor = 'crosshair';
      ui.showTooltip(data!.items[hit.image], pointerX, pointerY, `patch row ${row + 1}, col ${col + 1}`);
      ui.showPatchBox(hit.slot, row, col, patches!.grid);
      return;
    }
    ui.hidePatchBox();
    const id = cloud.pick(pointerX, pointerY, camera, window.innerWidth, window.innerHeight);
    state.hovered = id;
    canvas.style.cursor = id >= 0 ? 'pointer' : rig.flyMode ? 'crosshair' : '';
    if (id >= 0) ui.showTooltip(data!.items[id], pointerX, pointerY);
    else ui.hideTooltip();
  }

  // -------------------------------------------------------------- keyboard
  window.addEventListener('keydown', (e) => {
    if (rig.isTyping() || e.metaKey || e.ctrlKey || e.altKey) return;
    switch (e.code) {
      case 'KeyR': rig.reset(); break;
      case 'KeyF': setFly(!rig.flyMode); break;
      case 'KeyG': setGrid(!state.grid); break;
      case 'KeyP':
        if (patches) setLayout(inPatchMode() ? 'pca' : 'patch');
        break;
      case 'Escape': clearSelection(); break;
      case 'Slash':
        e.preventDefault();
        (document.getElementById('search') as HTMLInputElement).focus();
        break;
    }
  });

  // ---------------------------------------------------------------- resize
  function resize(): void {
    const w = window.innerWidth;
    const h = window.innerHeight;
    if (!w || !h) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
    rig.setHome(extent.center, radius);
    if (!Number.isFinite(rig.current.distance)) rig.snapHome();
  }
  window.addEventListener('resize', resize);
  resize();

  // ------------------------------------------------------------ render loop
  let last = performance.now();
  let frames = 0;
  let fpsT = last;
  let fps = 0;
  const frame = (now: number) => {
    const dt = (now - last) / 1000;
    last = now;
    rig.update(dt, now);
    let changed = cloud.update(now, dt, camera, window.innerHeight);
    if (patches && patches.update(dt, renderer.getPixelRatio())) changed = true;
    if (changed && (neighborLines.visible || compareLine.visible) && cloud.animatingLayout) updateLines();
    if (pickDirty || rig.moved || changed) {
      pickDirty = false;
      updateHover();
    }
    renderer.render(scene, camera);

    frames++;
    if (now - fpsT >= 500) {
      fps = Math.round((frames * 1000) / (now - fpsT));
      frames = 0;
      fpsT = now;
      ui.setStats(`${fps} fps · ${n.toLocaleString()} pts`);
    }
  };
  const loop = (now: number) => {
    frame(now);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
  if (import.meta.env.DEV) {
    (window as any).__explorer.step = (frames = 1, ms = 16.7) => {
      for (let i = 0; i < frames; i++) frame(last + ms);
    };
    (window as any).__explorer.now = () => last;
    Object.assign((window as any).__explorer, { select, compareWith, setLayout, clearSelection });
  }

  function animateEmpty(): void {
    const tick = () => {
      renderer.render(scene, camera);
      requestAnimationFrame(tick);
    };
    tick();
  }
}

function cloudExtent(data: PointsData) {
  const box = new THREE.Box3();
  for (const layout of Object.values(data.layouts)) for (const p of layout) box.expandByPoint(_tmp.set(p[0], p[1], p[2]));
  if (box.isEmpty()) box.set(new THREE.Vector3(-50, -50, -50), new THREE.Vector3(50, 50, 50));
  const center = box.getCenter(new THREE.Vector3());
  const size = Math.max(1, ...box.getSize(new THREE.Vector3()).toArray());
  const radius = Math.max(1, box.getBoundingSphere(new THREE.Sphere()).radius);
  return { box, center, size, radius, min: box.min };
}
const _tmp = new THREE.Vector3();

/** Median nearest-neighbor distance, estimated from a sample of points. */
function medianNeighborDistance(positions: number[][]): number {
  const n = positions.length;
  if (n < 2) return 10;
  const samples = Math.min(n, 400);
  const stride = n / samples;
  const ds: number[] = [];
  for (let s = 0; s < samples; s++) {
    const i = Math.floor(s * stride);
    const [x, y, z] = positions[i];
    let best = Infinity;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const q = positions[j];
      const d = (q[0] - x) ** 2 + (q[1] - y) ** 2 + (q[2] - z) ** 2;
      if (d < best) best = d;
    }
    ds.push(Math.sqrt(best));
  }
  ds.sort((a, b) => a - b);
  return ds[ds.length >> 1];
}

main();
