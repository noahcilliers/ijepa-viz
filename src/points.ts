import * as THREE from 'three';
import type { Category, PointsData } from './data';

/**
 * All images in one instanced draw call. Far away a point is a colored dot of fixed
 * pixel size; up close it is its thumbnail, read from a single texture atlas that is
 * filled progressively with sub-image uploads so loading never stalls a frame.
 */

const vertexShader = /* glsl */ `
  attribute vec3 aPos0;
  attribute vec3 aPos1;
  attribute vec3 aColor;
  attribute float aAlpha;
  attribute float aHighlight;
  attribute float aIndex;
  attribute float aHidden;

  uniform float uMix;
  uniform float uThumbSize;
  uniform float uDotPx;
  uniform float uViewportHeight;
  uniform float uFovScale;
  uniform float uLodLo;
  uniform float uLodHi;
  uniform vec2 uGrid;

  varying vec2 vUv;
  varying vec2 vCell;
  varying vec3 vColor;
  varying float vAlpha;
  varying float vHighlight;
  varying float vLod;
  varying float vDist;
  varying float vPx;

  void main() {
    // hidden images fade out through aAlpha, then collapse off screen so they cost nothing
    if (aHidden > 0.5 && aAlpha < 0.02) {
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      return;
    }
    vec3 center = mix(aPos0, aPos1, uMix);
    vec4 mv = modelViewMatrix * vec4(center, 1.0);
    float dist = max(1e-4, -mv.z);
    float thumbPx = uThumbSize * uViewportHeight * 0.5 * uFovScale / dist;
    float lod = smoothstep(uLodLo, uLodHi, thumbPx);
    float boost = 1.0 + 0.6 * step(0.5, aHighlight) + 0.3 * step(0.2, aHighlight);
    float dotWorld = uDotPx * boost * mix(0.75, 1.0, aAlpha) * dist * 2.0 / (uViewportHeight * uFovScale);
    float size = mix(dotWorld, uThumbSize, lod);
    mv.xy += position.xy * size;
    gl_Position = projectionMatrix * mv;

    vUv = uv;
    vCell = vec2(mod(aIndex, uGrid.x), floor(aIndex / uGrid.x));
    vColor = aColor;
    vAlpha = aAlpha;
    vHighlight = aHighlight;
    vLod = lod;
    vDist = dist;
    vPx = size * uViewportHeight * 0.5 * uFovScale / dist;
  }
`;

const fragmentShader = /* glsl */ `
  precision highp float;

  uniform sampler2D uAtlas;
  uniform vec2 uGrid;
  uniform vec3 uBg;
  uniform float uFogNear;
  uniform float uFogFar;
  uniform float uFogMax;
  uniform vec3 uSelectColor;
  uniform vec3 uCompareColor;

  varying vec2 vUv;
  varying vec2 vCell;
  varying vec3 vColor;
  varying float vAlpha;
  varying float vHighlight;
  varying float vLod;
  varying float vDist;
  varying float vPx;

  void main() {
    vec2 p = vUv - 0.5;
    float r = length(p);
    float aa = 1.5 / max(vPx, 1.0);

    // dot: anti-aliased disc
    float disc = 1.0 - smoothstep(0.5 - aa, 0.5, r);
    // thumbnail: rounded square
    vec2 q = abs(p) - 0.5 + 0.06;
    float rd = length(max(q, 0.0)) - 0.06;
    float square = 1.0 - smoothstep(-aa, 0.0, rd);

    float shape = mix(disc, square, smoothstep(0.35, 0.65, vLod));
    if (shape < 0.02) discard;

    vec2 atlasUv = (vCell + vec2(vUv.x, 1.0 - vUv.y)) / uGrid;
    vec3 tex = texture2D(uAtlas, atlasUv).rgb;

    // thin category-colored frame on thumbnails, brighter for highlighted ones
    float frame = smoothstep(-aa * 2.5 - 0.035, -aa * 2.5 - 0.035 + aa, rd);
    vec3 frameColor = vColor;
    if (vHighlight > 1.5) frameColor = uCompareColor;
    else if (vHighlight > 0.9) frameColor = uSelectColor;
    float frameStrength = vHighlight > 0.9 ? 1.0 : (vHighlight > 0.2 ? 0.8 : 0.45);
    vec3 thumb = mix(tex, frameColor, frame * frameStrength);

    vec3 dot = vColor;
    if (vHighlight > 0.9) {
      float ring = smoothstep(0.30, 0.30 + aa, r);
      dot = mix(dot, vHighlight > 1.5 ? uCompareColor : uSelectColor, ring);
    }

    vec3 col = mix(dot, thumb, smoothstep(0.35, 0.65, vLod));

    // dimming: fade toward the background rather than using alpha, so depth stays correct
    float dim = mix(0.03, 1.0, vAlpha * vAlpha);
    col = mix(uBg, col, dim);

    // subtle linear depth fog
    float fog = clamp((vDist - uFogNear) / (uFogFar - uFogNear), 0.0, 1.0) * uFogMax;
    col = mix(col, uBg, fog);

    gl_FragColor = vec4(col, shape);
    #include <colorspace_fragment>
  }
`;

export interface PointCloudOptions {
  background: THREE.Color;
  thumbSize: number;
  fogNear: number;
  fogFar: number;
}

export class PointCloud {
  readonly mesh: THREE.Mesh;
  /** Number of live points. */
  count: number;
  /** Slots allocated up front so images can be appended without rebuilding. */
  readonly capacity: number;
  readonly material: THREE.ShaderMaterial;
  /** Interpolated positions in world space, kept in sync with the shader. */
  readonly positions: Float32Array;
  readonly thumbSize: number;

  private readonly geometry: THREE.InstancedBufferGeometry;
  private readonly pos0: THREE.InstancedBufferAttribute;
  private readonly pos1: THREE.InstancedBufferAttribute;
  private readonly alphaAttr: THREE.InstancedBufferAttribute;
  private readonly highlightAttr: THREE.InstancedBufferAttribute;
  private readonly colorAttr: THREE.InstancedBufferAttribute;
  private readonly hiddenAttr: THREE.InstancedBufferAttribute;
  /** 1 for images removed from the scene (not drawn, not pickable). */
  readonly hidden: Uint8Array;
  readonly alphaTarget: Float32Array;
  private alphaSettled = true;
  private layoutAnim: { start: number; duration: number } | null = null;
  private lastNow = performance.now();
  private layoutFrom: Float32Array;
  private layoutTo: Float32Array;
  private currentLayout: string;

  private readonly renderer: THREE.WebGLRenderer;
  readonly atlas: THREE.DataTexture;
  readonly cell: number;
  readonly cols: number;
  loaded = 0;
  private aborted = false;

  private readonly _m = new THREE.Matrix4();
  private readonly _v = new THREE.Vector4();

  constructor(renderer: THREE.WebGLRenderer, data: PointsData, categories: Map<string, Category>, opts: PointCloudOptions) {
    this.renderer = renderer;
    const n = (this.count = data.items.length);
    const cap = (this.capacity = n + Math.max(256, n));
    this.thumbSize = opts.thumbSize;
    const layoutNames = Object.keys(data.layouts);
    this.currentLayout = layoutNames.includes('pca') ? 'pca' : layoutNames[0];

    // --- atlas sizing: biggest cell that fits every thumbnail into one texture
    const maxTex = Math.min(renderer.capabilities.maxTextureSize, 8192);
    const cols = Math.ceil(Math.sqrt(cap));
    let cell = 128;
    while (cell > 16 && cols * cell > maxTex) cell /= 2;
    this.cell = cell;
    this.cols = cols;
    const rows = Math.ceil(cap / cols);
    const atlas = new THREE.DataTexture(null, cols * cell, rows * cell, THREE.RGBAFormat, THREE.UnsignedByteType);
    atlas.colorSpace = THREE.SRGBColorSpace;
    atlas.minFilter = THREE.LinearFilter;
    atlas.magFilter = THREE.LinearFilter;
    atlas.generateMipmaps = false;
    atlas.flipY = false;
    atlas.source.dataReady = false; // allocate on the GPU, fill with sub-uploads
    atlas.needsUpdate = true;
    this.atlas = atlas;

    // --- geometry: one quad, instanced
    const quad = new THREE.PlaneGeometry(1, 1);
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.index = quad.index;
    geometry.setAttribute('position', quad.getAttribute('position'));
    geometry.setAttribute('uv', quad.getAttribute('uv'));
    geometry.instanceCount = n;

    const flat = (name: string) => {
      const arr = new Float32Array(cap * 3);
      const src = data.layouts[name];
      for (let i = 0; i < n; i++) {
        arr[i * 3] = src[i][0];
        arr[i * 3 + 1] = src[i][1];
        arr[i * 3 + 2] = src[i][2];
      }
      return arr;
    };
    this.layoutFrom = flat(this.currentLayout);
    this.layoutTo = this.layoutFrom;
    this.positions = new Float32Array(this.layoutFrom);

    const colors = new Float32Array(cap * 3);
    const index = new Float32Array(cap);
    for (let i = 0; i < cap; i++) {
      index[i] = i;
      if (i >= n) continue;
      const c = categories.get(data.items[i].category ?? '')!.color;
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }
    this.alphaTarget = new Float32Array(cap).fill(1);
    this.hidden = new Uint8Array(cap);

    this.pos0 = new THREE.InstancedBufferAttribute(new Float32Array(this.layoutFrom), 3);
    this.pos1 = new THREE.InstancedBufferAttribute(new Float32Array(this.layoutFrom), 3);
    this.alphaAttr = new THREE.InstancedBufferAttribute(new Float32Array(cap).fill(1), 1);
    this.highlightAttr = new THREE.InstancedBufferAttribute(new Float32Array(cap), 1);
    this.hiddenAttr = new THREE.InstancedBufferAttribute(new Float32Array(cap), 1);
    this.hiddenAttr.setUsage(THREE.DynamicDrawUsage);
    this.colorAttr = new THREE.InstancedBufferAttribute(colors, 3);
    this.colorAttr.setUsage(THREE.DynamicDrawUsage);
    this.pos0.setUsage(THREE.DynamicDrawUsage);
    this.pos1.setUsage(THREE.DynamicDrawUsage);
    this.alphaAttr.setUsage(THREE.DynamicDrawUsage);
    this.highlightAttr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('aPos0', this.pos0);
    geometry.setAttribute('aPos1', this.pos1);
    geometry.setAttribute('aColor', this.colorAttr);
    geometry.setAttribute('aAlpha', this.alphaAttr);
    geometry.setAttribute('aHighlight', this.highlightAttr);
    geometry.setAttribute('aHidden', this.hiddenAttr);
    geometry.setAttribute('aIndex', new THREE.InstancedBufferAttribute(index, 1));
    this.geometry = geometry;

    this.material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        uMix: { value: 0 },
        uThumbSize: { value: opts.thumbSize },
        uDotPx: { value: 5 },
        uViewportHeight: { value: 1 },
        uFovScale: { value: 1 },
        uLodLo: { value: 9 },
        uLodHi: { value: 22 },
        uGrid: { value: new THREE.Vector2(cols, rows) },
        uAtlas: { value: atlas },
        uBg: { value: opts.background },
        uFogNear: { value: opts.fogNear },
        uFogFar: { value: opts.fogFar },
        uFogMax: { value: 0.75 },
        uSelectColor: { value: new THREE.Color('#ffffff') },
        uCompareColor: { value: new THREE.Color('#7aa2ff') },
      },
      transparent: true,
      depthWrite: true,
      depthTest: true,
    });

    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.frustumCulled = false;
  }

  get layout(): string {
    return this.currentLayout;
  }

  /** Animate every point from where it is now to its position in `name`. */
  setLayout(data: PointsData, name: string, duration = 1000): void {
    if (!data.layouts[name] || name === this.currentLayout) return;
    this.currentLayout = name;
    this.layoutFrom = new Float32Array(this.positions);
    const src = data.layouts[name];
    const to = new Float32Array(this.capacity * 3);
    for (let i = 0; i < this.count; i++) {
      to[i * 3] = src[i][0];
      to[i * 3 + 1] = src[i][1];
      to[i * 3 + 2] = src[i][2];
    }
    this.layoutTo = to;
    (this.pos0.array as Float32Array).set(this.layoutFrom);
    (this.pos1.array as Float32Array).set(to);
    this.pos0.needsUpdate = true;
    this.pos1.needsUpdate = true;
    this.material.uniforms.uMix.value = 0;
    this.layoutAnim = { start: this.lastNow, duration };
  }

  get animatingLayout(): boolean {
    return this.layoutAnim !== null;
  }

  setHighlight(id: number, level: number): void {
    const arr = this.highlightAttr.array as Float32Array;
    if (arr[id] !== level) {
      arr[id] = level;
      this.highlightAttr.needsUpdate = true;
    }
  }

  clearHighlights(): void {
    (this.highlightAttr.array as Float32Array).fill(0);
    this.highlightAttr.needsUpdate = true;
  }

  /** Remove an image from the scene (it fades out) or bring it back (it fades in). */
  setHidden(id: number, on: boolean): void {
    const v = on ? 1 : 0;
    if (this.hidden[id] === v) return;
    this.hidden[id] = v;
    (this.hiddenAttr.array as Float32Array)[id] = v;
    this.hiddenAttr.needsUpdate = true;
  }

  /** Dots only (false) or thumbnails up close (true). */
  setThumbnails(on: boolean): void {
    this.material.uniforms.uLodLo.value = on ? 9 : 1e9;
    this.material.uniforms.uLodHi.value = on ? 22 : 2e9;
  }

  /** Call after editing `alphaTarget`. */
  touchAlpha(): void {
    this.alphaSettled = false;
  }

  /** Returns true if anything visual changed this frame. */
  update(now: number, dt: number, camera: THREE.PerspectiveCamera, viewportHeight: number): boolean {
    let changed = false;
    this.lastNow = now;
    const u = this.material.uniforms;
    u.uViewportHeight.value = viewportHeight;
    u.uFovScale.value = 1 / Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);

    if (this.layoutAnim) {
      const t = THREE.MathUtils.clamp((now - this.layoutAnim.start) / this.layoutAnim.duration, 0, 1);
      const k = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
      u.uMix.value = k;
      const a = this.layoutFrom;
      const b = this.layoutTo;
      const p = this.positions;
      for (let i = 0; i < p.length; i++) p[i] = a[i] + (b[i] - a[i]) * k;
      if (t >= 1) this.layoutAnim = null;
      changed = true;
    }

    if (!this.alphaSettled) {
      const arr = this.alphaAttr.array as Float32Array;
      const tgt = this.alphaTarget;
      const a = 1 - Math.exp(-12 * dt);
      let maxDiff = 0;
      for (let i = 0; i < arr.length; i++) {
        const d = tgt[i] - arr[i];
        arr[i] += d * a;
        if (Math.abs(d) > maxDiff) maxDiff = Math.abs(d);
      }
      if (maxDiff < 0.002) {
        arr.set(tgt);
        this.alphaSettled = true;
      }
      this.alphaAttr.needsUpdate = true;
      changed = true;
    }
    return changed;
  }

  /**
   * Pick the point under a screen position by projecting every point on the CPU.
   * Returns the nearest (in depth) point whose on-screen footprint contains the cursor.
   */
  pick(px: number, py: number, camera: THREE.PerspectiveCamera, width: number, height: number): number {
    const m = this._m.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const e = m.elements;
    const fovScale = 1 / Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    const lo = this.material.uniforms.uLodLo.value as number;
    const hi = this.material.uniforms.uLodHi.value as number;
    const dotPx = this.material.uniforms.uDotPx.value as number;
    const p = this.positions;
    const v = this._v;
    let best = -1;
    let bestDepth = Infinity;
    for (let i = 0; i < this.count; i++) {
      if (this.hidden[i]) continue;
      const x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
      v.set(
        e[0] * x + e[4] * y + e[8] * z + e[12],
        e[1] * x + e[5] * y + e[9] * z + e[13],
        e[2] * x + e[6] * y + e[10] * z + e[14],
        e[3] * x + e[7] * y + e[11] * z + e[15],
      );
      if (v.w <= 0.01) continue;
      const sx = (v.x / v.w * 0.5 + 0.5) * width;
      const sy = (-v.y / v.w * 0.5 + 0.5) * height;
      const dist = v.w;
      const thumbPx = (this.thumbSize * height * 0.5 * fovScale) / dist;
      const t = THREE.MathUtils.smoothstep(thumbPx, lo, hi);
      const radius = Math.max(dotPx * 0.5 + 3, THREE.MathUtils.lerp(dotPx * 0.5 + 3, thumbPx * 0.5, t));
      const dx = sx - px, dy = sy - py;
      if (dx * dx + dy * dy <= radius * radius && dist < bestDepth) {
        bestDepth = dist;
        best = i;
      }
    }
    return best;
  }

  /**
   * Add one image in a free slot. Returns false when the cloud is full.
   * The point fades in at its position in the current layout.
   */
  append(id: number, positions: Record<string, number[]>, currentLayout: string, color: THREE.Color, thumbUrl: string): boolean {
    if (id !== this.count || id >= this.capacity) return false;
    const p = positions[currentLayout] ?? positions[Object.keys(positions)[0]] ?? [0, 0, 0];
    const o = id * 3;
    for (const arr of [this.positions, this.layoutFrom, this.layoutTo, this.pos0.array as Float32Array, this.pos1.array as Float32Array]) {
      arr[o] = p[0];
      arr[o + 1] = p[1];
      arr[o + 2] = p[2];
    }
    const c = this.colorAttr.array as Float32Array;
    c[o] = color.r;
    c[o + 1] = color.g;
    c[o + 2] = color.b;
    (this.alphaAttr.array as Float32Array)[id] = 0;
    this.alphaTarget[id] = 1;
    (this.highlightAttr.array as Float32Array)[id] = 0;
    this.hidden[id] = 0;
    (this.hiddenAttr.array as Float32Array)[id] = 0;
    this.pos0.needsUpdate = this.pos1.needsUpdate = this.colorAttr.needsUpdate = true;
    this.alphaAttr.needsUpdate = this.highlightAttr.needsUpdate = this.hiddenAttr.needsUpdate = true;
    this.alphaSettled = false;
    this.count = id + 1;
    this.geometry.instanceCount = this.count;
    void this.loadThumbnail(id, thumbUrl);
    return true;
  }

  /** Fetch one thumbnail and copy it into its atlas cell. */
  async loadThumbnail(i: number, url: string): Promise<void> {
    const cell = this.cell;
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(String(res.status));
      const blob = await res.blob();
      const bmp = await createImageBitmap(blob, { resizeWidth: cell, resizeHeight: cell, resizeQuality: 'high' });
      if (this.aborted) {
        bmp.close();
        return;
      }
      const src = new THREE.Texture(bmp);
      src.flipY = false;
      src.colorSpace = THREE.SRGBColorSpace;
      this.renderer.copyTextureToTexture(src, this.atlas, null, new THREE.Vector2((i % this.cols) * cell, Math.floor(i / this.cols) * cell));
      bmp.close();
    } catch {
      /* leave the cell empty; the sprite still renders as a dot */
    }
  }

  /** Stream thumbnails into the atlas. Resolves when all are done. */
  async loadThumbnails(urls: string[], onProgress?: (done: number, total: number) => void): Promise<void> {
    const total = urls.length;
    let next = 0;
    const worker = async () => {
      while (next < total && !this.aborted) {
        const i = next++;
        await this.loadThumbnail(i, urls[i]);
        this.loaded++;
        onProgress?.(this.loaded, total);
      }
    };
    await Promise.all(Array.from({ length: 8 }, worker));
  }

  dispose(): void {
    this.aborted = true;
    this.geometry.dispose();
    this.material.dispose();
    this.atlas.dispose();
  }
}
