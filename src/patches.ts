import * as THREE from 'three';
import type { Category, PointsData } from './data';

/**
 * Every patch token of every image as a point in a shared PCA space. The whole set is drawn
 * faintly, colored by category, as context. The patches of the selected image (slot A) and
 * of the compared image (slot B) are drawn bright and large, so two similar images show up as
 * two overlapping clouds.
 */

const vertexShader = /* glsl */ `
  attribute vec3 aColor;
  attribute float aSlot;
  attribute float aHidden;

  uniform float uOpacity;
  uniform float uGrow;
  uniform float uBgAlpha;
  uniform float uBasePx;
  uniform float uHiPx;
  uniform float uHome;
  uniform float uPixelRatio;
  uniform vec3 uColorA;
  uniform vec3 uColorB;

  varying vec3 vColor;
  varying float vAlpha;
  varying float vDist;

  void main() {
    if (aHidden > 0.5) {
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      gl_PointSize = 0.0;
      return;
    }
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float dist = max(1e-4, -mv.z);
    float hi = step(0.5, aSlot);
    float att = clamp(uHome / dist, 0.5, 5.0);
    float size = mix(uBasePx, mix(uBasePx, uHiPx * att, uGrow), hi);
    gl_PointSize = size * uPixelRatio;
    gl_Position = projectionMatrix * mv;

    vColor = hi > 0.5 ? (aSlot > 1.5 ? uColorB : uColorA) : aColor;
    // background fades further while something is highlighted so the clouds stand out
    float bg = uBgAlpha * mix(1.0, 0.35, uGrow);
    vAlpha = mix(bg, 0.95, hi * uGrow) * uOpacity;
    vDist = dist;
  }
`;

const fragmentShader = /* glsl */ `
  precision highp float;

  uniform vec3 uBg;
  uniform float uFogNear;
  uniform float uFogFar;
  uniform float uFogMax;

  varying vec3 vColor;
  varying float vAlpha;
  varying float vDist;

  void main() {
    vec2 p = gl_PointCoord - 0.5;
    float r = length(p);
    float disc = 1.0 - smoothstep(0.42, 0.5, r);
    if (disc < 0.02) discard;
    float fog = clamp((vDist - uFogNear) / (uFogFar - uFogNear), 0.0, 1.0) * uFogMax;
    vec3 col = mix(vColor, uBg, fog);
    gl_FragColor = vec4(col, disc * vAlpha);
    #include <colorspace_fragment>
  }
`;

export interface PatchCloudOptions {
  background: THREE.Color;
  fogNear: number;
  fogFar: number;
  homeDistance: number;
  pixelRatio: number;
}

export interface PatchHit {
  image: number;
  patch: number;
  slot: 1 | 2;
}

export class PatchCloud {
  readonly points: THREE.Points;
  readonly material: THREE.ShaderMaterial;
  readonly perImage: number;
  readonly grid: [number, number];
  /** Images currently in the cloud. */
  count: number;
  readonly capacity: number;
  /** xyz of every patch, image-major: [image][patch][3] */
  readonly positions: Float32Array;

  private readonly slotAttr: THREE.BufferAttribute;
  private readonly hiddenAttr: THREE.BufferAttribute;
  private readonly posAttr: THREE.BufferAttribute;
  private readonly colorAttr: THREE.BufferAttribute;
  private slotA = -1;
  private slotB = -1;
  private opacity = 0;
  private opacityTarget = 0;
  private grow = 0;
  private growTarget = 0;

  private readonly _m = new THREE.Matrix4();
  private readonly _v = new THREE.Vector4();

  constructor(data: PointsData, categories: Map<string, Category>, xyz: Float32Array, capacity: number, opts: PatchCloudOptions) {
    const info = data.patches!;
    const n = data.items.length;
    this.perImage = info.count;
    this.grid = info.grid;
    this.count = n;
    this.capacity = Math.max(capacity, n);
    this.positions = new Float32Array(this.capacity * info.count * 3);
    this.positions.set(xyz.subarray(0, n * info.count * 3));

    const total = n * info.count;
    const colors = new Float32Array(this.capacity * info.count * 3);
    for (let i = 0; i < n; i++) {
      const c = categories.get(data.items[i].category ?? '')!.color;
      for (let k = 0; k < info.count; k++) {
        const o = (i * info.count + k) * 3;
        colors[o] = c.r;
        colors[o + 1] = c.g;
        colors[o + 2] = c.b;
      }
    }

    const geometry = new THREE.BufferGeometry();
    this.posAttr = new THREE.BufferAttribute(this.positions, 3);
    this.colorAttr = new THREE.BufferAttribute(colors, 3);
    this.posAttr.setUsage(THREE.DynamicDrawUsage);
    this.colorAttr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('position', this.posAttr);
    geometry.setAttribute('aColor', this.colorAttr);
    this.slotAttr = new THREE.BufferAttribute(new Float32Array(this.capacity * info.count), 1);
    this.slotAttr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('aSlot', this.slotAttr);
    this.hiddenAttr = new THREE.BufferAttribute(new Float32Array(this.capacity * info.count), 1);
    this.hiddenAttr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('aHidden', this.hiddenAttr);
    geometry.setDrawRange(0, total);

    this.material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        uOpacity: { value: 0 },
        uGrow: { value: 0 },
        // denser clouds get fainter dots so the overall density stays readable
        uBgAlpha: { value: THREE.MathUtils.clamp(0.3 * Math.sqrt(40000 / total), 0.03, 0.3) },
        uBasePx: { value: 2.2 },
        uHiPx: { value: 7 },
        uHome: { value: opts.homeDistance },
        uPixelRatio: { value: opts.pixelRatio },
        uColorA: { value: new THREE.Color('#ffffff') },
        uColorB: { value: new THREE.Color('#7aa2ff') },
        uBg: { value: opts.background },
        uFogNear: { value: opts.fogNear },
        uFogFar: { value: opts.fogFar },
        uFogMax: { value: 0.8 },
      },
      transparent: true,
      depthWrite: false,
      depthTest: true,
    });

    this.points = new THREE.Points(geometry, this.material);
    this.points.frustumCulled = false;
    this.points.visible = false;
    this.points.renderOrder = 1;
  }

  /** Add the patches of image `id` (must be the next slot). */
  append(id: number, xyz: number[][], color: THREE.Color): boolean {
    if (id !== this.count || id >= this.capacity || xyz.length !== this.perImage) return false;
    const base = id * this.perImage * 3;
    const col = this.colorAttr.array as Float32Array;
    for (let k = 0; k < this.perImage; k++) {
      const o = base + k * 3;
      this.positions[o] = xyz[k][0];
      this.positions[o + 1] = xyz[k][1];
      this.positions[o + 2] = xyz[k][2];
      col[o] = color.r;
      col[o + 1] = color.g;
      col[o + 2] = color.b;
    }
    this.count = id + 1;
    (this.hiddenAttr.array as Float32Array).fill(0, id * this.perImage, (id + 1) * this.perImage);
    this.hiddenAttr.needsUpdate = true;
    this.posAttr.needsUpdate = true;
    this.colorAttr.needsUpdate = true;
    this.points.geometry.setDrawRange(0, this.count * this.perImage);
    return true;
  }

  setVisible(on: boolean): void {
    this.opacityTarget = on ? 1 : 0;
    if (on) this.points.visible = true;
  }

  get visible(): boolean {
    return this.opacityTarget > 0;
  }

  /** Drop (or restore) every patch of one image. */
  setHidden(image: number, on: boolean): void {
    const p = this.perImage;
    (this.hiddenAttr.array as Float32Array).fill(on ? 1 : 0, image * p, (image + 1) * p);
    this.hiddenAttr.needsUpdate = true;
  }

  /** Highlight the patches of image `a` (white) and image `b` (accent). Pass -1 to clear. */
  setHighlight(a: number, b: number): void {
    if (a === this.slotA && b === this.slotB) return;
    const arr = this.slotAttr.array as Float32Array;
    const p = this.perImage;
    if (this.slotA >= 0) arr.fill(0, this.slotA * p, (this.slotA + 1) * p);
    if (this.slotB >= 0) arr.fill(0, this.slotB * p, (this.slotB + 1) * p);
    if (a >= 0) arr.fill(1, a * p, (a + 1) * p);
    if (b >= 0) arr.fill(2, b * p, (b + 1) * p);
    this.slotAttr.needsUpdate = true;
    const hadAny = this.slotA >= 0 || this.slotB >= 0;
    this.slotA = a;
    this.slotB = b;
    this.growTarget = a >= 0 || b >= 0 ? 1 : 0;
    if (!hadAny && this.growTarget > 0) this.grow = 0; // pop in from the background size
  }

  patchCenter(image: number, patch: number, out: THREE.Vector3): THREE.Vector3 {
    const o = (image * this.perImage + patch) * 3;
    return out.set(this.positions[o], this.positions[o + 1], this.positions[o + 2]);
  }

  /** Returns true if anything changed this frame. */
  update(dt: number, pixelRatio: number): boolean {
    const u = this.material.uniforms;
    u.uPixelRatio.value = pixelRatio;
    let changed = false;
    const a = 1 - Math.exp(-8 * dt);
    if (Math.abs(this.opacity - this.opacityTarget) > 0.002) {
      this.opacity += (this.opacityTarget - this.opacity) * a;
      changed = true;
    } else if (this.opacity !== this.opacityTarget) {
      this.opacity = this.opacityTarget;
      changed = true;
    }
    if (Math.abs(this.grow - this.growTarget) > 0.002) {
      this.grow += (this.growTarget - this.grow) * a;
      changed = true;
    } else if (this.grow !== this.growTarget) {
      this.grow = this.growTarget;
      changed = true;
    }
    u.uOpacity.value = this.opacity;
    u.uGrow.value = this.grow;
    this.points.visible = this.opacity > 0.005;
    return changed;
  }

  /** Pick among the highlighted patches only (at most two images' worth). */
  pick(px: number, py: number, camera: THREE.PerspectiveCamera, width: number, height: number): PatchHit | null {
    if (this.opacity < 0.5) return null;
    const m = this._m.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const e = m.elements;
    const home = this.material.uniforms.uHome.value as number;
    const hiPx = this.material.uniforms.uHiPx.value as number;
    const v = this._v;
    let best: PatchHit | null = null;
    let bestDepth = Infinity;
    const slots: Array<[number, 1 | 2]> = [];
    if (this.slotA >= 0) slots.push([this.slotA, 1]);
    if (this.slotB >= 0) slots.push([this.slotB, 2]);
    for (const [image, slot] of slots) {
      for (let k = 0; k < this.perImage; k++) {
        const o = (image * this.perImage + k) * 3;
        const x = this.positions[o], y = this.positions[o + 1], z = this.positions[o + 2];
        v.set(
          e[0] * x + e[4] * y + e[8] * z + e[12],
          e[1] * x + e[5] * y + e[9] * z + e[13],
          e[2] * x + e[6] * y + e[10] * z + e[14],
          e[3] * x + e[7] * y + e[11] * z + e[15],
        );
        if (v.w <= 0.01) continue;
        const sx = (v.x / v.w * 0.5 + 0.5) * width;
        const sy = (-v.y / v.w * 0.5 + 0.5) * height;
        const radius = Math.max(4, (hiPx * THREE.MathUtils.clamp(home / v.w, 0.5, 5)) * 0.5 + 2);
        const dx = sx - px, dy = sy - py;
        if (dx * dx + dy * dy <= radius * radius && v.w < bestDepth) {
          bestDepth = v.w;
          best = { image, patch: k, slot };
        }
      }
    }
    return best;
  }

  dispose(): void {
    this.points.geometry.dispose();
    this.material.dispose();
  }
}

/**
 * For each patch of image `a`, its most similar patch in image `b` by cosine over the
 * reduced patch features (or by 3D distance when no features are available).
 * Returns matches [patchInB per patch of A] and the per-patch similarity.
 */
export function matchPatches(
  a: number,
  b: number,
  perImage: number,
  feats: Float32Array | null,
  featDims: number,
  xyz: Float32Array,
): { match: Int32Array; sim: Float32Array; mean: number; exact: boolean } {
  const match = new Int32Array(perImage);
  const sim = new Float32Array(perImage);
  let total = 0;
  if (feats) {
    const norms = (img: number) => {
      const out = new Float32Array(perImage);
      for (let k = 0; k < perImage; k++) {
        let s = 0;
        const o = (img * perImage + k) * featDims;
        for (let d = 0; d < featDims; d++) s += feats[o + d] * feats[o + d];
        out[k] = Math.sqrt(s) || 1;
      }
      return out;
    };
    const na = norms(a);
    const nb = norms(b);
    for (let i = 0; i < perImage; i++) {
      const oa = (a * perImage + i) * featDims;
      let best = -Infinity;
      let bestJ = 0;
      for (let j = 0; j < perImage; j++) {
        const ob = (b * perImage + j) * featDims;
        let dot = 0;
        for (let d = 0; d < featDims; d++) dot += feats[oa + d] * feats[ob + d];
        const c = dot / (na[i] * nb[j]);
        if (c > best) {
          best = c;
          bestJ = j;
        }
      }
      match[i] = bestJ;
      sim[i] = best;
      total += best;
    }
    return { match, sim, mean: total / perImage, exact: true };
  }
  for (let i = 0; i < perImage; i++) {
    const oa = (a * perImage + i) * 3;
    let best = Infinity;
    let bestJ = 0;
    for (let j = 0; j < perImage; j++) {
      const ob = (b * perImage + j) * 3;
      const d = (xyz[oa] - xyz[ob]) ** 2 + (xyz[oa + 1] - xyz[ob + 1]) ** 2 + (xyz[oa + 2] - xyz[ob + 2]) ** 2;
      if (d < best) {
        best = d;
        bestJ = j;
      }
    }
    match[i] = bestJ;
    sim[i] = Math.sqrt(best);
    total += sim[i];
  }
  return { match, sim, mean: total / perImage, exact: false };
}
