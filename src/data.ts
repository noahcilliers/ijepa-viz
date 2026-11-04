import * as THREE from 'three';

export interface Item {
  id: number;
  file: string;
  thumb: string;
  image?: string;
  category: string;
}

export interface Neighbor {
  id: number;
  sim: number;
}

export interface PatchInfo {
  file: string;
  count: number;
  grid: [number, number];
  feats?: string;
  featDims?: number;
}

export interface PointsData {
  model: string;
  pooling: string;
  dim?: number;
  layouts: Record<string, number[][]>;
  items: Item[];
  neighbors: Neighbor[][];
  similarity?: string;
  patches?: PatchInfo;
}

export const LAYOUT_LABELS: Record<string, string> = { pca: 'PCA', umap: 'UMAP', patch: 'Patches' };

/** Data folder under public/, switchable with ?data=<folder> for comparing runs. */
export const DATA_BASE = (new URLSearchParams(location.search).get('data') ?? 'data').replace(/\/+$/, '') + '/';

export async function loadPoints(): Promise<PointsData | null> {
  const res = await fetch(`${DATA_BASE}points.json`, { cache: 'no-cache' });
  if (!res.ok) return null;
  return (await res.json()) as PointsData;
}

/** Fetch a raw float32 file and check its length. */
export async function loadFloat32(url: string, expected: number): Promise<Float32Array | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const buf = await res.arrayBuffer();
    if (buf.byteLength !== expected * 4) return null;
    return new Float32Array(buf);
  } catch {
    return null;
  }
}

/** Full N x N cosine matrix, when the embedding script wrote one. */
export async function loadSimilarity(data: PointsData): Promise<Float32Array | null> {
  if (!data.similarity) return null;
  try {
    const res = await fetch(DATA_BASE + data.similarity);
    if (!res.ok) return null;
    const buf = await res.arrayBuffer();
    const n = data.items.length;
    if (buf.byteLength !== n * n * 4) return null;
    return new Float32Array(buf);
  } catch {
    return null;
  }
}

/** Exact cosine similarity if the matrix is loaded, else from the neighbor lists, else null. */
export function similarityBetween(
  data: PointsData,
  sim: Float32Array | null,
  a: number,
  b: number,
): number | null {
  if (a === b) return 1;
  if (sim) return sim[a * data.items.length + b];
  const fromA = data.neighbors[a]?.find((nb) => nb.id === b);
  if (fromA) return fromA.sim;
  const fromB = data.neighbors[b]?.find((nb) => nb.id === a);
  return fromB ? fromB.sim : null;
}

const PALETTE = [
  '#7aa2ff', '#ff8a65', '#4dd0a4', '#ffd166', '#c792ea', '#4fc3f7',
  '#f06292', '#aed581', '#ffab40', '#80cbc4', '#b39ddb', '#e57373',
];
export const NEUTRAL = '#9aa3b2';

export interface Category {
  name: string;
  hex: string;
  color: THREE.Color;
  count: number;
}

export function buildCategories(items: Item[]): Map<string, Category> {
  const counts = new Map<string, number>();
  for (const it of items) counts.set(it.category ?? '', (counts.get(it.category ?? '') ?? 0) + 1);
  const names = [...counts.keys()].filter((n) => n !== '').sort((a, b) => a.localeCompare(b));
  const out = new Map<string, Category>();
  names.forEach((name, i) => {
    const hex = i < PALETTE.length ? PALETTE[i] : hslHex((i * 137.508) % 360, 62, 68);
    out.set(name, { name, hex, color: new THREE.Color(hex), count: counts.get(name)! });
  });
  if (counts.has('')) out.set('', { name: '', hex: NEUTRAL, color: new THREE.Color(NEUTRAL), count: counts.get('')! });
  return out;
}

/** Register a category that appeared after load (an upload). */
export function addCategory(categories: Map<string, Category>, name: string): Category {
  const existing = categories.get(name);
  if (existing) return existing;
  if (name === '') {
    const c = { name, hex: NEUTRAL, color: new THREE.Color(NEUTRAL), count: 0 };
    categories.set('', c);
    return c;
  }
  const i = [...categories.keys()].filter((k) => k !== '').length;
  const hex = i < PALETTE.length ? PALETTE[i] : hslHex((i * 137.508) % 360, 62, 68);
  const c = { name, hex, color: new THREE.Color(hex), count: 0 };
  categories.set(name, c);
  return c;
}

function hslHex(h: number, s: number, l: number): string {
  const c = new THREE.Color().setHSL(h / 360, s / 100, l / 100);
  return '#' + c.getHexString();
}
