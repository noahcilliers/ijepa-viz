import { DATA_BASE, LAYOUT_LABELS, type Category, type Item, type Neighbor } from './data';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

export interface UICallbacks {
  onSearch(query: string): void;
  onLayout(name: string): void;
  onGrid(on: boolean): void;
  onLines(on: boolean): void;
  onThumbnails(on: boolean): void;
  onToggleFly(): void;
  onReset(): void;
  onSelect(id: number): void;
  onClose(): void;
  /** Show or hide every image of a category. */
  onCategoryToggle(name: string, visible: boolean): void;
  /** Show only this category (or everything again if it is already alone). */
  onCategorySolo(name: string): void;
  onImageToggle(id: number, visible: boolean): void;
  onShowAll(visible: boolean): void;
  onFiles(files: File[], category: string): void;
  onUploadOpen(): void;
}

interface DrawerCategory {
  name: string;
  ids: number[];
  el: HTMLElement;
  box: HTMLElement;
  count: HTMLElement;
  grid: HTMLElement;
  built: boolean;
}

const DRAWER_KEY = 'jepa-explorer:drawer';

export class UI {
  private readonly tooltip = $('tooltip');
  private readonly tooltipImg = $<HTMLImageElement>('tooltip-img');
  private readonly tooltipFile = $('tooltip-file');
  private readonly tooltipCat = $('tooltip-cat');
  private readonly tooltipSub = $('tooltip-sub');
  private readonly patchBox = $('patch-box');
  private readonly patchBoxB = $('patch-box-b');
  private readonly patchHint = $('patch-hint');
  private readonly comparePatches = $('compare-patches');
  private readonly comparePatchSim = $('compare-patch-sim');
  private readonly comparePatchLabel = $('compare-patch-label');
  private readonly panel = $('panel');
  private readonly panelImg = $<HTMLImageElement>('panel-img');
  private readonly panelFile = $('panel-file');
  private readonly panelCat = $('panel-cat');
  private readonly neighborsEl = $('neighbors');
  private readonly compare = $('compare');
  private readonly compareA = $<HTMLImageElement>('compare-a');
  private readonly compareB = $<HTMLImageElement>('compare-b');
  private readonly compareSim = $('compare-sim');
  private readonly compareFile = $('compare-file');
  private readonly search = $<HTMLInputElement>('search');
  private readonly searchCount = $('search-count');
  private readonly drawer = $('drawer');
  private readonly drawerList = $('drawer-list');
  private readonly drawerCount = $('drawer-count');
  private readonly btnLayers = $('btn-layers');
  private readonly optGrid = $<HTMLInputElement>('opt-grid');
  private readonly optLines = $<HTMLInputElement>('opt-lines');
  private readonly optThumbs = $<HTMLInputElement>('opt-thumbs');
  private readonly stats = $('stats');
  private readonly loading = $('loading');
  private readonly loadingFill = $('loading-fill');
  private readonly loadingText = $('loading-text');
  private readonly btnFly = $('btn-fly');
  private readonly layoutToggle = $('layout-toggle');
  private readonly upload = $('upload');
  private readonly uploadStatus = $('upload-status');
  private readonly uploadCategory = $<HTMLInputElement>('upload-category');
  private readonly btnAdd = $('btn-add');
  private readonly dropOverlay = $('drop-overlay');

  private tooltipId = -1;
  private tooltipSubText = '';
  private tooltipX = 0;
  private tooltipY = 0;
  private categories: Map<string, Category> = new Map();
  private items: Item[] = [];
  private hidden: Uint8Array = new Uint8Array(0);
  private drawerCats: DrawerCategory[] = [];
  private expanded = new Set<string>();
  private thumbEls = new Map<number, HTMLElement>();
  private selectedId = -1;
  private comparedId = -1;

  constructor(private readonly cb: UICallbacks) {
    let timer = 0;
    this.search.addEventListener('input', () => {
      clearTimeout(timer);
      timer = window.setTimeout(() => cb.onSearch(this.search.value.trim()), 80);
    });
    this.search.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        this.search.value = '';
        cb.onSearch('');
        this.search.blur();
      }
      e.stopPropagation();
    });

    this.layoutToggle.querySelectorAll<HTMLButtonElement>('button').forEach((b) =>
      b.addEventListener('click', () => cb.onLayout(b.dataset.layout!)),
    );
    this.optGrid.addEventListener('change', () => cb.onGrid(this.optGrid.checked));
    this.optLines.addEventListener('change', () => cb.onLines(this.optLines.checked));
    this.optThumbs.addEventListener('change', () => cb.onThumbnails(this.optThumbs.checked));
    this.btnLayers.addEventListener('click', () => this.setDrawerOpen(!!this.drawer.hidden));
    $('show-all').addEventListener('click', () => cb.onShowAll(true));
    $('hide-all').addEventListener('click', () => cb.onShowAll(false));
    let drawerOpen = window.innerWidth > 900;
    try {
      const saved = localStorage.getItem(DRAWER_KEY);
      if (saved !== null) drawerOpen = saved === '1';
    } catch { /* private mode */ }
    this.setDrawerOpen(drawerOpen, false);
    this.btnFly.addEventListener('click', cb.onToggleFly);
    $('btn-reset').addEventListener('click', cb.onReset);
    $('panel-close').addEventListener('click', cb.onClose);

    const help = $('help');
    $('help-toggle').addEventListener('click', () => {
      const open = help.classList.toggle('open');
      $('help-toggle').setAttribute('aria-expanded', String(open));
    });
    document.addEventListener('pointerdown', (e) => {
      if (!help.contains(e.target as Node)) help.classList.remove('open');
      if (!this.upload.contains(e.target as Node) && !this.btnAdd.contains(e.target as Node)) this.setUploadOpen(false);
    });

    // uploads: button + popover, file picker, and drag-and-drop anywhere
    this.btnAdd.addEventListener('click', () => this.setUploadOpen(!!this.upload.hidden));
    const files = $<HTMLInputElement>('upload-files');
    $('upload-choose').addEventListener('click', () => files.click());
    files.addEventListener('change', () => {
      if (files.files?.length) cb.onFiles([...files.files], this.uploadCategory.value.trim());
      files.value = '';
    });
    this.uploadCategory.addEventListener('keydown', (e) => e.stopPropagation());
    let dragDepth = 0;
    window.addEventListener('dragenter', (e) => {
      if (!e.dataTransfer?.types.includes('Files')) return;
      e.preventDefault();
      dragDepth++;
      this.dropOverlay.hidden = false;
    });
    window.addEventListener('dragover', (e) => {
      if (e.dataTransfer?.types.includes('Files')) e.preventDefault();
    });
    window.addEventListener('dragleave', () => {
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) this.dropOverlay.hidden = true;
    });
    window.addEventListener('drop', (e) => {
      dragDepth = 0;
      this.dropOverlay.hidden = true;
      if (!e.dataTransfer?.files.length) return;
      e.preventDefault();
      const imgs = [...e.dataTransfer.files].filter((f) => /^image\/(jpeg|png|webp)$/.test(f.type));
      if (imgs.length) cb.onFiles(imgs, this.uploadCategory.value.trim());
      else this.setUploadStatus('Only JPEG, PNG and WebP images can be added.', 'err');
    });
  }

  setUploadOpen(open: boolean): void {
    this.upload.hidden = !open;
    this.btnAdd.setAttribute('aria-expanded', String(open));
    if (open) this.cb.onUploadOpen();
  }

  setUploadStatus(html: string, kind: 'ok' | 'err' | '' = ''): void {
    this.uploadStatus.innerHTML = html;
    this.uploadStatus.className = `upload-status ${kind}`;
  }

  setMeta(text: string): void {
    $('meta').textContent = text;
  }

  setLayouts(names: string[], active: string): void {
    this.layoutToggle.querySelectorAll<HTMLButtonElement>('button').forEach((b) => {
      const name = b.dataset.layout!;
      b.disabled = !names.includes(name);
      b.classList.toggle('active', name === active);
      b.textContent = LAYOUT_LABELS[name] ?? name;
      b.title = b.disabled
        ? name === 'umap' ? 'Run embed.py with --umap to enable' : 'Re-run embed.py to export patch tokens'
        : name === 'patch' ? 'Every image as its patch token embeddings' : '';
    });
  }

  setGrid(on: boolean): void {
    this.optGrid.checked = on;
  }

  get linesEnabled(): boolean {
    return this.optLines.checked;
  }

  get drawerOpen(): boolean {
    return !this.drawer.hidden;
  }

  setDrawerOpen(open: boolean, remember = true): void {
    this.drawer.hidden = !open;
    this.btnLayers.setAttribute('aria-pressed', String(open));
    if (remember) {
      try {
        localStorage.setItem(DRAWER_KEY, open ? '1' : '0');
      } catch { /* ignore */ }
    }
  }

  setFly(on: boolean): void {
    this.btnFly.setAttribute('aria-pressed', String(on));
  }

  setStats(text: string): void {
    if (this.stats.textContent !== text) this.stats.textContent = text;
  }

  setLoading(done: number, total: number, label = 'Loading thumbnails'): void {
    if (done >= total) {
      this.loadingFill.style.width = '100%';
      this.loadingText.textContent = label === 'Embedding' ? `Embedded ${total} image${total === 1 ? '' : 's'}` : `${total} thumbnails loaded`;
      this.loading.classList.add('done');
      return;
    }
    this.loading.classList.remove('done');
    this.loadingFill.style.width = `${(100 * done) / total}%`;
    this.loadingText.textContent = `${label} ${done} / ${total}`;
  }

  setSearchCount(matches: number, total: number, active: boolean): void {
    this.searchCount.textContent = active ? `${matches} / ${total}` : '';
  }

  // ---------------------------------------------------------------- drawer

  /** (Re)build the category list. Call again when images or categories are added. */
  buildDrawer(categories: Map<string, Category>, items: Item[], hidden: Uint8Array): void {
    this.categories = categories;
    this.items = items;
    this.hidden = hidden;
    const byCat = new Map<string, number[]>();
    for (const it of items) {
      const key = it.category ?? '';
      let list = byCat.get(key);
      if (!list) byCat.set(key, (list = []));
      list.push(it.id);
    }
    const names = [...categories.keys()].filter((k) => k !== '' && byCat.has(k));
    if (byCat.has('')) names.push('');

    this.drawerList.innerHTML = '';
    this.drawerCats = [];
    this.thumbEls.clear();
    if (names.length === 0) {
      this.drawerList.innerHTML = '<div class="drawer-empty">No images yet.</div>';
    }
    for (const name of names) {
      const c = categories.get(name)!;
      const ids = byCat.get(name)!;
      const el = document.createElement('div');
      el.className = 'cat';
      el.style.setProperty('--c', c.hex);
      el.innerHTML =
        `<div class="cat-row">` +
        `<button class="cat-box" role="checkbox" aria-checked="true"></button>` +
        `<span class="cat-name"></span><span class="cat-count"></span>` +
        `<button class="cat-only" title="Show only these images">only</button>` +
        `<button class="cat-expand" title="Pick individual images" aria-expanded="false">` +
        `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="m9 6 6 6-6 6"/></svg>` +
        `</button></div><div class="cat-grid" hidden></div>`;
      const nameEl = el.querySelector<HTMLElement>('.cat-name')!;
      nameEl.textContent = name || 'uncategorized';
      nameEl.classList.toggle('unnamed', name === '');
      const box = el.querySelector<HTMLElement>('.cat-box')!;
      const grid = el.querySelector<HTMLElement>('.cat-grid')!;
      const expand = el.querySelector<HTMLElement>('.cat-expand')!;
      const entry: DrawerCategory = { name, ids, el, box, count: el.querySelector('.cat-count')!, grid, built: false };
      const toggle = () => this.cb.onCategoryToggle(name, box.getAttribute('aria-checked') !== 'true');
      box.addEventListener('click', toggle);
      nameEl.addEventListener('click', toggle);
      el.querySelector('.cat-only')!.addEventListener('click', () => this.cb.onCategorySolo(name));
      expand.addEventListener('click', () => this.setExpanded(entry, !!grid.hidden));
      this.drawerList.appendChild(el);
      this.drawerCats.push(entry);
      if (this.expanded.has(name)) this.setExpanded(entry, true);
    }
    this.updateDrawer();
  }

  private setExpanded(entry: DrawerCategory, open: boolean): void {
    if (open && !entry.built) {
      entry.built = true;
      const frag = document.createDocumentFragment();
      for (const id of entry.ids) frag.appendChild(this.makeThumb(id));
      entry.grid.appendChild(frag);
    }
    entry.grid.hidden = !open;
    entry.el.querySelector('.cat-expand')!.setAttribute('aria-expanded', String(open));
    if (open) this.expanded.add(entry.name);
    else this.expanded.delete(entry.name);
    if (open) this.updateDrawer();
  }

  private makeThumb(id: number): HTMLElement {
    const it = this.items[id];
    const b = document.createElement('button');
    b.className = 'cat-thumb';
    b.title = `${it.file}\nClick to show or hide · double-click to focus`;
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.decoding = 'async';
    img.alt = '';
    img.src = DATA_BASE + it.thumb;
    b.appendChild(img);
    b.addEventListener('click', () => this.cb.onImageToggle(id, !!this.hidden[id]));
    b.addEventListener('dblclick', () => this.cb.onSelect(id));
    this.thumbEls.set(id, b);
    return b;
  }

  /** Refresh checkbox states, counts and thumbnail states from the hidden flags. */
  updateDrawer(): void {
    let shown = 0;
    const total = this.items.length;
    for (let i = 0; i < total; i++) if (!this.hidden[i]) shown++;
    this.drawerCount.textContent = total === 0 ? '' : shown === total ? `${total}` : `${shown} of ${total}`;

    const visibleCats = this.drawerCats.filter((c) => c.ids.some((id) => !this.hidden[id]));
    for (const c of this.drawerCats) {
      let on = 0;
      for (const id of c.ids) if (!this.hidden[id]) on++;
      const state = on === 0 ? 'false' : on === c.ids.length ? 'true' : 'mixed';
      c.box.setAttribute('aria-checked', state);
      c.count.textContent = on === c.ids.length ? `${c.ids.length}` : `${on} / ${c.ids.length}`;
      c.el.classList.toggle('off', on === 0);
      c.el.classList.toggle('solo', visibleCats.length === 1 && visibleCats[0] === c && this.drawerCats.length > 1);
      if (c.built) for (const id of c.ids) this.thumbEls.get(id)?.classList.toggle('off', !!this.hidden[id]);
    }
  }

  /** Outline the selected / compared image in the thumbnail grids. */
  setDrawerSelection(selected: number, compared: number): void {
    this.thumbEls.get(this.selectedId)?.classList.remove('selected');
    this.thumbEls.get(this.comparedId)?.classList.remove('compared');
    this.selectedId = selected;
    this.comparedId = compared;
    this.thumbEls.get(selected)?.classList.add('selected');
    this.thumbEls.get(compared)?.classList.add('compared');
  }

  // --------------------------------------------------------------- tooltip

  showTooltip(item: Item, x: number, y: number, sub = ''): void {
    if (this.tooltipSubText !== sub) {
      this.tooltipSubText = sub;
      this.tooltipSub.textContent = sub;
    }
    if (this.tooltipId !== item.id) {
      this.tooltipId = item.id;
      this.tooltipImg.src = DATA_BASE + item.thumb;
      this.tooltipFile.textContent = item.file;
      this.setCategoryLabel(this.tooltipCat, item.category);
      this.tooltip.hidden = false;
      void this.tooltip.offsetWidth; // flush so the fade-in transition runs
      this.tooltip.classList.add('visible');
    }
    this.tooltipX = x;
    this.tooltipY = y;
    this.placeTooltip();
  }

  hideTooltip(): void {
    if (this.tooltipId === -1) return;
    this.tooltipId = -1;
    this.tooltip.classList.remove('visible');
    setTimeout(() => {
      if (this.tooltipId === -1) this.tooltip.hidden = true;
    }, 150);
  }

  private placeTooltip(): void {
    const w = this.tooltip.offsetWidth + 20;
    const h = this.tooltip.offsetHeight + 20;
    const x = this.tooltipX + w > window.innerWidth ? this.tooltipX - w : this.tooltipX;
    const y = this.tooltipY + h > window.innerHeight ? this.tooltipY - h : this.tooltipY;
    this.tooltip.style.transform = `translate(${x}px, ${y}px)`;
    this.tooltip.style.left = '14px';
    this.tooltip.style.top = '14px';
  }

  // ----------------------------------------------------------------- panel

  showPanel(item: Item, neighbors: Neighbor[], items: Item[], hidden?: Uint8Array): void {
    this.panelImg.src = DATA_BASE + (item.image ?? item.thumb);
    this.panelImg.alt = item.file;
    this.panelFile.textContent = item.file;
    this.setCategoryLabel(this.panelCat, item.category);

    this.neighborsEl.innerHTML = '';
    for (const nb of neighbors) {
      const other = items[nb.id];
      if (!other) continue;
      const li = document.createElement('li');
      li.className = hidden?.[nb.id] ? 'neighbor is-hidden' : 'neighbor';
      li.innerHTML = `<img loading="lazy" alt="" /><div class="neighbor-text"><div class="neighbor-file"></div><div class="neighbor-bar"><span></span></div></div><div class="neighbor-sim"></div>`;
      (li.querySelector('img') as HTMLImageElement).src = DATA_BASE + other.thumb;
      li.querySelector('.neighbor-file')!.textContent = other.file;
      (li.querySelector('.neighbor-bar > span') as HTMLElement).style.width = `${Math.max(0, Math.min(1, nb.sim)) * 100}%`;
      li.querySelector('.neighbor-sim')!.textContent = nb.sim.toFixed(3);
      li.title = other.category ? `${other.file} · ${other.category}` : other.file;
      li.addEventListener('click', () => this.cb.onSelect(nb.id));
      this.neighborsEl.appendChild(li);
    }
    this.panel.hidden = false;
    this.panel.scrollTop = 0;
  }

  hidePanel(): void {
    this.panel.hidden = true;
    this.compare.hidden = true;
    this.hidePatchBox();
  }

  setPatchHint(text: string | null): void {
    this.patchHint.hidden = !text;
    if (text) this.patchHint.textContent = text;
  }

  setPatchCompare(mean: number | null, exact: boolean): void {
    this.comparePatches.hidden = mean === null;
    if (mean === null) return;
    this.comparePatchSim.textContent = exact ? mean.toFixed(3) : mean.toFixed(1);
    this.comparePatchLabel.textContent = exact
      ? 'mean best-match patch cosine (PCA space)'
      : 'mean distance to nearest patch (3D)';
  }

  /** Outline patch (row, col) on the panel image (slot 1) or the compared thumbnail (slot 2). */
  showPatchBox(slot: 1 | 2, row: number, col: number, grid: [number, number]): void {
    const box = slot === 1 ? this.patchBox : this.patchBoxB;
    const img = slot === 1 ? this.panelImg : this.compareB;
    const host = img.parentElement as HTMLElement;
    const hw = host.clientWidth;
    const hh = host.clientHeight;
    const nw = img.naturalWidth || 1;
    const nh = img.naturalHeight || 1;
    // the model squashes the whole image to a square, so patches tile the full picture
    const scale = Math.min(hw / nw, hh / nh);
    const dw = nw * scale;
    const dh = nh * scale;
    const ox = (hw - dw) / 2;
    const oy = (hh - dh) / 2;
    const cw = dw / grid[1];
    const ch = dh / grid[0];
    box.style.left = `${ox + col * cw}px`;
    box.style.top = `${oy + row * ch}px`;
    box.style.width = `${cw}px`;
    box.style.height = `${ch}px`;
    box.hidden = false;
    (slot === 1 ? this.patchBoxB : this.patchBox).hidden = true;
  }

  hidePatchBox(): void {
    this.patchBox.hidden = true;
    this.patchBoxB.hidden = true;
  }

  showCompare(a: Item, b: Item, sim: number | null): void {
    this.compareA.src = DATA_BASE + (a.image ?? a.thumb);
    this.compareB.src = DATA_BASE + (b.image ?? b.thumb);
    this.comparePatches.hidden = true;
    this.compareSim.textContent = sim === null ? 'n/a' : sim.toFixed(4);
    this.compareFile.textContent = sim === null ? `${b.file} · not in the stored neighbor lists` : b.file;
    this.compare.hidden = false;
  }

  hideCompare(): void {
    this.compare.hidden = true;
    this.patchBoxB.hidden = true;
  }

  private setCategoryLabel(el: HTMLElement, category: string): void {
    el.innerHTML = '';
    if (!category) return;
    const c = this.categories.get(category);
    const dot = document.createElement('span');
    dot.className = 'dot';
    dot.style.background = c?.hex ?? '#9aa3b2';
    el.appendChild(dot);
    el.appendChild(document.createTextNode(category));
  }

  showEmptyState(): void {
    this.loading.classList.add('done');
    const el = document.createElement('div');
    el.className = 'empty';
    el.innerHTML = `<div class="empty-card"><h2>No data yet</h2><p>Embed a folder of images first, then reload:</p><p><code>python embed.py --images ./images --out ./public/data</code></p><p>Or render a synthetic cloud to try the viewer:</p><p><code>python scripts/synthetic.py --n 2000 --out ./public/data</code></p></div>`;
    document.body.appendChild(el);
  }
}
