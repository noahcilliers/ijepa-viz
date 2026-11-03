import { DATA_BASE, LAYOUT_LABELS, type Category, type Item, type Neighbor } from './data';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

export interface UICallbacks {
  onSearch(query: string): void;
  onLayout(name: string): void;
  onToggleGrid(): void;
  onToggleFly(): void;
  onReset(): void;
  onSelect(id: number): void;
  onClose(): void;
  onCategoryToggle(name: string): void;
}

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
  private readonly legend = $('legend');
  private readonly stats = $('stats');
  private readonly loading = $('loading');
  private readonly loadingFill = $('loading-fill');
  private readonly loadingText = $('loading-text');
  private readonly btnGrid = $('btn-grid');
  private readonly btnFly = $('btn-fly');
  private readonly layoutToggle = $('layout-toggle');

  private tooltipId = -1;
  private tooltipSubText = '';
  private tooltipX = 0;
  private tooltipY = 0;
  private categories: Map<string, Category> = new Map();

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
    this.btnGrid.addEventListener('click', cb.onToggleGrid);
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
    });
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
    this.btnGrid.setAttribute('aria-pressed', String(on));
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
      this.loadingText.textContent = `${total} thumbnails loaded`;
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

  // ---------------------------------------------------------------- legend

  buildLegend(categories: Map<string, Category>): void {
    this.categories = categories;
    const named = [...categories.values()].filter((c) => c.name !== '');
    this.legend.innerHTML = '';
    if (named.length === 0) {
      this.legend.hidden = true;
      return;
    }
    this.legend.hidden = false;
    for (const c of named) {
      const el = document.createElement('div');
      el.className = 'legend-item';
      el.dataset.name = c.name;
      el.title = `Click to isolate ${c.name}`;
      el.innerHTML = `<span class="legend-swatch" style="background:${c.hex}"></span><span class="legend-name"></span><span class="legend-count">${c.count}</span>`;
      el.querySelector('.legend-name')!.textContent = c.name;
      el.addEventListener('click', () => this.cb.onCategoryToggle(c.name));
      this.legend.appendChild(el);
    }
  }

  setLegendState(active: Set<string> | null): void {
    this.legend.querySelectorAll<HTMLElement>('.legend-item').forEach((el) => {
      const name = el.dataset.name!;
      el.classList.toggle('off', active !== null && !active.has(name));
      el.classList.toggle('active', active !== null && active.has(name));
    });
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

  showPanel(item: Item, neighbors: Neighbor[], items: Item[]): void {
    this.panelImg.src = DATA_BASE + (item.image ?? item.thumb);
    this.panelImg.alt = item.file;
    this.panelFile.textContent = item.file;
    this.setCategoryLabel(this.panelCat, item.category);

    this.neighborsEl.innerHTML = '';
    for (const nb of neighbors) {
      const other = items[nb.id];
      if (!other) continue;
      const li = document.createElement('li');
      li.className = 'neighbor';
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
