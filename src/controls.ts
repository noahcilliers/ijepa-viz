import * as THREE from 'three';

/**
 * Camera rig: orbit / pan / zoom-to-cursor / fly, with critically damped easing.
 *
 * State is a focus point plus a spherical offset (distance, theta, phi). Every input
 * moves the *goal* state; the *current* state chases it each frame, which gives
 * ease-in, ease-out and glide-on-release for free. Scripted flights (click-to-focus,
 * reset) drive both and are cancelled by any user input.
 */

interface Pose {
  target: THREE.Vector3;
  distance: number;
  theta: number;
  phi: number;
}

interface Flight {
  from: Pose;
  to: Pose;
  start: number;
  duration: number;
}

interface PointerState {
  id: number;
  x: number;
  y: number;
  startX: number;
  startY: number;
  button: number;
  type: string;
  shift: boolean;
}

const MIN_PHI = 0.02;
const MAX_PHI = Math.PI - 0.02;
const MIN_DISTANCE = 0.05;
const MAX_DISTANCE = 20000;

const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export class CameraRig {
  readonly camera: THREE.PerspectiveCamera;
  readonly dom: HTMLElement;

  current: Pose = { target: new THREE.Vector3(), distance: 200, theta: 0.6, phi: 1.15 };
  goal: Pose = { target: new THREE.Vector3(), distance: 200, theta: 0.6, phi: 1.15 };
  home: Pose = { target: new THREE.Vector3(), distance: 200, theta: 0.6, phi: 1.15 };

  flyMode = false;
  /** Units per second for WASD. Set from the cloud size. */
  flySpeed = 40;
  rotateSpeed = 1.0;
  zoomSpeed = 1.0;
  /** Chase stiffness: higher is snappier. */
  stiffness = 10;
  /** How many seconds of pointer velocity to carry after releasing a drag. */
  glide = 0.08;

  /** Set to true for one frame whenever the camera actually moves. */
  moved = true;
  onInteract?: () => void;

  private flight: Flight | null = null;
  private pointers = new Map<number, PointerState>();
  private dragMode: 'none' | 'orbit' | 'pan' | 'look' = 'none';
  private velX = 0;
  private velY = 0;
  private lastMoveT = 0;
  private pinchDist = 0;
  private pinchX = 0;
  private pinchY = 0;
  private keys = new Set<string>();
  private flyVel = new THREE.Vector3();
  private suppressClickUntil = 0;
  private now = performance.now();

  private readonly _v = new THREE.Vector3();
  private readonly _offset = new THREE.Vector3();
  private readonly _right = new THREE.Vector3();
  private readonly _up = new THREE.Vector3();
  private readonly _fwd = new THREE.Vector3();
  private readonly _ray = new THREE.Ray();
  private readonly _plane = new THREE.Plane();
  private readonly _ndc = new THREE.Vector2();
  private readonly _before = new THREE.Vector3();
  private readonly _beforeQ = new THREE.Quaternion();

  constructor(camera: THREE.PerspectiveCamera, dom: HTMLElement) {
    this.camera = camera;
    this.dom = dom;
    dom.addEventListener('pointerdown', this.onPointerDown);
    dom.addEventListener('pointermove', this.onPointerMove);
    dom.addEventListener('pointerup', this.onPointerUp);
    dom.addEventListener('pointercancel', this.onPointerUp);
    dom.addEventListener('wheel', this.onWheel, { passive: false });
    dom.addEventListener('contextmenu', (e) => e.preventDefault());
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
    window.addEventListener('blur', () => this.keys.clear());
    this.applyPose(this.current);
  }

  // ------------------------------------------------------------------ public

  /** Define the overview the reset key returns to. */
  setHome(center: THREE.Vector3, radius: number): void {
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const aspect = Number.isFinite(this.camera.aspect) && this.camera.aspect > 0 ? Math.max(0.6, this.camera.aspect) : 1.5;
    const vfit = radius / Math.sin(fov / 2);
    const hfit = radius / Math.sin(Math.atan(Math.tan(fov / 2) * aspect));
    this.home = { target: center.clone(), distance: Math.max(vfit, hfit) * 1.05, theta: 0.6, phi: 1.15 };
    this.flySpeed = Math.max(5, radius * 0.45);
  }

  /** Snap to the overview without animating. */
  snapHome(): void {
    this.flight = null;
    this.current = clonePose(this.home);
    this.goal = clonePose(this.home);
    this.applyPose(this.current);
  }

  /** Animate to the overview. */
  reset(duration = 900): void {
    this.startFlight(this.home, duration);
  }

  /** Animate the focus point to `point`, keeping the current viewing direction. */
  flyTo(point: THREE.Vector3, distance: number, duration = 600): void {
    const to = clonePose(this.current);
    to.target.copy(point);
    to.distance = distance;
    this.startFlight(to, duration);
  }

  setFlyMode(on: boolean): void {
    this.flyMode = on;
    this.dom.style.cursor = on ? 'crosshair' : '';
  }

  /** True if the last pointer gesture was a drag rather than a click. */
  wasDrag(): boolean {
    return performance.now() < this.suppressClickUntil;
  }

  isTyping(): boolean {
    const el = document.activeElement;
    return !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || (el as HTMLElement).isContentEditable);
  }

  update(dt: number, now: number): void {
    dt = Math.min(dt, 0.05);
    this.now = now;
    const before = this._before.copy(this.camera.position);
    const beforeQ = this._beforeQ.copy(this.camera.quaternion);

    if (this.flight) {
      const f = this.flight;
      const t = THREE.MathUtils.clamp((now - f.start) / f.duration, 0, 1);
      const k = easeInOutCubic(t);
      this.current.target.lerpVectors(f.from.target, f.to.target, k);
      this.current.distance = Math.exp(THREE.MathUtils.lerp(Math.log(f.from.distance), Math.log(f.to.distance), k));
      this.current.theta = THREE.MathUtils.lerp(f.from.theta, f.to.theta, k);
      this.current.phi = THREE.MathUtils.lerp(f.from.phi, f.to.phi, k);
      this.goal = clonePose(this.current);
      if (t >= 1) this.flight = null;
    } else {
      this.applyFly(dt);
      const a = 1 - Math.exp(-this.stiffness * dt);
      const c = this.current;
      const g = this.goal;
      c.target.lerp(g.target, a);
      c.distance = Math.exp(THREE.MathUtils.lerp(Math.log(c.distance), Math.log(g.distance), a));
      c.theta += (g.theta - c.theta) * a;
      c.phi += (g.phi - c.phi) * a;
    }

    // never let a stray input lose the data
    const maxD = Math.max(MAX_DISTANCE, this.home.distance * 40);
    this.current.distance = THREE.MathUtils.clamp(this.current.distance, MIN_DISTANCE, maxD);
    this.goal.distance = THREE.MathUtils.clamp(this.goal.distance, MIN_DISTANCE, maxD);
    this.applyPose(this.current);
    this.moved =
      before.distanceToSquared(this.camera.position) > 1e-10 || beforeQ.angleTo(this.camera.quaternion) > 1e-6;
  }

  dispose(): void {
    this.dom.removeEventListener('pointerdown', this.onPointerDown);
    this.dom.removeEventListener('pointermove', this.onPointerMove);
    this.dom.removeEventListener('pointerup', this.onPointerUp);
    this.dom.removeEventListener('pointercancel', this.onPointerUp);
    this.dom.removeEventListener('wheel', this.onWheel);
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
  }

  // ----------------------------------------------------------------- internals

  private applyPose(p: Pose): void {
    sphericalOffset(this._offset, p.theta, p.phi).multiplyScalar(p.distance);
    this.camera.position.copy(p.target).add(this._offset);
    this.camera.lookAt(p.target);
  }

  private startFlight(to: Pose, duration: number): void {
    this.flight = { from: clonePose(this.current), to: clonePose(to), start: this.now, duration };
    this.goal = clonePose(this.current);
    this.flyVel.set(0, 0, 0);
  }

  private interrupt(): void {
    if (this.flight) {
      this.flight = null;
      this.goal = clonePose(this.current);
    }
    this.onInteract?.();
  }

  private basis(): void {
    this.camera.matrixWorld.extractBasis(this._right, this._up, this._fwd);
    this._fwd.negate();
  }

  private applyFly(dt: number): void {
    const k = this.keys;
    const want = this._v.set(0, 0, 0);
    if (k.has('KeyW') || k.has('ArrowUp')) want.z += 1;
    if (k.has('KeyS') || k.has('ArrowDown')) want.z -= 1;
    if (k.has('KeyA') || k.has('ArrowLeft')) want.x -= 1;
    if (k.has('KeyD') || k.has('ArrowRight')) want.x += 1;
    if (k.has('KeyE')) want.y += 1;
    if (k.has('KeyQ')) want.y -= 1;
    const active = want.lengthSq() > 0;
    if (active) {
      want.normalize().multiplyScalar(this.flySpeed * (k.has('ShiftLeft') || k.has('ShiftRight') ? 3.2 : 1));
      if (this.flight) this.interrupt();
    }
    // ease velocity toward the requested one; faster to start than to stop
    const a = 1 - Math.exp(-(active ? 7 : 5) * dt);
    this.flyVel.lerp(want, a);
    if (this.flyVel.lengthSq() < 1e-6) {
      this.flyVel.set(0, 0, 0);
      return;
    }
    this.basis();
    const move = this._offset
      .set(0, 0, 0)
      .addScaledVector(this._right, this.flyVel.x * dt)
      .addScaledVector(this._up, this.flyVel.y * dt)
      .addScaledVector(this._fwd, this.flyVel.z * dt);
    this.goal.target.add(move);
    this.current.target.add(move);
  }

  /** World point under the cursor at the depth of the focus plane. */
  private cursorOnFocusPlane(clientX: number, clientY: number, out: THREE.Vector3): THREE.Vector3 {
    const rect = this.dom.getBoundingClientRect();
    this._ndc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    this._ray.origin.setFromMatrixPosition(this.camera.matrixWorld);
    this._ray.direction.set(this._ndc.x, this._ndc.y, 0.5).unproject(this.camera).sub(this._ray.origin).normalize();
    this.basis();
    this._plane.setFromNormalAndCoplanarPoint(this._fwd, this.goal.target);
    if (!this._ray.intersectPlane(this._plane, out)) out.copy(this.goal.target);
    return out;
  }

  private dolly(factor: number, clientX: number, clientY: number): void {
    const g = this.goal;
    const next = THREE.MathUtils.clamp(g.distance * factor, MIN_DISTANCE, MAX_DISTANCE);
    const real = next / g.distance;
    // keep the point under the cursor fixed on screen
    const p = this.cursorOnFocusPlane(clientX, clientY, this._v);
    g.target.lerp(p, 1 - real);
    g.distance = next;
  }

  private orbitBy(dx: number, dy: number): void {
    const h = this.dom.clientHeight || 1;
    const g = this.goal;
    g.theta -= (2 * Math.PI * dx * this.rotateSpeed) / h;
    g.phi -= (2 * Math.PI * dy * this.rotateSpeed) / h;
    g.phi = THREE.MathUtils.clamp(g.phi, MIN_PHI, MAX_PHI);
  }

  private lookBy(dx: number, dy: number): void {
    const h = this.dom.clientHeight || 1;
    const g = this.goal;
    // keep the camera position fixed and swing the focus point around it
    const pos = sphericalOffset(this._v, g.theta, g.phi).multiplyScalar(g.distance).add(g.target);
    g.theta += (Math.PI * dx * this.rotateSpeed) / h;
    g.phi = THREE.MathUtils.clamp(g.phi + (Math.PI * dy * this.rotateSpeed) / h, MIN_PHI, MAX_PHI);
    sphericalOffset(this._offset, g.theta, g.phi).multiplyScalar(g.distance);
    g.target.copy(pos).sub(this._offset);
  }

  private panBy(dx: number, dy: number): void {
    const h = this.dom.clientHeight || 1;
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const perPx = (2 * this.goal.distance * Math.tan(fov / 2)) / h;
    this.basis();
    this.goal.target.addScaledVector(this._right, -dx * perPx).addScaledVector(this._up, dy * perPx);
  }

  private dragDelta(mode: 'orbit' | 'pan' | 'look', dx: number, dy: number): void {
    if (mode === 'orbit') this.orbitBy(dx, dy);
    else if (mode === 'look') this.lookBy(dx, dy);
    else this.panBy(dx, dy);
  }

  // ------------------------------------------------------------------ events

  private onPointerDown = (e: PointerEvent): void => {
    if (e.button === 1) return;
    try {
      this.dom.setPointerCapture(e.pointerId);
    } catch {
      /* synthetic pointer ids cannot be captured */
    }
    this.dom.focus?.();
    this.interrupt();
    this.pointers.set(e.pointerId, {
      id: e.pointerId, x: e.clientX, y: e.clientY, startX: e.clientX, startY: e.clientY,
      button: e.button, type: e.pointerType, shift: e.shiftKey,
    });
    this.velX = this.velY = 0;
    this.lastMoveT = performance.now();

    if (this.pointers.size === 2) {
      const [a, b] = [...this.pointers.values()];
      this.pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
      this.pinchX = (a.x + b.x) / 2;
      this.pinchY = (a.y + b.y) / 2;
      this.dragMode = 'pan';
    } else if (this.pointers.size === 1) {
      const pan = e.button === 2 || (e.button === 0 && e.shiftKey && e.pointerType === 'mouse' && !this.flyMode);
      // shift+click is reserved for compare; only treat as pan once it actually moves
      this.dragMode = pan ? 'pan' : this.flyMode ? 'look' : 'orbit';
    }
  };

  private onPointerMove = (e: PointerEvent): void => {
    const p = this.pointers.get(e.pointerId);
    if (!p) return;
    const now = performance.now();
    const dx = e.clientX - p.x;
    const dy = e.clientY - p.y;
    p.x = e.clientX;
    p.y = e.clientY;

    if (this.pointers.size >= 2) {
      const [a, b] = [...this.pointers.values()];
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      const cx = (a.x + b.x) / 2;
      const cy = (a.y + b.y) / 2;
      if (this.pinchDist > 0 && dist > 0) this.dolly(this.pinchDist / dist, cx, cy);
      this.panBy(cx - this.pinchX, cy - this.pinchY);
      this.pinchDist = dist;
      this.pinchX = cx;
      this.pinchY = cy;
      this.suppressClickUntil = now + 400;
      return;
    }

    const moved = Math.hypot(e.clientX - p.startX, e.clientY - p.startY);
    if (moved > 3) this.suppressClickUntil = now + 300;
    if (moved <= 3 && p.shift && p.button === 0) return; // might still be a shift-click

    if (this.dragMode === 'none') return;
    this.dragDelta(this.dragMode, dx, dy);

    const dt = Math.max(4, now - this.lastMoveT);
    const blend = 0.5;
    const cap = 1500; // px/s
    this.velX = THREE.MathUtils.clamp(this.velX * (1 - blend) + (dx / dt) * 1000 * blend, -cap, cap);
    this.velY = THREE.MathUtils.clamp(this.velY * (1 - blend) + (dy / dt) * 1000 * blend, -cap, cap);
    this.lastMoveT = now;
  };

  private onPointerUp = (e: PointerEvent): void => {
    const p = this.pointers.get(e.pointerId);
    if (!p) return;
    this.pointers.delete(e.pointerId);
    try {
      if (this.dom.hasPointerCapture(e.pointerId)) this.dom.releasePointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }

    if (this.pointers.size === 1) {
      // two fingers down to one: continue as a single-finger orbit from here
      const rest = [...this.pointers.values()][0];
      rest.startX = rest.x;
      rest.startY = rest.y;
      this.dragMode = this.flyMode ? 'look' : 'orbit';
      this.pinchDist = 0;
      return;
    }
    if (this.pointers.size > 0) return;

    // glide: carry the release velocity a little further
    const idle = performance.now() - this.lastMoveT;
    if (this.dragMode !== 'none' && idle < 60 && (Math.abs(this.velX) > 20 || Math.abs(this.velY) > 20)) {
      this.dragDelta(this.dragMode, this.velX * this.glide, this.velY * this.glide);
    }
    this.dragMode = 'none';
    this.velX = this.velY = 0;
  };

  private onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    this.interrupt();
    let d = e.deltaY;
    if (e.deltaMode === 1) d *= 16;
    else if (e.deltaMode === 2) d *= 400;
    // trackpad pinch arrives as a wheel event with ctrlKey set; it needs more gain
    const gain = e.ctrlKey ? 0.012 : 0.0022;
    const factor = Math.exp(THREE.MathUtils.clamp(d * gain * this.zoomSpeed, -1, 1));
    this.dolly(factor, e.clientX, e.clientY);
  };

  private onKeyDown = (e: KeyboardEvent): void => {
    if (this.isTyping()) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    this.keys.add(e.code);
    if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE'].includes(e.code)) {
      e.preventDefault();
      if (this.flight) this.interrupt();
    }
  };

  private onKeyUp = (e: KeyboardEvent): void => {
    this.keys.delete(e.code);
  };
}

function sphericalOffset(out: THREE.Vector3, theta: number, phi: number): THREE.Vector3 {
  const sp = Math.sin(phi);
  return out.set(sp * Math.sin(theta), Math.cos(phi), sp * Math.cos(theta));
}

function clonePose(p: Pose): Pose {
  return { target: p.target.clone(), distance: p.distance, theta: p.theta, phi: p.phi };
}

