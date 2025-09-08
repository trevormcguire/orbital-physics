import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

/**
 * Focus-zoom orbital viewer
 * - Keep simulation positions in METERS.
 * - Each frame, map meters -> world by a dynamic transform:
 *      world = (meters - dynamicOrigin) * (sceneScale * zoomBoost)
 * - dynamicOrigin smoothly follows the focused body (dblclick to set, Esc to clear).
 * - zoomBoost grows as you zoom in (camera gets closer to target), so local distances inflate
 *   and inner planets become discernible. Outer planets may go beyond far plane—intended.
 */

/* ----------------------- Config ----------------------- */
const SIZE_METHOD = "linear";           // 'linear' | 'sqrt' | 'log' | 'loglog'
const SIZE_RANGE = [0.1, 20.0];         // sprite world-size after view scale (base)
const TARGET_RADIUS = 500.0;            // world units for farthest body at initial frame

const HOVER_SCALE = 1.15;               // hovered sprite scale multiplier

// Flash/twinkle
const FLASH_DURATION_MS = 1000;
const FLASH_INTERVAL_MS = FLASH_DURATION_MS / 5;

const API_POLL_MS = 1000;
const TRAIL_MAX = 5000;
// const AU_METERS = 1.495978707e11;
// const BACKEND_SENDS_AU = true;

/* ----- Focus Zoom Tuning ----- */
const ENABLE_FOCUS_ZOOM = true;
// How aggressively local space inflates as you zoom in.
// Effective zoomBoost ~ (initialDistance / currentDistance) ** ZOOM_KAPPA
const ZOOM_KAPPA = 0.8;
// Maximum zoom boost multiplier (safety clamp)
const ZOOM_BOOST_MAX = 1e6;
// Smoothing for dynamic origin/scale [0..1], higher = snappier
const TRANSFORM_SMOOTH = 0.18;
// If no focus body, enable a mild boost when extremely close anyway? (kept off by default)
const BOOST_WITHOUT_FOCUS = false;

/* ----------------------- Scene ------------------------ */
const canvas = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0xffffff, 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 1e8);
camera.up.set(0, 0, 1);
camera.position.set(0, -40, 24);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
// Keep target at the world origin; the dynamic transform moves the universe, not the camera target.
controls.target.set(0, 0, 0);

/* --------------- Global transforms (meters -> world) --------------- */
let framed = false;
// Barycentric center (meters) and base scale (world/meter) found on first frame
let sceneCenter = new THREE.Vector3();
let sceneScale = 1.0;

// Dynamic transform (what actually drives rendering)
let dynamicOrigin = new THREE.Vector3();  // meters
let targetOrigin = new THREE.Vector3();   // meters
let dynamicScale = 1.0;                   // world/meter
let targetScale = 1.0;

// Track when transform changes to refresh trails efficiently
let transformVersion = 0;
let lastTransformKey = "";

// Camera distance baselining for zoom boost
let initialCamDistance = 60; // updated on frame
function currentCamDistance() { return camera.position.distanceTo(controls.target); }

/* ---------------------- Tooltip ---------------------- */
const tooltip = document.getElementById("tooltip");
const mouse = new THREE.Vector2();
const raycaster = new THREE.Raycaster();
let hovered = null; // Body currently hovered

/* -------------------- Helpers: circle sprite -------------------- */
function makeCircleTexture(fillStyle = "#bbbbbbff", size = 128, strokeWidth = 3) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, size, size);

  const r = (size - strokeWidth) / 2;
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, r, 0, Math.PI * 2);
  ctx.fillStyle = fillStyle;
  ctx.fill();

  ctx.lineWidth = strokeWidth;
  ctx.strokeStyle = "#000";
  ctx.stroke();

  const tex = new THREE.CanvasTexture(c);
  tex.anisotropy = 8;
  return tex;
}

// Soft radial burst texture
function makeBurstTexture(size = 256) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0.0, "rgba(11, 1, 21, 1)");
  g.addColorStop(0.5, "rgba(255,255,255,0.6)");
  g.addColorStop(1.0, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(c);
  tex.anisotropy = 8;
  return tex;
}
const FLASH_TEX = makeBurstTexture(256);

/* ------------------- Size scaling by radius ------------------- */
function makeRadiusScaler(radiiKm, method = SIZE_METHOD, outRange = SIZE_RANGE) {
  const tx = (r) => {
    const x = Math.max(r, 1e-6);
    if (method === "linear") return x;
    if (method === "sqrt") return Math.sqrt(x);
    if (method === "log") return Math.log(x);
    if (method === "loglog") return Math.log(Math.max(Math.log(x), 1e-6));
    return Math.log(x);
  };
  const vals = radiiKm.map(tx);
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const span = Math.max(hi - lo, 1e-12);
  return (r) => {
    const t = (tx(r) - lo) / span; // [0..1]
    return outRange[0] + t * (outRange[1] - outRange[0]);
  };
}

/* -------------------- Formatting helpers -------------------- */
const fmtInt = (n) => new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(n);
const fmt3 = (n) => new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 }).format(n);
const fmtSci = (n) => {
  if (!isFinite(n)) return String(n);
  const e = n.toExponential(3);
  return e.replace("+", "");
};
function metersToMkm(m) { return m / 1e9; }

function formatPeriod(sec) {
  if (sec == null || !isFinite(sec)) return "—";
  if (sec < 60) return `${Math.round(sec)} s`;
  if (sec < 3600) return `${(sec/60).toFixed(1)} min`;
  if (sec < 86400) return `${(sec/3600).toFixed(2)} h`;
  if (sec < 31557600) return `${(sec/86400).toFixed(2)} d`;
  return `${(sec/31557600).toFixed(3)} yr`;
}
function formatFg(g) {
  if (g == null || !isFinite(g)) return "—";
  return `${fmt3(g)} m/s²`;
}

/* ---------------- meters -> world transform helpers ---------------- */
function metersToWorldVec(mx, my, mz) {
  const sx = (mx - dynamicOrigin.x) * dynamicScale;
  const sy = (my - dynamicOrigin.y) * dynamicScale;
  const sz = (mz - dynamicOrigin.z) * dynamicScale;
  return new THREE.Vector3(sx, sy, sz);
}
function metersToWorldInPlace(mv, out) {
  out.set(
    (mv.x - dynamicOrigin.x) * dynamicScale,
    (mv.y - dynamicOrigin.y) * dynamicScale,
    (mv.z - dynamicOrigin.z) * dynamicScale
  );
  return out;
}

/* --------------------- Body visualization --------------------- */
class Body {
  constructor(id, name, radiusKm, massKg) {
    this.id = id;
    this.name = name;
    this.radiusKm = radiusKm;
    this.massKg = massKg;

    // Last known position in METERS
    this.lastMeters = new THREE.Vector3();

    // Rendering sprite
    const tex = makeCircleTexture("#bbb", 128, 3);
    this.baseMap = tex;
    this.material = new THREE.SpriteMaterial({ map: tex, transparent: true });
    this.sprite = new THREE.Sprite(this.material);
    this.sprite.userData.ref = this; // for picking
    scene.add(this.sprite);

    // Base (untransformed) sprite world size (we multiply by nothing else)
    this.baseScale = 0.2;

    // Interpolation state in METERS
    this._prevMeters = new THREE.Vector3();
    this._nextMeters = new THREE.Vector3();
    this._lerpStart = 0;
    this._lerpDur = 0;

    // Current displayed METERS (after lerp), used to re-project when transform changes
    this._currMeters = new THREE.Vector3();

    // Trail: store METERS samples; geometry holds WORLD-projected positions
    this.trailMeters = []; // Array<THREE.Vector3>
    const posArr = new Float32Array(TRAIL_MAX * 3);
    this.trailGeometry = new THREE.BufferGeometry();
    this.trailGeometry.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
    this.trailGeometry.setDrawRange(0, 0);
    this.trailMaterial = new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.18 });
    this.trailLine = new THREE.Line(this.trailGeometry, this.trailMaterial);
    scene.add(this.trailLine);

    // Flash color state
    this._flashTimeout = null;
    this._currentTempMap = null;

    // Cache last transform version applied to trail geometry
    this._lastTrailTransformVersion = -1;
  }

  setScale(worldSize) {
    this.baseScale = worldSize;
    this.sprite.scale.set(worldSize, worldSize, 1);
  }

//   setTrailFromHistory(historyArr) {
//     if (!historyArr || historyArr.length === 0) return;
//     this.trailMeters.length = 0;
//     for (let i = 0; i < historyArr.length && this.trailMeters.length < TRAIL_MAX; ++i) {
//       const e = historyArr[i];
//       let hx, hy, hz;
//       if (Array.isArray(e) && e.length >= 3) {
//         [hx, hy, hz] = e;
//       } else if (e && typeof e === "object" && "x" in e && "y" in e && "z" in e) {
//         ({ x: hx, y: hy, z: hz } = e);
//       } else {
//         continue;
//       }
//       this.trailMeters.push(new THREE.Vector3(hx, hy, hz));
//     }
//     if (this.trailMeters.length === 0) return;

//     const last = this.trailMeters[this.trailMeters.length - 1];
//     this.lastMeters.copy(last);
//     this._prevMeters.copy(last);
//     this._nextMeters.copy(last);
//     this._currMeters.copy(last);
//     // Immediate projection
//     const worldPos = metersToWorldVec(last.x, last.y, last.z);
//     this.sprite.position.copy(worldPos);
//     this._updateTrailGeometry(); // projects with current transform
//   }
  setTrailFromHistory(historyArr) {
    if (!historyArr || historyArr.length === 0) return;
    this.trailMeters.length = 0;

    // ensure we take the most recent TRAIL_MAX samples (not the earliest)
    const start = Math.max(0, historyArr.length - TRAIL_MAX);
    for (let i = start; i < historyArr.length; ++i) {
      const e = historyArr[i];
      let hx, hy, hz;
      if (Array.isArray(e) && e.length >= 3) {
        [hx, hy, hz] = e;
      } else if (e && typeof e === "object" && "x" in e && "y" in e && "z" in e) {
        ({ x: hx, y: hy, z: hz } = e);
      } else {
        continue;
      }
      this.trailMeters.push(new THREE.Vector3(hx, hy, hz));
    }
    if (this.trailMeters.length === 0) return;

    const last = this.trailMeters[this.trailMeters.length - 1];
    this.lastMeters.copy(last);
    this._prevMeters.copy(last);
    this._nextMeters.copy(last);
    this._currMeters.copy(last);
    // Immediate projection
    const worldPos = metersToWorldVec(last.x, last.y, last.z);
    this.sprite.position.copy(worldPos);
    this._updateTrailGeometry(); // projects with current transform
  }
  setImmediatePositionMeters(mx, my, mz) {
    this.lastMeters.set(mx, my, mz);
    this._prevMeters.set(mx, my, mz);
    this._nextMeters.set(mx, my, mz);
    this._currMeters.set(mx, my, mz);

    // initialize a short trail at current meters position
    this.trailMeters.length = 0;
    for (let i = 0; i < Math.min(8, TRAIL_MAX); ++i) this.trailMeters.push(new THREE.Vector3(mx, my, mz));

    // project to world
    metersToWorldInPlace(this._currMeters, this.sprite.position);
    this._updateTrailGeometry();
  }

  moveToMeters(mx, my, mz, durationMs = API_POLL_MS) {
    this.lastMeters.set(mx, my, mz);
    this._prevMeters.copy(this._currMeters);
    this._nextMeters.set(mx, my, mz);
    this._lerpStart = performance.now();
    this._lerpDur = Math.max(50, durationMs);
    // push a trail sample at the start (in meters)
    this._pushTrailSample(this._prevMeters.clone());
  }

  updateLerp(now) {
    if (this._lerpDur > 0 && this._lerpStart > 0) {
      const t = Math.min(1, (now - this._lerpStart) / this._lerpDur);
      this._currMeters.lerpVectors(this._prevMeters, this._nextMeters, t);

      // Occasionally add intermediate meter samples
      if (Math.random() < 0.02) this._pushTrailSample(this._currMeters.clone());

      if (t >= 1) {
        this._lerpStart = 0;
        this._pushTrailSample(this._nextMeters.clone());
      }
    } else {
      // No active lerp; ensure _currMeters == lastMeters
      this._currMeters.copy(this.lastMeters);
    }

    // Project current meters to world for rendering
    metersToWorldInPlace(this._currMeters, this.sprite.position);
  }

  refreshProjectionIfNeeded() {
    if (this._lastTrailTransformVersion !== transformVersion) {
      // Re-project the whole trail when transform changed
      this._updateTrailGeometry();
      // Also re-project sprite position (currMeters -> world)
      metersToWorldInPlace(this._currMeters, this.sprite.position);
      this._lastTrailTransformVersion = transformVersion;
    }
  }

  _pushTrailSample(mv) {
    this.trailMeters.push(mv);
    if (this.trailMeters.length > TRAIL_MAX) this.trailMeters.shift();
    this._updateTrailGeometry();
  }

  _updateTrailGeometry() {
    const drawCount = this.trailMeters.length;
    const attr = this.trailGeometry.getAttribute("position");
    const tmp = new THREE.Vector3();
    for (let i = 0; i < drawCount; ++i) {
      metersToWorldInPlace(this.trailMeters[i], tmp);
      attr.setXYZ(i, tmp.x, tmp.y, tmp.z);
    }
    // zero remaining
    for (let i = drawCount; i < TRAIL_MAX; ++i) attr.setXYZ(i, 0, 0, 0);
    attr.needsUpdate = true;
    this.trailGeometry.setDrawRange(0, drawCount);
    this._lastTrailTransformVersion = transformVersion;
  }

  setHovered(on) {
    if (on) {
      this.sprite.scale.set(this.baseScale * HOVER_SCALE, this.baseScale * HOVER_SCALE, 1);
      this.trailMaterial.color.set(0x000000);
      this.trailMaterial.opacity = 1.0;
    } else {
      this.sprite.scale.set(this.baseScale, this.baseScale, 1);
      this.trailMaterial.color.set(0x000000);
      this.trailMaterial.opacity = 0.20;
    }
    this.trailMaterial.needsUpdate = true;
  }

  flashColor(color, durationMs = FLASH_DURATION_MS) {
    if (this._flashTimeout) {
      clearTimeout(this._flashTimeout);
      this._flashTimeout = null;
    }
    if (this._currentTempMap) {
      try { this._currentTempMap.dispose(); } catch (e) {}
      this._currentTempMap = null;
    }

    const tmp = makeCircleTexture(color, 128, 3);
    this._currentTempMap = tmp;
    this.material.map = tmp;
    this.material.needsUpdate = true;

    this._flashTimeout = setTimeout(() => {
      this._flashTimeout = null;
      try {
        this.material.map = this.baseMap;
        this.material.needsUpdate = true;
      } catch (e) {}
      try { tmp.dispose(); } catch (e) {}
      this._currentTempMap = null;
    }, durationMs);
  }
}

/* ------------------------- State ------------------------- */
const bodies = new Map(); // id -> Body
let radiusScaler = (r) => 0.2;

// Focused body (dblclick to set). null means barycentric view.
let focusBodyId = null;
function getFocusedBody() {
  return focusBodyId ? bodies.get(focusBodyId) : null;
}

/* ---------------------- Layout / frame ---------------------- */
function frameIfNeeded(data) {
  if (framed || !data?.bodies?.length) return;

  // Barycenter if provided; otherwise compute
  let bc = data.barycenter;
  if (!bc) {
    let cx = 0, cy = 0, cz = 0;
    for (const b of data.bodies) { cx += b.position.x; cy += b.position.y; cz += b.position.z; }
    bc = { x: cx / data.bodies.length, y: cy / data.bodies.length, z: cz / data.bodies.length };
  }
  sceneCenter.set(bc.x, bc.y, bc.z);

  // Max distance from center (meters)
  let maxR = 1;
  for (const b of data.bodies) {
    const dx = b.position.x - bc.x;
    const dy = b.position.y - bc.y;
    const dz = b.position.z - bc.z;
    const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (r > maxR) maxR = r;
  }
  sceneScale = TARGET_RADIUS / maxR;

  // Auto-camera distance
  const dist = Math.max(8, TARGET_RADIUS * 2.4 + 2);
  camera.position.set(0, -dist, dist * 0.6);
  controls.target.set(0, 0, 0);
  controls.update();

  // Baseline camera distance for zoom boost
  initialCamDistance = currentCamDistance();

  // Start dynamic transform at barycenter
  dynamicOrigin.copy(sceneCenter);
  targetOrigin.copy(sceneCenter);
  dynamicScale = sceneScale;
  targetScale = sceneScale;

  // Build sprite radius scaler (km -> base world size)
  const radiiKm = data.bodies.map(b => b.radius_km);
  radiusScaler = makeRadiusScaler(radiiKm, SIZE_METHOD, SIZE_RANGE);

  framed = true;
}

/* ------------------------ Data ------------------------ */
function updateSimTimeFromPayload(payload) {
  const simTime = document.getElementById("simTime");
  const timeElapsed = document.getElementById("timeElapsed");
  if (!simTime | !timeElapsed) return;
    if (payload?.time_elapsed != null) {
      const days = (payload.time_elapsed / 86400).toFixed(2);
      timeElapsed.textContent = `elapsed: ${days} d`;
      simTime.textContent = `time: ${payload.sim_time_iso}`;

    }
}

async function fetchState() {
  const res = await fetch("/api/state", { cache: "no-store" });
  if (!res.ok) return;
  const data = await res.json();
  updateSimTimeFromPayload(data);
  frameIfNeeded(data);

  for (const b of data.bodies) {
    let body = bodies.get(b.id);
    if (!body) {
      body = new Body(b.id, b.name, b.radius_km, b.mass_kg);
      // update metadata
      body.T_seconds = b.T_seconds ?? body.T_seconds;
      body.fg_ms2 = b.fg_ms2 ?? body.fg_ms2;

      bodies.set(b.id, body);
      body.setScale(radiusScaler(b.radius_km));

      const hist = window.__BOOTSTRAP__?.history?.[b.name];
      if (hist && hist.length) {
        body.setTrailFromHistory(hist); // initializes sprite using current transform
        body.lastMeters.set(b.position.x, b.position.y, b.position.z);
        body._currMeters.copy(body.lastMeters);
      } else {
        body.setImmediatePositionMeters(b.position.x, b.position.y, b.position.z);
      }
    } else {
      // update metadata
      body.T_seconds = b.T_seconds ?? body.T_seconds;
      body.fg_ms2 = b.fg_ms2 ?? body.fg_ms2;

      body.radiusKm = b.radius_km;
      body.massKg = b.mass_kg;
      body.setScale(radiusScaler(b.radius_km));
      body.moveToMeters(b.position.x, b.position.y, b.position.z, API_POLL_MS);
    }
  }
}

function bootstrapInitial() {
  const boot = window.__BOOTSTRAP__;
  if (!boot || !boot.snapshot || !boot.snapshot.bodies) return;
  updateSimTimeFromPayload(boot.snapshot);
  frameIfNeeded(boot.snapshot);

  for (const b of boot.snapshot.bodies) {
    if (bodies.has(b.id)) continue;
    const body = new Body(b.id, b.name, b.radius_km, b.mass_kg);
    bodies.set(b.id, body);
    body.setScale(radiusScaler(b.radius_km));

    const hist = boot.history?.[b.name];
    if (hist && hist.length) {
      body.setTrailFromHistory(hist);
      body.lastMeters.set(b.position.x, b.position.y, b.position.z);
      body._currMeters.copy(body.lastMeters);
    } else {
      body.setImmediatePositionMeters(b.position.x, b.position.y, b.position.z);
    }
  }
  // temporarily set sol to the focus body if present
  const fb = Array.from(bodies.values()).find(x => x.name && x.name.toLowerCase() === "sol");
  if (fb) {
    focusBodyId = fb.id;
    targetOrigin.copy(fb.lastMeters);
    // update select UI if present
    try { focusSelect.value = focusBodyId; } catch (e) {}
    // optional flash to show selection
    flashSingleBody(fb);
  }
}

/* ----------------------- Hover & focus ----------------------- */
function pickBodyAtPointer(ev) {
  const rect = renderer.domElement.getBoundingClientRect();
  const x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  const y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  mouse.set(x, y);

  const sprites = [];
  bodies.forEach(b => sprites.push(b.sprite));
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(sprites, false);
  return hits.length ? hits[0].object.userData.ref : null;
}

function onPointerMove(ev) {
  const body = pickBodyAtPointer(ev);

  // Clear previous hover
  if (hovered && hovered !== body) hovered.setHovered(false);
  hovered = null;
  tooltip.style.transform = "translate(-9999px,-9999px)";

  if (body) {
    hovered = body;
    hovered.setHovered(true);

    // distance from barycenter (for info)
    const dx = body.lastMeters.x - sceneCenter.x;
    const dy = body.lastMeters.y - sceneCenter.y;
    const dz = body.lastMeters.z - sceneCenter.z;
    const dist_m = Math.sqrt(dx * dx + dy * dy + dz * dz);

    tooltip.innerHTML = `
      <span class="name">${body.name}</span>
      <span class="kv">r = ${fmtInt(body.radiusKm)} km</span> •
      <span class="kv">m = ${fmtSci(body.massKg)} kg</span> •
      <span class="kv">d = ${fmt3(metersToMkm(dist_m))} Mkm</span>
      <br/>
      <span class="kv">T = ${formatPeriod(body.T_seconds)}</span> •
      <span class="kv">g = ${formatFg(body.fg_ms2)}</span>
    `;
    const px = ev.clientX + 12;
    const py = ev.clientY + 12;
    tooltip.style.transform = `translate(${px}px, ${py}px)`;
  }
}
renderer.domElement.addEventListener("mousemove", onPointerMove);

// Double-click to focus on a body; Esc to clear focus
renderer.domElement.addEventListener("dblclick", (ev) => {
  const b = pickBodyAtPointer(ev);
  if (b) {
    focusBodyId = b.id;
    // Nudge target origin immediately for responsiveness
    targetOrigin.copy(b.lastMeters);
  }
});

window.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") {
    focusBodyId = null;
    targetOrigin.copy(sceneCenter);
  }
});

/* ----------------------- Flashing ----------------------- */
const activeFlashes = []; // { sprite, start, body, orig, prevDepthTest, prevRenderOrder }
let isFlashing = false;

function triggerFlash() {
  if (!bodies.size || isFlashing) return;
  isFlashing = true;

  let maxSize = 0;
  bodies.forEach(b => { if (b.baseScale > maxSize) maxSize = b.baseScale; });
  if (maxSize <= 0) { isFlashing = false; return; }

  const list = Array.from(bodies.values());
  list.forEach((body, i) => {
    setTimeout(() => {
      const orig = body.sprite.scale.clone();
      const prevDepthTest = body.sprite.material.depthTest ?? true;
      const prevRenderOrder = body.sprite.renderOrder ?? 0;
      body.sprite.material.depthTest = false;
      body.sprite.renderOrder = 998;
      body.sprite.scale.set(maxSize, maxSize, 1);
      body.flashColor("#000", FLASH_DURATION_MS);

      const mat = new THREE.SpriteMaterial({
        map: FLASH_TEX,
        transparent: true,
        opacity: 1.0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        depthTest: false
      });
      const s = new THREE.Sprite(mat);
      s.position.copy(body.sprite.position);
      s.scale.set(maxSize, maxSize, 1);
      s.renderOrder = 999;
      scene.add(s);
      activeFlashes.push({ sprite: s, start: performance.now(), body, orig, prevDepthTest, prevRenderOrder });

      setTimeout(() => {
        try {
          if (body.sprite) {
            body.sprite.scale.copy(orig);
            body.sprite.material.depthTest = prevDepthTest;
            body.sprite.renderOrder = prevRenderOrder;
          }
        } catch (e) {}
      }, FLASH_DURATION_MS);
    }, i * FLASH_INTERVAL_MS);
  });

  const totalMs = list.length * FLASH_INTERVAL_MS + FLASH_DURATION_MS;
  setTimeout(() => { isFlashing = false; }, totalMs + 20);
}

/* ----------------------- Transform updater ----------------------- */
function updateDynamicTransform() {
  // Decide target origin: follow focus body (live position), else barycenter
  const focus = getFocusedBody();
  if (focus) {
    targetOrigin.copy(focus._currMeters); // use interpolated/live meters
  } else {
    targetOrigin.copy(sceneCenter);
  }

  // Compute zoom boost from camera distance
  let boost = 1.0;
  if (ENABLE_FOCUS_ZOOM && (focus || BOOST_WITHOUT_FOCUS)) {
    const d0 = Math.max(1e-6, initialCamDistance);
    const d = Math.max(1e-6, currentCamDistance());
    const raw = Math.pow(d0 / d, ZOOM_KAPPA);
    boost = Math.min(Math.max(1.0, raw), ZOOM_BOOST_MAX);
  }

  targetScale = sceneScale * boost;

  // Smoothly approach targets
  // (exponential smoothing; per-frame blend)
  const a = TRANSFORM_SMOOTH;
  dynamicOrigin.lerp(targetOrigin, a);
  dynamicScale = dynamicScale + (targetScale - dynamicScale) * a;

  // Bump transform version if meaningful change
  const key = `${dynamicOrigin.x.toFixed(6)},${dynamicOrigin.y.toFixed(6)},${dynamicOrigin.z.toFixed(6)}|${dynamicScale.toExponential(6)}`;
  if (key !== lastTransformKey) {
    transformVersion++;
    lastTransformKey = key;
  }
}


/* -------------------- Focus selector UI -------------------- */
// const focusPanel = document.getElementById("focusPanel");
const focusSelect = document.getElementById("focusSelect");
const focusSearch = document.getElementById("focusSearch");
const clearFocusBtn = document.getElementById("clearFocusBtn");

function rebuildFocusList(filter = "") {
  const list = Array.from(bodies.values()).sort((a, b) => a.name.localeCompare(b.name));
  const f = filter.trim().toLowerCase();
  focusSelect.innerHTML = "";
  for (const b of list) {
    if (f && !b.name.toLowerCase().includes(f)) continue;
    const opt = document.createElement("option");
    opt.value = b.id;
    opt.textContent = b.name;
    focusSelect.appendChild(opt);
  }
  // keep select synced with current focus
  if (focusBodyId) {
    focusSelect.value = focusBodyId;
  } else {
    focusSelect.selectedIndex = -1;
  }
}

focusSelect.addEventListener("change", () => {
  const id = focusSelect.value || null;
  focusBodyId = id;
  const fb = getFocusedBody();
  if (fb) {
    targetOrigin.copy(fb.lastMeters);
    flashSingleBody(fb);
  } else {
    targetOrigin.copy(sceneCenter);
  }
});
function flashSingleBody(body) {
  if (!body) return;
  try {
    const orig = body.sprite.scale.clone();
    const prevDepthTest = body.sprite.material.depthTest ?? true;
    const prevRenderOrder = body.sprite.renderOrder ?? 0;

    // scale for visual flash (use baseScale so single-body flash is proportional)
    // const size = Math.max(body.baseScale, 1.0);
    let maxSize = 0;
    bodies.forEach(b => { if (b.baseScale > maxSize) maxSize = b.baseScale; });
    if (maxSize <= 0) { isFlashing = false; return; }
    let size = maxSize;

    body.sprite.material.depthTest = false;
    body.sprite.renderOrder = 998;
    body.sprite.scale.set(size, size, 1);
    body.flashColor("#000", FLASH_DURATION_MS * 1.5);

    const mat = new THREE.SpriteMaterial({
      map: FLASH_TEX,
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: false
    });
    const s = new THREE.Sprite(mat);
    s.position.copy(body.sprite.position);
    s.scale.set(size, size, 1);
    s.renderOrder = 999;
    scene.add(s);

    activeFlashes.push({ sprite: s, start: performance.now(), body, orig, prevDepthTest, prevRenderOrder });

    // restore original sprite properties after duration
    setTimeout(() => {
      try {
        if (body.sprite) {
          body.sprite.scale.copy(orig);
          body.sprite.material.depthTest = prevDepthTest;
          body.sprite.renderOrder = prevRenderOrder;
        }
      } catch (e) {}
    }, FLASH_DURATION_MS);
  } catch (e) {}
}

focusSearch.addEventListener("input", () => {
  rebuildFocusList(focusSearch.value);
});

clearFocusBtn.addEventListener("click", () => {
  focusBodyId = null;
  targetOrigin.copy(sceneCenter);
  focusSelect.selectedIndex = -1;
  focusSearch.value = "";
  rebuildFocusList();
});

// Ensure the list is populated after bootstrap and after each fetch update
const orig_bootstrapInitial = bootstrapInitial;
bootstrapInitial = function() {
  orig_bootstrapInitial();
  rebuildFocusList();
};

const orig_fetchState = fetchState;
fetchState = async function() {
  await orig_fetchState();
  rebuildFocusList(focusSearch.value);
};

/* ----------------------- Render loop ----------------------- */
function onResize() {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
window.addEventListener("resize", onResize);

let lastFetch = 0;

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();

  if (now - lastFetch > API_POLL_MS) {
    fetchState().catch(() => {});
    lastFetch = now;
  }

  // Update focus-zoom transform before projecting bodies
  updateDynamicTransform();

  // Update active flashes
  for (let i = activeFlashes.length - 1; i >= 0; --i) {
    const item = activeFlashes[i];
    const dt = now - item.start;
    const f = Math.max(0, 1 - dt / FLASH_DURATION_MS);
    item.sprite.material.opacity = f;
    if (f <= 0) {
      scene.remove(item.sprite);
      if (item.sprite.material.map && item.sprite.material.map !== FLASH_TEX) {
        item.sprite.material.map.dispose();
      }
      item.sprite.material.dispose();
      activeFlashes.splice(i, 1);
    }
  }

  // Advance body interpolations (in meters) and project to world
  bodies.forEach(b => b.updateLerp(now));
  // If transform changed, re-project trails in one pass
  bodies.forEach(b => b.refreshProjectionIfNeeded());

  controls.update();
  renderer.render(scene, camera);
}

/* ----------------------- Boot ----------------------- */
onResize();
bootstrapInitial();
lastFetch = performance.now();
animate();

const flashBtn = document.getElementById("flashBtn");
if (flashBtn) {
  flashBtn.addEventListener("click", () => { triggerFlash(); });
}

window.onload = () => {triggerFlash();};