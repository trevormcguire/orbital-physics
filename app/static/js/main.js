import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

/**
 * Minimal viewer:
 * - White background
 * - Each body = circle sprite (black outline), size ~ radius via transform
 * - Positions come in METERS; we do one-time centering + scale for visibility
 * - Hover tooltip shows name, radius, mass, distance to barycenter (meters->Mkm)
 */

/* ----------------------- Config ----------------------- */
const SIZE_METHOD = 'log';              // 'linear' | 'sqrt' | 'log' | 'loglog'
const SIZE_RANGE = [0.1, 10.0];         // sprite world-size after view scale
const TARGET_RADIUS = 100.0;             // world units for farthest body after scaling

const POLL_HZ = 20;
const HOVER_SCALE = 1.15;               // hovered sprite scale multiplier

// Flash config: same world-size for every flash
const FLASH_SIZE = 10.0;                 // world units for flash sprite scale
const FLASH_DURATION_MS = 1000;          // duration of flash fade (ms)
const FLASH_INTERVAL_MS = FLASH_DURATION_MS / 5;

/* ----------------------- Scene ------------------------ */
const canvas = document.getElementById('scene');
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
controls.target.set(0, 0, 0);

/* --------------- View normalization (meters -> world) --------------- */
let framed = false;
let sceneCenter = new THREE.Vector3(); // meters
let sceneScale = 1.0;                  // world per meter (viewer-only scaling)

/* ---------------------- Tooltip ---------------------- */
const tooltip = document.getElementById('tooltip');
const mouse = new THREE.Vector2();
const raycaster = new THREE.Raycaster();
let hovered = null; // Body currently hovered

/* -------------------- Helpers: circle sprite -------------------- */
function makeCircleTexture(fillStyle = "#bbbbbbff", size = 128, strokeWidth = 3) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
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

// Soft radial "burst of light" (white core -> transparent edge), good for additive blending
function makeBurstTexture(size = 256) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
  g.addColorStop(0.0, 'rgba(11, 1, 21, 1)');
  g.addColorStop(0.5, 'rgba(255,255,255,0.6)');
  g.addColorStop(1.0, 'rgba(255,255,255,0)');
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
    if (method === 'linear') return x;
    if (method === 'sqrt')   return Math.sqrt(x);
    if (method === 'log')    return Math.log(x);
    if (method === 'loglog') return Math.log(Math.max(Math.log(x), 1e-6));
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
const fmtInt = (n) => new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(n);
const fmt3 = (n) => new Intl.NumberFormat('en-US', { maximumFractionDigits: 3 }).format(n);
const fmtSci = (n) => {
  if (!isFinite(n)) return String(n);
  const e = n.toExponential(3);
  // make it a bit friendlier (e.g., 4.868e24)
  return e.replace('+', '');
};
function metersToMkm(m) { return m / 1e9; } // million km

/* --------------------- Body visualization --------------------- */
class Body {
  constructor(id, name, radiusKm, massKg) {
    this.id = id;
    this.name = name;
    this.radiusKm = radiusKm;
    this.massKg = massKg;

    this.lastMeters = new THREE.Vector3();

    const tex = makeCircleTexture("#bbb", 128, 3);
    this.material = new THREE.SpriteMaterial({ map: tex, transparent: true });
    this.sprite = new THREE.Sprite(this.material);
    this.sprite.userData.ref = this; // for picking
    scene.add(this.sprite);

    this.baseScale = 0.2;

    // flash color state
    this._flashTimeout = null;
    this._currentTempMap = null;
  }
  setScale(worldSize) {
    this.baseScale = worldSize;
    this.sprite.scale.set(worldSize, worldSize, 1);
  }
  setPositionMeters(mx, my, mz) {
    this.lastMeters.set(mx, my, mz);
    // center and scale
    const wx = (mx - sceneCenter.x) * sceneScale;
    const wy = (my - sceneCenter.y) * sceneScale;
    const wz = (mz - sceneCenter.z) * sceneScale;
    this.sprite.position.set(wx, wy, wz);
}
  setHovered(on) {
    if (on) {
      this.sprite.scale.set(this.baseScale * HOVER_SCALE, this.baseScale * HOVER_SCALE, 1);
    } else {
      this.sprite.scale.set(this.baseScale, this.baseScale, 1);
    }
  }
  // Temporarily tint / recolor the body's sprite by swapping its texture.
  // color: CSS color string (e.g. "#000" or "rgba(0,0,0,1)")
  // durationMs: how long before restoring (default: FLASH_DURATION_MS)
  flashColor(color, durationMs = FLASH_DURATION_MS) {
    // clear any pending restore
    if (this._flashTimeout) {
      clearTimeout(this._flashTimeout);
      this._flashTimeout = null;
    }
    // dispose any previous temp map
    if (this._currentTempMap) {
      try { this._currentTempMap.dispose(); } catch (e) {}
      this._currentTempMap = null;
    }

    // create a temporary circle texture in the requested color
    const tmp = makeCircleTexture(color, 128, 3);
    this._currentTempMap = tmp;
    this.material.map = tmp;
    this.material.needsUpdate = true;

    // schedule restore
    this._flashTimeout = setTimeout(() => {
      this._flashTimeout = null;
      // restore original map (guard if removed)
      try {
        this.material.map = this.baseMap;
        this.material.needsUpdate = true;
      } catch (e) {}
      // dispose temp map
      try { tmp.dispose(); } catch (e) {}
      this._currentTempMap = null;
    }, durationMs);
  }
}

/* ------------------------- State ------------------------- */
const bodies = new Map(); // id -> Body
let radiusScaler = (r) => 0.2;

/* ---------------------- Layout / frame ---------------------- */
function frameIfNeeded(data) {
  if (framed || !data?.bodies?.length) return;

  // Barycenter if provided; otherwise compute here
  let bc = data.barycenter;
  if (!bc) {
    let cx = 0, cy = 0, cz = 0;
    for (const b of data.bodies) { cx += b.position.x; cy += b.position.y; cz += b.position.z; }
    bc = { x: cx / data.bodies.length, y: cy / data.bodies.length, z: cz / data.bodies.length };
  }
  sceneCenter.set(bc.x, bc.y, bc.z);

  // Max distance from center (in meters)
  let maxR = 1;
  for (const b of data.bodies) {
    const dx = b.position.x - bc.x;
    const dy = b.position.y - bc.y;
    const dz = b.position.z - bc.z;
    const r = Math.sqrt(dx*dx + dy*dy + dz*dz);
    if (r > maxR) maxR = r;
  }
  sceneScale = TARGET_RADIUS / maxR;

  // Auto-camera distance
  const dist = Math.max(8, TARGET_RADIUS * 2.4 + 2);
  camera.position.set(0, -dist, dist * 0.6);
  controls.target.set(0, 0, 0);
  controls.update();

  // Build a scaler for radii (km) -> world size
  const radiiKm = data.bodies.map(b => b.radius_km);
  radiusScaler = makeRadiusScaler(radiiKm, SIZE_METHOD, SIZE_RANGE);

  framed = true;
}

/* ------------------------ Data loop ------------------------ */
async function fetchState() {
  const res = await fetch("/api/state", { cache: "no-store" });
  if (!res.ok) return;
  const data = await res.json();

  frameIfNeeded(data);

  // Update / create bodies
  for (const b of data.bodies) {
    let body = bodies.get(b.id);
    if (!body) {
      body = new Body(b.id, b.name, b.radius_km, b.mass_kg);
      bodies.set(b.id, body);
    } else {
      body.radiusKm = b.radius_km;
      body.massKg = b.mass_kg;
    }
    body.setScale(radiusScaler(b.radius_km));
    body.setPositionMeters(b.position.x, b.position.y, b.position.z);
  }
}

/* ----------------------- Hover picking ----------------------- */
function onPointerMove(ev) {
  const rect = renderer.domElement.getBoundingClientRect();
  const x = ( (ev.clientX - rect.left) / rect.width ) * 2 - 1;
  const y = -( (ev.clientY - rect.top) / rect.height ) * 2 + 1;
  mouse.set(x, y);

  // Raycast against all sprites
  const sprites = [];
  bodies.forEach(b => sprites.push(b.sprite));
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(sprites, false);

  // Clear previous hover
  if (hovered) {
    hovered.setHovered(false);
    hovered = null;
  }
  tooltip.style.transform = 'translate(-9999px,-9999px)';

  if (hits.length) {
    const hit = hits[0].object;
    const body = hit.userData.ref;
    hovered = body;
    hovered.setHovered(true);

    // Distance from barycenter (in meters)
    const dx = body.lastMeters.x - sceneCenter.x;
    const dy = body.lastMeters.y - sceneCenter.y;
    const dz = body.lastMeters.z - sceneCenter.z;
    const dist_m = Math.sqrt(dx*dx + dy*dy + dz*dz);

    // Tooltip text (minimal)
    tooltip.innerHTML = `
      <span class="name">${body.name}</span>
      <span class="kv">r = ${fmtInt(body.radiusKm)} km</span> •
      <span class="kv">m = ${fmtSci(body.massKg)} kg</span> •
      <span class="kv">d = ${fmt3(metersToMkm(dist_m))} Mkm</span>
    `;

    // Place near cursor with small offset
    const px = ev.clientX + 12;
    const py = ev.clientY + 12;
    tooltip.style.transform = `translate(${px}px, ${py}px)`;
  }
}

renderer.domElement.addEventListener('mousemove', onPointerMove);

/* ----------------------- Flashing ----------------------- */
const activeFlashes = []; // { sprite: THREE.Sprite, start: number, body?: Body, origScale?: THREE.Vector3 }
let isFlashing = false;

// Trigger a flash from every body (one after another).
// During a body's turn we temporarily set its sprite scale to the largest object's size
// so the twinkle appears as a supernova (same world-size for every twinkle).
function triggerFlash() {
  if (!bodies.size || isFlashing) return;
  isFlashing = true;

  // determine largest visible object size (world units)
  let maxSize = 0;
  bodies.forEach(b => { if (b.baseScale > maxSize) maxSize = b.baseScale; });
  if (maxSize <= 0) {
    isFlashing = false;
    return;
  }

  const list = Array.from(bodies.values());
  list.forEach((body, i) => {
    setTimeout(() => {
      // store original scale so we can restore it
      const orig = body.sprite.scale.clone();
      // temporarily render the body on top
      const prevDepthTest = body.sprite.material.depthTest ?? true;
      const prevRenderOrder = body.sprite.renderOrder ?? 0;
      body.sprite.material.depthTest = false;
      body.sprite.renderOrder = 998;
      // resize body to max size so the "supernova" covers the same world area
      body.sprite.scale.set(maxSize, maxSize, 1);
      body.flashColor("#000", FLASH_DURATION_MS);

      // create a burst sprite drawn on top
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

      // restore the body's scale / material state after flash duration
      setTimeout(() => {
        // body might have been removed; guard
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

  // clear flashing flag after entire sequence completes
  const totalMs = list.length * FLASH_INTERVAL_MS + FLASH_DURATION_MS;
  setTimeout(() => { isFlashing = false; }, totalMs + 20);
}
/* ----------------------- Render loop ----------------------- */
function onResize() {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
window.addEventListener('resize', onResize);

let lastFetch = 0;
const fetchIntervalMs = 1000 / POLL_HZ;

function animate(t) {
  requestAnimationFrame(animate);
  if (t - lastFetch > fetchIntervalMs) {
    fetchState().catch(() => {});
    lastFetch = t;
  }
  // update active flashes
  const now = performance.now();
  for (let i = activeFlashes.length - 1; i >= 0; --i) {
    const item = activeFlashes[i];
    const dt = now - item.start;
    const f = Math.max(0, 1 - dt / FLASH_DURATION_MS);
    item.sprite.material.opacity = f;
    if (f <= 0) {
      scene.remove(item.sprite);
      // if (item.sprite.material.map) item.sprite.material.map.dispose();
      // only dispose per-sprite maps if they do not reference the shared FLASH_TEX
      if (item.sprite.material.map && item.sprite.material.map !== FLASH_TEX) {
        item.sprite.material.map.dispose();
      }
      // item.sprite.material.dispose();
      item.sprite.material.dispose(); // keep shared FLASH_TEX alive
      activeFlashes.splice(i, 1);
    }
  }
//   for (let i = activeFlashes.length - 1; i >= 0; --i) {
//     const item = activeFlashes[i];
//     const p = (now - item.start) / FLASH_DURATION_MS; // 0..1
//     if (p >= 1) {
//       scene.remove(item.sprite);
//       if (item.sprite.material.map) item.sprite.material.map.dispose();
//       item.sprite.material.dispose();
//       activeFlashes.splice(i, 1);
//       continue;
//     }
//     // Waxing to the SAME size at mid-duration, then waning.
//     // At p = 0.5, sin(pi * 0.5) = 1 -> size == largest object's size.
//     const amp = Math.sin(Math.PI * p); // 0 -> 1 -> 0
//     const s = Math.max(amp * item.maxSize, 1e-3);
//     item.sprite.scale.set(s, s, 1);
//     item.sprite.material.opacity = amp;
// }
  // 
  controls.update();
  renderer.render(scene, camera);
}
onResize();
animate(0);
const flashBtn = document.getElementById('flashBtn');
if (flashBtn) {
  flashBtn.addEventListener('click', () => {
    triggerFlash();
  });
}