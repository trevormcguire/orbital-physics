import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";

/**
 * Minimalist viewer (meters in API; visually normalized in the viewer).
 * - White background
 * - Bodies: circle sprites (fill by mass heatmap, black outline)
 * - Full path: ultra-thin black line
 * - Tail: stacked segments (thicker near body, thinner + more transparent back in time)
 * - Size ~ radius_km (sqrt compression)
 */

// -------------------------------------------------------------
// Scene & Renderer
// -------------------------------------------------------------
const canvas = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0xffffff, 1.0); // white

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 5000);
camera.position.set(0, -12, 7);
camera.up.set(0, 0, 1);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enablePan = true;
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0);

// -------------------------------------------------------------
// Helpers: circle sprite with black outline
// -------------------------------------------------------------
function makeCircleTexture(fillStyle = "#ff0000", size = 128, strokeWidth = 3) {
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
  ctx.strokeStyle = "#000000";
  ctx.stroke();

  const tex = new THREE.CanvasTexture(c);
  tex.anisotropy = 8;
  tex.needsUpdate = true;
  return tex;
}

// -------------------------------------------------------------
// Mass->Color (log) and Radius->Size mapping
// -------------------------------------------------------------
let MASS_MIN = 1, MASS_MAX = 1;
let RADIUS_MIN = 1, RADIUS_MAX = 1;

// Visual normalization (frozen after first fetch)
let framed = false;
let sceneCenter = new THREE.Vector3(0, 0, 0); // meters
let sceneScale = 1.0;                          // meters -> world units (viewer only)
const TARGET_RADIUS = 14.0;                    // world units for farthest body after scaling

function normalizeLog(x, lo, hi) {
  x = Math.max(x, 1e-12);
  lo = Math.max(lo, 1e-12);
  const lx = Math.log10(x), llo = Math.log10(lo), lhi = Math.log10(hi);
  return (lx - llo) / Math.max(1e-8, (lhi - llo));
}

function massToColor(mass) {
  const t = THREE.MathUtils.clamp(normalizeLog(mass, MASS_MIN, MASS_MAX), 0, 1);
  const hue = (1.0 - t) * 240; // blue->...->red
  return `hsl(${hue.toFixed(1)}, 90%, 45%)`;
}

function radiusToWorldScale(radius_km) {
  const t = THREE.MathUtils.clamp(
    Math.sqrt(Math.max(radius_km, 1) / Math.max(RADIUS_MAX, 1)),
    0, 1
  );
  const minSize = 0.06, maxSize = 0.8;
  return THREE.MathUtils.lerp(minSize, maxSize, t);
}

// -------------------------------------------------------------
// Body viz: sprite + full path + tapered trail
// -------------------------------------------------------------
class BodyViz {
  constructor(id, name, mass, radiusKm, initialPosMeters) {
    this.id = id;
    this.name = name;
    this.mass = mass;
    this.radiusKm = radiusKm;

    this.history = [];   // scaled, centered positions for ultra-thin path
    this.tail = [];      // scaled, centered positions for tapered trail
    this.maxHistory = 4000;
    this.maxTail = 160;

    // Sprite (circle with outline)
    const color = massToColor(mass);
    const tex = makeCircleTexture(color, 128, 3);
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: true });
    this.sprite = new THREE.Sprite(mat);
    scene.add(this.sprite);

    // Ultra-thin full path
    const thinGeom = new THREE.BufferGeometry();
    thinGeom.setAttribute("position", new THREE.Float32BufferAttribute([], 3));
    const thinMat = new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.9 });
    this.fullPath = new THREE.Line(thinGeom, thinMat);
    scene.add(this.fullPath);

    // Tapered trail segments using Line2 (pixel linewidth)
    this.segmentDefs = [
      { pct: 0.40, width: 6.0, opacity: 0.90 }, // newest, thickest
      { pct: 0.30, width: 4.0, opacity: 0.60 },
      { pct: 0.20, width: 2.0, opacity: 0.35 },
      { pct: 0.10, width: 1.5, opacity: 0.20 }, // oldest in tail
    ];
    this.segments = this.segmentDefs.map(() => {
      const g = new LineGeometry();
      const m = new LineMaterial({
        color: 0x000000,
        linewidth: 0.001,
        transparent: true,
        opacity: 1.0,
      });
      m.resolution.set(window.innerWidth, window.innerHeight);
      const l = new Line2(g, m);
      l.computeLineDistances();
      scene.add(l);
      return l;
    });

    this.update(initialPosMeters);
  }

  _updateSpriteScale() {
    const s = radiusToWorldScale(this.radiusKm);
    this.sprite.scale.set(s, s, 1);
  }

  _toWorld(posMeters) {
    // center in meters, then scale to viewer units
    return {
      x: (posMeters.x - sceneCenter.x) * sceneScale,
      y: (posMeters.y - sceneCenter.y) * sceneScale,
      z: (posMeters.z - sceneCenter.z) * sceneScale,
    };
  }

  update(posMeters) {
    const world = this._toWorld(posMeters);
    this.sprite.position.set(world.x, world.y, world.z);
    this._updateSpriteScale();

    // histories in WORLD units (not meters)
    this.history.push(world.x, world.y, world.z);
    if (this.history.length > this.maxHistory * 3) {
      this.history.splice(0, this.history.length - this.maxHistory * 3);
    }
    this.tail.push(world.x, world.y, world.z);
    if (this.tail.length > this.maxTail * 3) {
      this.tail.splice(0, this.tail.length - this.maxTail * 3);
    }

    // ultra-thin full path
    const thinAttr = this.fullPath.geometry.getAttribute("position");
    const thinArray = new Float32Array(this.history);
    if (!thinAttr || thinAttr.array.length !== thinArray.length) {
      this.fullPath.geometry.setAttribute("position", new THREE.Float32BufferAttribute(thinArray, 3));
    } else {
      thinAttr.array.set(thinArray);
      thinAttr.needsUpdate = true;
    }
    this.fullPath.geometry.setDrawRange(0, this.history.length / 3);

    // tapered segments
    const N = this.tail.length / 3;
    if (N >= 2) {
      let start = 0;
      let idx = 0;
      this.segmentDefs.forEach(def => {
        const segCount = Math.max(2, Math.floor(N * def.pct));
        const segStart = Math.max(0, N - (start + segCount));
        const segEnd = Math.max(0, N - start);
        start += segCount;

        const slice = this.tail.slice(segStart * 3, segEnd * 3);
        const line = this.segments[idx++];
        line.geometry.setPositions(slice);
        line.material.linewidth = def.width;
        line.material.opacity = def.opacity;
        line.material.needsUpdate = true;
      });
    }
  }

  resize(w, h) {
    this.segments.forEach(s => s.material.resolution.set(w, h));
  }
}

// -------------------------------------------------------------
// Data wiring
// -------------------------------------------------------------
const bodies = new Map(); // id -> BodyViz

function frameIfNeeded(data) {
  if (framed || !data?.bodies?.length) return;

  // Use server barycenter if present
  const bc = data.barycenter || { x: 0, y: 0, z: 0 };
  sceneCenter.set(bc.x, bc.y, bc.z);

  // One-time scale so the farthest body ~ TARGET_RADIUS in world units
  // Compute extent from barycenter (in meters)
  let maxR = 1.0;
  for (const b of data.bodies) {
    const dx = b.position.x - bc.x;
    const dy = b.position.y - bc.y;
    const dz = b.position.z - bc.z;
    const r = Math.sqrt(dx*dx + dy*dy + dz*dz);
    if (r > maxR) maxR = r;
  }
  sceneScale = TARGET_RADIUS / maxR;

  // Set a camera distance based on TARGET_RADIUS
  const dist = Math.max(8, TARGET_RADIUS * 2.4 + 2);
  camera.position.set(0, -dist, dist * 0.6);
  controls.target.set(0, 0, 0);
  controls.update();

  framed = true;
}

async function fetchState() {
  const res = await fetch("/api/state", { cache: "no-store" });
  if (!res.ok) return;
  const data = await res.json();

  MASS_MIN = data.mass_min;
  MASS_MAX = data.mass_max;
  RADIUS_MIN = data.radius_min;
  RADIUS_MAX = data.radius_max;

  frameIfNeeded(data);

  for (const b of data.bodies) {
    const id = b.id;
    const posMeters = new THREE.Vector3(b.position.x, b.position.y, b.position.z);

    if (!bodies.has(id)) {
      const viz = new BodyViz(id, b.name, b.mass_kg, b.radius_km, posMeters);
      bodies.set(id, viz);
    } else {
      const viz = bodies.get(id);
      viz.mass = b.mass_kg;
      viz.radiusKm = b.radius_km;
      viz.update(posMeters);
    }
  }
}

// -------------------------------------------------------------
// Render loop
// -------------------------------------------------------------
function onResize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  bodies.forEach(v => v.resize(w, h));
}
window.addEventListener("resize", onResize);

let lastFetch = 0;
const fetchIntervalMs = 1000 / 30; // ~30 Hz

function animate(t) {
  requestAnimationFrame(animate);
  if (t - lastFetch > fetchIntervalMs) {
    fetchState().catch(() => {});
    lastFetch = t;
  }
  controls.update();
  renderer.render(scene, camera);
}
onResize();
animate(0);
