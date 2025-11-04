from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone, timedelta

import numpy as np
from flask import Flask, jsonify, render_template

from core.constants import J2000_JD, JULIAN_DAY
from core.datasets import solar_system_v2, System
from core.engine import SimulationEngine, run_simulation
from core.physics import Object, Coordinates, ObjectCollection



def generate_solar_system(
    dt: float,
    max_hist: int = None,
    cache_fp: str = "solar_system_cache.jsonl",
    cache_every_n: int = 600,
):
    """Generate a SimulationEngine with the solar system bodies."""
    system: System = solar_system_v2(moons=True)
    system.standardize_units(
        mass_unit="kilograms",
        distance_unit="meters",
        angle_unit="radians",
        time_unit="seconds"
    )
    bodies = []
    for body in system:
        r, v = body.get_state()
        if body.parent is not None:
            parent_position, parent_velocity = body.parent.get_state()
            r = np.array(parent_position) + np.array(r)  # Adjust position relative to parent
            v = np.array(parent_velocity) + np.array(v)  # Adjust velocity relative to parent
        bodies.append(
            Object(
                mass=body.mass.value,
                radius=body.radius.value,
                velocity=np.array(v, dtype=np.float64),
                coordinates=Coordinates(*r),
                name=body.name
            )
        )
    collection = ObjectCollection(bodies)
    engine = SimulationEngine(
        collection,
        dt=dt,
        softening=1e6,
        restitution=1.0,
        max_hist=max_hist,
        cache=True,
        cache_fp=cache_fp,
        cache_every_n=cache_every_n
    )
    engine.body_map = {b.name: b for b in system.bodies}
    engine.system = system
    return engine

with open("config.json", "r") as f:
    CONFIG = json.loads(f.read())

# INTERVAL = 3600.  # 1 hour
INTERVAL = float(os.getenv("SIM_INTERVAL", 1800.))  # default 1 hour
INITIAL_STEPS = int(os.getenv("SIM_INITIAL_STEPS", 5000))  # hours to warm up with
MAX_HISTORY = int(os.getenv("SIM_MAX_HISTORY", 7000))
CACHE_FP = os.getenv("CACHE_FP")
if CACHE_FP is None:
    raise EnvironmentError("CACHE_FP environment variable not set")
CACHE_EVERY_N = int(os.getenv("CACHE_EVERY_N", "600"))  # ~once/min at 10 Hz
# build engine
engine = generate_solar_system(
    dt=INTERVAL,
    max_hist=MAX_HISTORY,
    cache_fp=CACHE_FP,
    cache_every_n=CACHE_EVERY_N
)  # each frame is 1 hour
epoch_ts = (J2000_JD - 2440587.5) * JULIAN_DAY  # seconds since Unix epoch
engine.sim_epoch = datetime.fromtimestamp(epoch_ts, tz=timezone.utc)
engine.sim_epoch_jd = float(J2000_JD)

# start with some history
print("Warming up simulation...")
run_simulation(engine, steps=INITIAL_STEPS, print_every=100)  # 1 month of history
print("Done.")

app = Flask(__name__)


AU_METERS = 1.495978707e11
# WORLD_SCALE = 1.0 / AU_METERS  # world units == AU
WORLD_SCALE = 1.
STOP_SIMULATION = False
SIM_FPS = 10.0  # tick rate of the engine loop (wall clock)

# engine_lock = threading.Lock()
def engine_loop():
    t_target = 1.0 / SIM_FPS
    while not STOP_SIMULATION:
        # with engine_lock:
        t0 = time.time()
        engine.step()
        # simple pacing
        time.sleep(max(0.0, t_target - (time.time() - t0)))

thread = threading.Thread(target=engine_loop, daemon=True)
thread.start()

def get_bodies():
    def unwrap_unit(val):
        # return a plain float when possible
        try:
            if hasattr(val, "value"):
                return float(val.value)
            return float(val)
        except Exception:
            return None
    bodies = []
    masses = []
    radii_km = []

    for obj in engine.objects:
        pos_m = obj.position()               # meters
        pos_world = (pos_m * WORLD_SCALE)    # AU for the viewer
        r_km = float(obj.radius) / 1000.0

        obj_body = engine.body_map.get(obj.name)

        bodies.append({
            "id": obj.uuid,
            "name": obj.name,
            "mass_kg": float(obj.mass),
            "radius_km": r_km,
            "T_seconds": unwrap_unit(obj_body.T),
            "fg_ms2": obj_body.fg,
            "position": {
                "x": float(pos_world[0]),
                "y": float(pos_world[1]),
                "z": float(pos_world[2]),
            }
        })
        masses.append(float(obj.mass))
        radii_km.append(r_km)

    if not masses:
        masses = [1.0]
    if not radii_km:
        radii_km = [1.0]

    sim_jd = engine.sim_epoch_jd + (engine.time_elapsed / JULIAN_DAY)
    sim_iso = (engine.sim_epoch + timedelta(seconds=engine.time_elapsed)).isoformat()
    return {
        "bodies": bodies,
        "mass_min": min(masses),
        "mass_max": max(masses),
        "radius_min": min(radii_km),
        "radius_max": max(radii_km),
        "time_elapsed": engine.time_elapsed,
        "sim_time_jd": sim_jd,
        "sim_time_iso": sim_iso,
    }

@app.route("/")
def index():
    # jsonify and send engine.history
    raw_hist = engine.named_history(limit=5000)  # { name: [ [x,y,z], ... ] } (meters)
    world_hist = {}
    for name, pts in raw_hist.items():
        converted = []
        for p in pts:
            x, y, z = p[0], p[1], p[2]
            converted.append([x * WORLD_SCALE, y * WORLD_SCALE, z * WORLD_SCALE])
        world_hist[name] = converted
    return render_template(
        "index.html",
        initial_state=world_hist,
        bodies=get_bodies(),
        version=CONFIG["version"],
        system="sol"
    )

@app.route("/api/state")
def api_state():
    """
    Exposes current positions & properties for all objects.

    Output fields:
      - id: object uuid
      - name
      - mass_kg
      - radius_km  (converted from meters)
      - position (x,y,z) in WORLD units (AU here)
    Also returns mass/radius min/max for color/size scaling.
    """
    return jsonify(get_bodies())


@app.get("/health")
def health():
    """Kubernetes liveness/readiness probe endpoint."""
    return jsonify(status="ok"), 200

