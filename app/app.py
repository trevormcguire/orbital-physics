from __future__ import annotations

import threading
import time
import numpy as np
from datetime import datetime, timezone, timedelta

from flask import Flask, jsonify, render_template

from core.engine import SimulationEngine, run_simulation
from core.sol import OrbitalElements, elements_to_state, J2000_JD, JULIAN_DAY
from core.physics import Object, Coordinates, ObjectCollection
from core.units import Unit



def generate_engine_v2(
    dt: float,
    max_hist: int = None,
    include_sun: bool = True,
):
    from core.datasets import solar_system, G

    data = solar_system(moons=False).convert_types(
        mass_unit="kilograms",
        distance_unit="meters",
        angle_unit="radians"
    )

    sol: dict[str, Unit] = data.pop("Sol")
    mu_sun = G * sol["mass"].value
    bodies = []
    if include_sun:
        sun = Object(
            mass=sol["mass"].value,
            radius=sol["radius"].value,
            velocity=np.zeros(3),
            coordinates=Coordinates(0.0, 0.0, 0.0),
            name="Sol"
        )
        bodies.append(sun)
    for body in data.data:
        el = OrbitalElements(
            a=float(body["a"].value),
            e=float(body["e"]),
            i=float(body["I"].value),
            Omega=float(body["long.node"].value),
            omega=float(body["arg.peri"].value),
            M=float(body["M"].value)
        )
        # radius (dist from foci), velocity in inertial frame
        r, v = elements_to_state(mu_sun, el)
        bodies.append(
            Object(
                mass=body["mass"].value,
                radius=body["radius"].value,
                velocity=v.astype(np.float64),
                coordinates=Coordinates(*r.tolist()),
                name=body["name"]
            )
        )
    collection = ObjectCollection(bodies)
    return SimulationEngine(collection, dt=dt, softening=1e6, restitution=1.0, max_hist=max_hist)


INTERVAL = 3600.  # 1 hour
INITIAL_STEPS = 5000  # hours to warm up with
MAX_HISTORY = 50000
# 1-hour timestep; softening to avoid singularities if needed
engine = generate_engine_v2(dt=INTERVAL, max_hist=MAX_HISTORY, include_sun=True)  # each frame is 1 hour
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
    bodies = []
    masses = []
    radii_km = []

    for obj in engine.objects:
        pos_m = obj.position()               # meters
        pos_world = (pos_m * WORLD_SCALE)    # AU for the viewer
        r_km = float(obj.radius) / 1000.0

        bodies.append({
            "id": obj.uuid,
            "name": obj.name,
            "mass_kg": float(obj.mass),
            "radius_km": r_km,
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
    return render_template("index.html", initial_state=world_hist, bodies=get_bodies())

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
