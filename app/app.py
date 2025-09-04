from __future__ import annotations
import os
import threading
import time
import math
import numpy as np

from flask import Flask, jsonify, render_template

from core.datasets import solar_keplerian_elements, solar_physical_properties
from core.engine import SimulationEngine, run_simulation
from core.sol import OrbitalElements, elements_to_state, G, DAY
from core.physics import Object, Coordinates, ObjectCollection


def generate_engine(
    dt: float = None,
    include_sun: bool = True,
):
    """Build Sun+planets from core.datasets keplerian table and render a video."""

    # get dataset and convert to meters/radians (Dataset defaults do this)
    ds = solar_keplerian_elements().convert_types(angle_unit="radians")
    rows = ds.values()  # list[dict] with numeric 'a' (m), angles (rad)

    phys = solar_physical_properties().convert_types(mass_unit="kilograms", distance_unit="meters").values()
    phys = {row["name"].lower(): row for row in phys}

    # choose timestep
    dt = DAY if dt is None else dt
    mu_sun = G * phys["sun"]["mass"]
    default_mass = phys["earth"]["mass"]
    default_radius = phys["earth"]["radius"]

    bodies = []
    if include_sun:
        sun = Object(
            mass=phys["sun"]["mass"],
            radius=phys["sun"]["radius"],
            velocity=np.zeros(3),
            coordinates=Coordinates(0.0, 0.0, 0.0),
            name="Sol"
        )
        bodies.append(sun)

    for rec in rows:
        name = rec.get("name")
        a = float(rec["a"])
        e = float(rec["e"])
        I = float(rec["I"])
        L = float(rec["L"])  # mean longitude (defined as L = Ω + ω + M , or L = ϖ + M)
        varpi = float(rec["long.peri"])  # longitude of periapsis (closest point, defied as ϖ = Ω + ω)
        O = float(rec["long.node"])  # longitude of ascending node (defied as Ω)

        # Compute mean anomaly and argument of periapsis consistent with core/sol.jpl logic
        M = (L - varpi) % (2 * math.pi)  # mean anomaly
        omega = (varpi - O) % (2 * math.pi)  # argument of periapsis (ω)

        el = OrbitalElements(a=a, e=e, i=I, Omega=O, omega=omega, M=M)
        # radius (dist from foci), velocity in inertial frame
        r, v = elements_to_state(mu_sun, el)

        mass = phys[name.lower()]["mass"] if name.lower() in phys else default_mass
        radius = phys[name.lower()]["radius"] if name.lower() in phys else default_radius

        bodies.append(
            Object(
                mass=mass,
                radius=radius,
                velocity=v.astype(np.float64),
                coordinates=Coordinates(*r.tolist()),
                name=name
            )
        )

    collection = ObjectCollection(bodies)
    engine = SimulationEngine(collection, dt=dt, softening=1e6, restitution=1.0)

    return engine


app = Flask(__name__)

# 1-hour timestep; softening to avoid singularities if needed
# engine = SimulationEngine(objects=collection, dt=3600.0, softening=1e6, restitution=1.0)
engine = generate_engine(dt=3600.0, include_sun=True)  # each frame is 1 hour
AU_METERS = 1.495978707e11
WORLD_SCALE = 1.0 / AU_METERS  # world units == AU

STOP_SIMULATION = False
SIM_FPS = 10.0  # tick rate of the engine loop (wall clock)
def engine_loop():
    t_target = 1.0 / SIM_FPS
    while not STOP_SIMULATION:
        t0 = time.time()
        engine.step()
        # simple pacing
        time.sleep(max(0.0, t_target - (time.time() - t0)))

thread = threading.Thread(target=engine_loop, daemon=True)
thread.start()

@app.route("/")
def index():
    # jsonify and send engine.history
    return render_template("index.html")

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

    return jsonify({
        "bodies": bodies,
        "mass_min": min(masses),
        "mass_max": max(masses),
        "radius_min": min(radii_km),
        "radius_max": max(radii_km),
    })
