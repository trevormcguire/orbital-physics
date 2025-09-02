from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np

from core.physics import Coordinates, Object, ObjectCollection
from core.constants import STANDARD

G = STANDARD.G  # gravitational constant in m^3/(kg*s^2)
# ----------------------------- constants -------------------------------------

AU = 1.495_978_707e11        # m
DAY = 86_400.0               # s
JULIAN_DAY = 86_400.0        # s
J2000_JD = 2451545.0         # JD = 2000-01-01 12:00:00 TT ~ UTC for our purpose
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# Planet masses (kg) and mean radii (m). (JPL/NASA)
# (Rounded; visuals won't care about >0.1% differences.)
MASS = {
    "Sun":     1.9885e30,
    "Mercury": 3.3011e23,
    "Venus":   4.8675e24,
    "Earth":   5.9722e24,
    "Mars":    6.4171e23,
    "Jupiter": 1.8982e27,
    "Saturn":  5.6834e26,
    "Uranus":  8.6810e25,
    "Neptune": 1.02413e26,
}

RADIUS = {
    "Sun":     6.9634e8,
    "Mercury": 2.4397e6,
    "Venus":   6.0518e6,
    "Earth":   6.3710e6,
    "Mars":    3.3895e6,
    "Jupiter": 6.9911e7,
    "Saturn":  5.8232e7,  # mean
    "Uranus":  2.5362e7,
    "Neptune": 2.4622e7,
}

# ------------------- JPL Table 1: planets @ J2000 with rates -----------------
# See: https://ssd.jpl.nasa.gov/planets/approx_pos.html
# Columns: a [AU], e [-], i [deg], L [deg], varpi [deg], Omega [deg]
# Next line: rates per Julian century (same units as above).
# {name: (a0, aDot, e0, eDot, i0, iDot, L0, LDot, varpi0, varpiDot, Omega0, OmegaDot)}

JPL_T1 = {
    "Mercury": (0.38709927,  0.00000037, 0.20563593,  0.00001906, 7.00497902, -0.00594749,
                252.25032350, 149472.67411175,  77.45779628, 0.16047689,  48.33076593, -0.12534081),
    "Venus":   (0.72333566,  0.00000390, 0.00677672, -0.00004107, 3.39467605, -0.00078890,
                181.97909950,  58517.81538729, 131.60246718, 0.00268329,  76.67984255, -0.27769418),
    "EM Bary": (1.00000261,  0.00000562, 0.01671123, -0.00004392,-0.00001531,-0.01294668,
                100.46457166,  35999.37244981, 102.93768193, 0.32327364,   0.0,          0.0),
    "Mars":    (1.52371034,  0.00001847, 0.09339410,  0.00007882, 1.84969142, -0.00813131,
                 -4.55343205,  19140.30268499, -23.94362959, 0.44441088,  49.55953891, -0.29257343),
    "Jupiter": (5.20288700, -0.00011607, 0.04838624, -0.00013253, 1.30439695, -0.00183714,
                 34.39644051,   3034.74612775, 14.72847983, 0.21252668, 100.47390909,  0.20469106),
    "Saturn":  (9.53667594, -0.00125060, 0.05386179, -0.00050991, 2.48599187,  0.00193609,
                 49.95424423,   1222.49362201, 92.59887831,-0.41897216, 113.66242448, -0.28867794),
    "Uranus":  (19.18916464,-0.00196176, 0.04725744, -0.00004397,0.77263783, -0.00242939,
                313.23810451,    428.48202785,170.95427630, 0.40805281,  74.01692503,  0.04240589),
    "Neptune": (30.06992276, 0.00026291, 0.00859048,  0.00005105,1.77004347,  0.00035372,
                -55.12002969,    218.45945325, 44.96476227,-0.32241464, 131.78422574, -0.00508664),
}
# We'll treat EM Bary ≈ Earth for heliocentric placement (sufficient for visuals).

def datetime_to_jd(t: Optional[datetime]) -> float:
    """Convert naive or timezone-aware UTC datetime to Julian Date (rough TT-UTC ignored)."""
    if t is None:
        return J2000_JD
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    # Algorithm: unix epoch to JD
    unix = t.timestamp()
    # JD of Unix epoch 1970-01-01 00:00:00 UTC = 2440587.5
    return 2440587.5 + unix / JULIAN_DAY

def centuries_since_j2000(jd: float) -> float:
    return (jd - J2000_JD) / 36525.0

@dataclass
class Elements:
    a: float  # m
    e: float
    i: float     # rad
    Omega: float  # rad
    omega: float  # rad
    M: float  # rad

# ----------------- Keplerian → Cartesian (heliocentric/ecliptic) --------------

def solve_kepler(M: float, e: float, tol: float = 1e-12, itmax: int = 50) -> float:
    """Solve Kepler's equation M = E - e sin E for E (elliptic)."""
    E = M if e < 0.8 else math.pi
    for _ in range(itmax):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E

def elements_to_state(mu: float, el: Elements) -> Tuple[np.ndarray, np.ndarray]:
    """Return r,v in inertial frame given standard Keplerian elements."""
    a, e, i, O, w, M = el.a, el.e, el.i, el.Omega, el.omega, el.M
    E = solve_kepler(M, e)
    # position in orbital plane
    cosE, sinE = math.cos(E), math.sin(E)
    x_op = a * (cosE - e)
    y_op = a * math.sqrt(1 - e * e) * sinE
    # velocity in orbital plane
    n = math.sqrt(mu / (a * a * a))
    vx_op = -a * n * sinE / (1 - e * cosE)
    vy_op =  a * n * math.sqrt(1 - e * e) * cosE / (1 - e * cosE)

    # rotate by argument of periapsis, inclination, longitude of node
    cw, sw = math.cos(w), math.sin(w)
    cO, sO = math.cos(O), math.sin(O)
    ci, si = math.cos(i), math.sin(i)

    # rotation matrix R = Rz(O) * Rx(i) * Rz(w)
    R11 = cO*cw - sO*sw*ci
    R12 = -cO*sw - sO*cw*ci
    # R13 = sO*si
    R21 = sO*cw + cO*sw*ci
    R22 = -sO*sw + cO*cw*ci
    # R23 = -cO*si
    R31 = sw*si
    R32 = cw*si
    # R33 = ci

    r = np.array([R11*x_op + R12*y_op, R21*x_op + R22*y_op, R31*x_op + R32*y_op], dtype=float)
    v = np.array([R11*vx_op + R12*vy_op, R21*vx_op + R22*vy_op, R31*vx_op + R32*vy_op], dtype=float)
    return r, v

# ------------------------- planets @ arbitrary epoch ---------------------------

def jpl_elements_at(name: str, jd: float) -> Elements:
    """Compute planet heliocentric elements at JD from JPL Table 1 (linear in T centuries)."""
    if name == "Earth":
        key = "EM Bary"
    else:
        key = name
    (a0, aDot, e0, eDot, i0, iDot, L0, LDot, varpi0, varpiDot, O0, ODot) = JPL_T1[key]
    T = centuries_since_j2000(jd)
    a_AU = a0 + aDot * T
    e = e0 + eDot * T
    i = (i0 + iDot * T) * DEG2RAD
    L = (L0 + LDot * T) * DEG2RAD
    varpi = (varpi0 + varpiDot * T) * DEG2RAD
    O = (O0 + ODot * T) * DEG2RAD
    M = (L - varpi) % (2*math.pi)
    w = (varpi - O) % (2*math.pi)
    return Elements(a=a_AU * AU, e=e, i=i, Omega=O, omega=w, M=M)

# ------------------------------- moons ----------------------------------------

# Major moons: (mass kg, radius m, semi-major-axis m, period s, e≈, i≈deg)
MOONS = {
    "Earth": [
        # Moon (more realistic e & i so you get precession-looking shape)
        dict(name="Moon", mass=7.3477e22, radius=1.7374e6,
             a=384_400e3, period=27.321661 * DAY, e=0.0549, inc_deg=5.145),
    ],
    "Mars": [
        dict(name="Phobos", mass=1.0659e16, radius=11_266.7,  # mean radius ~11.27 km → m
             a=9_376e3, period=0.31891023 * DAY, e=0.0151, inc_deg=1.1),
        dict(name="Deimos", mass=1.4762e15, radius=6_200.0,
             a=23_463.2e3, period=1.263 * DAY, e=0.00033, inc_deg=0.9),
    ],
    "Jupiter": [
        dict(name="Io", mass=8.9319e22, radius=1.8216e6,
             a=421_800e3, period=1.769138 * DAY, e=0.0041, inc_deg=0.04),
        dict(name="Europa", mass=4.7998e22, radius=1.5608e6,
             a=671_100e3, period=3.551181 * DAY, e=0.009, inc_deg=0.47),
        dict(name="Ganymede", mass=1.4819e23, radius=2.6341e6,
             a=1_070_400e3, period=7.154553 * DAY, e=0.0013, inc_deg=0.21),
        dict(name="Callisto", mass=1.0759e23, radius=2.4103e6,
             a=1_882_700e3, period=16.689017 * DAY, e=0.0074, inc_deg=0.19),
    ],
    "Saturn": [
        dict(name="Mimas", mass=3.75e19, radius=198_000.0,
             a=185_539e3, period=0.942 * DAY, e=0.0196, inc_deg=1.6),
        dict(name="Enceladus", mass=1.08e20, radius=252_100.0,
             a=237_948e3, period=1.370 * DAY, e=0.0047, inc_deg=0.0),
        dict(name="Tethys", mass=6.17e20, radius=531_100.0,
             a=294_660e3, period=1.888 * DAY, e=0.0001, inc_deg=1.1),
        dict(name="Dione", mass=1.095e21, radius=561_400.0,
             a=377_400e3, period=2.736915 * DAY, e=0.0022, inc_deg=0.0),
        dict(name="Rhea", mass=2.306e21, radius=763_800.0,
             a=527_040e3, period=4.518 * DAY, e=0.001, inc_deg=0.3),
        dict(name="Titan", mass=1.3452e23, radius=2.575e6,
             a=1_221_870e3, period=(15 + 22/24) * DAY, e=0.0288, inc_deg=0.35),
        dict(name="Iapetus", mass=1.805e21, radius=734_500.0,
             a=3_560_820e3, period=79.3215 * DAY, e=0.0286, inc_deg=8.3),
    ],
    "Uranus": [
        dict(name="Miranda", mass=6.59e19, radius=235_800.0,
             a=129_390e3, period=1.413 * DAY, e=0.0013, inc_deg=4.2),
        dict(name="Ariel", mass=1.353e21, radius=578_900.0,
             a=190_930e3, period=2.520379 * DAY, e=0.0012, inc_deg=0.3),
        dict(name="Umbriel", mass=1.172e21, radius=584_700.0,
             a=266_000e3, period=4.144 * DAY, e=0.0039, inc_deg=0.1),
        dict(name="Titania", mass=3.527e21, radius=788_900.0,
             a=435_910e3, period=8.706 * DAY, e=0.0011, inc_deg=0.1),
        dict(name="Oberon", mass=3.014e21, radius=761_400.0,
             a=583_520e3, period=13.463 * DAY, e=0.0014, inc_deg=0.1),
    ],
    "Neptune": [
        dict(name="Triton", mass=2.14e22, radius=1.353e6,
             a=354_759e3, period=5.876854 * DAY, e=1.6e-5, inc_deg=157.0),  # retrograde
    ],
}

def make_planet_objects(jd: Optional[float] = None) -> List[Object]:
    """Sun + eight planets placed heliocentrically at JD."""
    jd = J2000_JD if jd is None else jd
    sun = Object(
        mass=MASS["Sun"], radius=RADIUS["Sun"],
        velocity=np.zeros(3), coordinates=Coordinates(0.0, 0.0, 0.0)
    )
    bodies = [sun]

    mu_sun = G * MASS["Sun"]
    for name in ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]:
        el = jpl_elements_at(name, jd)
        r, v = elements_to_state(mu_sun, el)
        bodies.append(
            Object(
                mass=MASS[name],
                radius=RADIUS[name],
                velocity=v.astype(np.float64),
                coordinates=Coordinates(*r.tolist()),
            )
        )
    return bodies

def _orbit_in_parent_frame(a: float, period: float, e: float, inc_deg: float, t_since_epoch: float) -> Tuple[np.ndarray, np.ndarray]:
    """Simplified moon state in its parent's frame using circular/elliptic approx and mean motion."""
    n = 2 * math.pi / period
    M = (n * t_since_epoch) % (2 * math.pi)
    # Use tiny e via Kepler to get a mild ellipse; if e≈0, this reduces to circle
    E = solve_kepler(M, e)
    cosE, sinE = math.cos(E), math.sin(E)
    # r_orb = a * (1 - e * cosE)
    x_op = a * (cosE - e)
    y_op = a * math.sqrt(1 - e * e) * sinE
    # velocities in orbital plane
    denom = (1 - e * cosE)
    vx_op = -a * n * sinE / denom
    vy_op =  a * n * math.sqrt(1 - e * e) * cosE / denom

    # tilt by inclination about x-axis (set node/arg peri to 0 for simple visuals)
    i = inc_deg * DEG2RAD
    ci, si = math.cos(i), math.sin(i)
    r = np.array([x_op,  ci*y_op,  si*y_op], dtype=float)
    v = np.array([vx_op, ci*vy_op, si*vy_op], dtype=float)
    return r, v

def attach_major_moons(bodies: List[Object], jd: Optional[float] = None) -> List[Object]:
    """Given Sun+planets, append major moons in planet-centric orbits."""
    jd = J2000_JD if jd is None else jd
    # Build a quick lookup from name prefix to object
    planets_by_name = {}
    for obj in bodies:
        for name in MASS.keys():
            if np.isclose(obj.mass, MASS[name], rtol=0, atol=0):  # crude match
                planets_by_name[name] = obj

    t_since_j2000 = (jd - J2000_JD) * DAY
    out = bodies[:]
    for parent_name, moons in MOONS.items():
        parent = planets_by_name.get(parent_name, None)
        if parent is None:
            continue
        r_parent = parent.coordinates.to_array()
        v_parent = parent.velocity
        # mu_parent = G * parent.mass

        for rec in moons:
            a = rec["a"]
            period = rec["period"]
            e = rec.get("e", 0.0)
            inc = rec.get("inc_deg", 0.0)
            # Moon state in parent's local frame
            r_loc, v_loc = _orbit_in_parent_frame(a, period, e, inc, t_since_j2000)
            # Promote to heliocentric state
            r_helio = r_parent + r_loc
            v_helio = v_parent + v_loc
            out.append(
                Object(
                    mass=rec["mass"],
                    radius=rec["radius"],
                    velocity=v_helio.astype(np.float64),
                    coordinates=Coordinates(*r_helio.tolist()),
                )
            )
    return out

def recenter_to_barycenter(objects: List[Object]) -> None:
    """Shift positions/velocities so total momentum and barycenter are at the origin."""
    Mtot = sum(o.mass for o in objects)
    r_cm = sum(o.mass * o.coordinates.to_array() for o in objects) / Mtot
    p_tot = sum(o.mass * o.velocity for o in objects)
    v_cm = p_tot / Mtot
    for o in objects:
        o.coordinates = Coordinates.from_iterable(o.coordinates.to_array() - r_cm)
        o.velocity = (o.velocity - v_cm).astype(np.float64)

def make_solar_system(
    when: Optional[datetime] = None,
    include_moons: bool = True,
    barycentric: bool = True,
) -> List[Object]:
    """
    Build Sun+planets (+major moons) at the given time.

    Args:
        when: datetime in UTC; if None ⇒ J2000 epoch.
        include_moons: attach major moons in simplified orbits.
        barycentric: recenter so the system barycenter is at origin and net momentum is zero.
    """
    jd = datetime_to_jd(when) if (when is not None) else J2000_JD
    bodies = make_planet_objects(jd=jd)
    if include_moons:
        bodies = attach_major_moons(bodies, jd=jd)
    if barycentric:
        recenter_to_barycenter(bodies)
    return bodies


def main(moons: bool = True, days: int = 365):
    # Example: build at J2000, run a quick sim
    from core.engine import SimulationEngine, run_simulation
    from core.plot import render_orbital_mp4

    bodies = make_solar_system(include_moons=moons, barycentric=True)
    collection = ObjectCollection(bodies)
    engine = SimulationEngine(collection, dt=DAY, softening=1e6, restitution=1.0)

    # simulate ~5 years for a quick smoke test
    run_simulation(engine, steps=days, print_every=100)
    render_orbital_mp4(
        engine,
        out_path="sol.mp4",
        plane="xy",
        fps=30,
        duration_s=30,
        with_velocity=False,
        show_barycenter=True,
        barycenter_trail=True,
        every_n=5
    )
