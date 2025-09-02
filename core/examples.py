"""Known Orbital Simulation Examples."""

import numpy as np

from core.constants import UnitSystem, get_unit_profile 
from core.engine import SimulationEngine, run_simulation
from core.physics import Coordinates, Object, ObjectCollection, set_circular_orbit
from core.plot import plot_orbits


def two_body_problem(
    body1_mass: float = 5.972e24,  # earth
    body1_radius: float = 6.371e6,  # earth radius
    body2_mass: float = 7.348e22,  # moon
    body2_radius: float = 1.737e6,  # moon radius
    distance: float = 384400e3,  # meters
    dt: float = 60*60,  # 1 hour in seconds
    steps: int = 1000,
    unit_profile: UnitSystem = "si"
):
    unit_profile = get_unit_profile(unit_profile)
    body1 = Object(
        mass=body1_mass,
        radius=body1_radius,
        velocity=np.zeros(3),
        coordinates=Coordinates(0, 0, 0)
    )

    body2 = Object(
        mass=body2_mass,
        radius=body2_radius,
        velocity=np.zeros(3),
        coordinates=Coordinates(distance, 0, 0)
    )

    # Add a constant drift so the barycenter moves straight (Galilean transform):
    # V_drift = np.array([3.0e4, 0.0, 0.0])  # e.g., 30 km/s
    # for b in [earth, moon]:
    #     b.velocity += V_drift
    set_circular_orbit(primary=body1, secondary=body2)

    collection = ObjectCollection([body1, body2])
    for obj in collection:
        print(obj)

    engine = SimulationEngine(collection, dt=dt, softening=1e3, restitution=1.0)

    run_simulation(engine, steps=steps)
    plot_orbits(engine, every_n=5, plane="xy", separate=False, with_velocity=False)


def sun_earth_moon(
    steps: int = 5000,
    dt: float = 900.0,              # 15 min; smaller dt = better Moon fidelity
    moon_incl_deg: float = 0.0,     # set to ~5.1 for realism
    softening: float = 1e3,
    unit_profile: UnitSystem = "si"
):
    """ Simulate the Earth-Moon system orbiting the Sun."""
    unit_profile = get_unit_profile(unit_profile)
    # --- constants ---
    M_sun,  R_sun  = 1.98847e30, 6.9634e8
    M_earth,R_earth = 5.972e24, 6.371e6
    M_moon, R_moon = 7.348e22, 1.737e6
    AU = 1.495978707e11
    R_em = 384400e3

    # --- bodies ---
    sun   = Object(M_sun,   R_sun,   velocity=np.zeros(3), coordinates=Coordinates(0, 0, 0))
    earth = Object(M_earth, R_earth, velocity=np.zeros(3), coordinates=Coordinates(AU, 0, 0))

    # Place Moon offset from Earth (choose x-offset; we'll set velocity ⟂ to radius)
    moon_pos = np.array([AU + R_em, 0.0, 0.0])
    if abs(moon_incl_deg) > 0:
        # rotate offset around x-axis to give inclination
        i = np.deg2rad(moon_incl_deg)
        moon_pos = np.array([AU + R_em, 0.0, R_em*np.sin(i)])  # simple tilt
    moon  = Object(M_moon, R_moon, velocity=np.zeros(3),
                   coordinates=Coordinates.from_iterable(moon_pos))

    # 1) Make Sun–Earth circular about their barycenter
    set_circular_orbit(sun, earth)          # total momentum of Sun+Earth = 0
    v_cm = earth.velocity.copy()             # solar-orbital velocity of Earth (also desired EM barycenter vel)

    # 2) Give Moon a circular velocity **relative to Earth**, then split it between Earth and Moon
    r_em_vec = moon.position() - earth.position()
    R = np.linalg.norm(r_em_vec)
    r_hat = r_em_vec / R

    # pick tangential unit vector in the XY plane (or projected to match inclination)
    # t_hat = ẑ × r̂  gives prograde motion in +y when r is +x
    z_hat = np.array([0.0, 0.0, 1.0])
    t_hat = np.cross(z_hat, r_hat)
    if np.linalg.norm(t_hat) < 1e-12:  # edge case if r_hat ∥ z_hat
        t_hat = np.array([0.0, 1.0, 0.0])
    t_hat /= np.linalg.norm(t_hat)

    v_rel = np.sqrt(unit_profile.G * (M_earth + M_moon) / R) * t_hat   # circular EM speed

    # Split so EM barycenter moves with v_cm (keeps Sun–(Earth+Moon) circular)
    earth.velocity = v_cm - (M_moon/(M_earth+M_moon)) * v_rel
    moon.velocity  = v_cm + (M_earth/(M_earth+M_moon)) * v_rel

    # --- simulate ---
    collection = ObjectCollection([sun, earth, moon])
    engine = SimulationEngine(collection, dt=dt, softening=softening, restitution=1.0)

    run_simulation(engine, steps=steps, print_every=500)
    plot_orbits(
        engine,
        every_n=10,
        plane="xy",
        separate=False,
        with_velocity=False,
        show_barycenter=True,
        barycenter_trail=True
    )
    return engine


def three_body_equilateral(
    m: float = 1e22,     # pick something modest
    R: float = 1e7,      # distance from center to each body
    dt: float = 50.0,
    steps: int = 8000,
    softening: float = 1e3,
    unit_profile: UnitSystem = "si"
):
    """Three equal-mass bodies in an equilateral triangle, with velocities for rigid rotation.
    Note as steps increases, the system becomes chaotic and eventually breaks symmetry.
    This is the classic Lagrange solution to the 3-body problem.
    """
    unit_profile = get_unit_profile(unit_profile)
    # Vertex positions (counterclockwise) at distance R from origin
    pos = [
        np.array([ R, 0.0, 0.0]),
        np.array([-0.5*R,  np.sqrt(3)/2*R, 0.0]),
        np.array([-0.5*R, -np.sqrt(3)/2*R, 0.0]),
    ]
    # Tangential directions for a rigid rotation (t_hat = ẑ × r̂)
    z_hat = np.array([0.0, 0.0, 1.0])
    t_hat = [np.cross(z_hat, p/np.linalg.norm(p)) for p in pos]

    # Required speed
    v = np.sqrt(unit_profile.G * m / (np.sqrt(3.0) * R))

    bodies = []
    for i in range(3):
        bodies.append(
            Object(
                mass=m,
                radius=(m/5000.0)**(1/3),  # arbitrary small radius for visuals
                velocity=v * t_hat[i],
                coordinates=Coordinates.from_iterable(pos[i]),
            )
        )

    collection = ObjectCollection(bodies)
    engine = SimulationEngine(collection, dt=dt, softening=softening, restitution=1.0)

    run_simulation(engine, steps=steps, print_every=500)
    plot_orbits(
        engine,
        every_n=5,
        plane="xy",
        separate=False,
        with_velocity=False,
        show_barycenter=True,
        barycenter_trail=True
    )
    return engine
