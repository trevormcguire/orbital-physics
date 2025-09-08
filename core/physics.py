"""Physics Engine."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Iterable
from uuid import uuid4


import numpy as np

from core.constants import STANDARD, UnitProfile


@dataclass
class Coordinates:
    """3D coordinates in space, where the origin (0,0,0) is arbitrary."""

    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        """Convert coordinates to a numpy array."""
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_iterable(cls, lst: Iterable[float]) -> Coordinates:
        """Create Coordinates from an iterable."""
        return cls(x=lst[0], y=lst[1], z=lst[2])
    
    @classmethod
    def random(self):
        """Generate random coordinates in [-1, 1] for each axis."""
        return Coordinates(
            x=np.random.uniform(-1, 1),
            y=np.random.uniform(-1, 1),
            z=np.random.uniform(-1, 1)
        )


def solve_kepler(M: float, e: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    """Estimate E in Kepler's equation M = E - e sin E for E (elliptic).

    https://en.wikipedia.org/wiki/Kepler%27s_equation
    
    Kepler’s equation links time to position along an elliptical orbit.

    Params
    -------
    1. M (mean anomaly) in radians
    2. e is eccentricity in radians
    
    Returns
    -------
    E, the eccentric anomaly, which is an angular parameter that defines the
    position of a body that is moving along an ellipse. It is useful to compute
    the position of a point moving in a Keplerian orbit.
    """
    # initial guess, use E=M if low eccentricity, otherwise pi
    E = M if e < 0.8 else math.pi
    # Newton–Raphson iteration
    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E  # radians

def moment_of_inertia(
    mass: float,
    radius: float,
    length: float = None,
    shape: Literal["sphere", "cylinder", "rod"] = "sphere"
) -> float:
    """
    Calculate the moment of inertia for common shapes.

    Args:
        mass (float): Mass of the object (kg).
        radius (float): Radius (m).
        length (float, optional): Length (m), only for rods.
        shape (str): Shape type ("sphere", "cylinder", "rod").

    Returns:
        float: Moment of inertia (kg*m^2).

    References:
        https://en.wikipedia.org/wiki/List_of_moments_of_inertia
    """
    if shape == "sphere":
        # Solid sphere about center: I = (2/5) * m * r^2
        return (2/5) * mass * radius**2
    elif shape == "cylinder":
        # Solid cylinder about axis: I = (1/2) * m * r^2
        return 0.5 * mass * radius**2
    elif shape == "rod":
        # Thin rod about center: I = (1/12) * m * L^2
        if length is None:
            raise ValueError("Length must be provided for rod shape.")
        return (1/12) * mass * length**2
    else:
        raise ValueError(f"Unknown shape: {shape}")


def random_angular_velocity(max_rotation_rps: float = 1.0, dim: int = 3) -> np.ndarray:
    """
    Generate a random angular velocity vector.

    Args:
        max_rotation_rps (float): Maximum magnitude (radians per second).

    Returns:
        np.ndarray: 3D angular velocity vector.
    """
    axis = np.random.randn(dim)
    axis /= np.linalg.norm(axis)  # Normalize to unit vector
    omega = np.random.uniform(0, max_rotation_rps)
    return omega * axis


def pairwise_accelerations(objects: list[Object], eps: float = 0.0, unit_profile: UnitProfile = STANDARD):
    """
    Returns:
        acc: dict[uuid] -> np.ndarray (acc vector)
        U: total potential energy (with softening)
    """
    n = len(objects)
    acc = {obj.uuid: np.zeros(3) for obj in objects}
    U = 0.0
    eps2 = eps * eps

    for i in range(n):
        oi = objects[i]
        ri = oi.position()
        mi = oi.mass
        for j in range(i + 1, n):
            oj = objects[j]
            rj = oj.position()
            mj = oj.mass

            rij = rj - ri
            r2 = float(rij @ rij) + eps2
            inv_r = 1.0 / np.sqrt(r2)
            inv_r3 = inv_r / r2  # 1/r^3 with softening

            # Acceleration contributions
            a_i = unit_profile.G * mj * inv_r3 * rij
            a_j = -unit_profile.G * mi * inv_r3 * rij

            acc[oi.uuid] += a_i
            acc[oj.uuid] += a_j

            # Potential energy (count each pair once)
            U += -unit_profile.G * mi * mj * inv_r
    return acc, U

class Object:
    """A massive object in 3D space.

    Attributes:
        mass (float): Mass of the object (kg).
        coordinates (Coordinates): Position in 3D space.
        uuid (str): Unique identifier.
    """
    def __init__(
        self,
        mass: float,
        radius: float,
        velocity: np.ndarray,
        coordinates: Coordinates = None,
        moi: float = None,
        angular_velocity: np.ndarray = None,
        uuid: str = None,
        unit_profile: UnitProfile = STANDARD,
        name: str = None
    ):
        self.mass = mass
        self.radius = radius
        self.coordinates = coordinates if coordinates else Coordinates.random()
        self.velocity = velocity.astype(np.float32) if velocity is not None else np.zeros(3).astype(np.float32)
        # Moment of Inertia for a solid sphere of uniform density (TODO: non-uniform): I = (2/5)mr^2
        self.moi = moi if moi is not None else moment_of_inertia(mass, radius, shape="sphere")
        # angular velocity vector (rad/s): a vector representing the axis and rate of rotation (radians per second).
        self.angular_velocity = angular_velocity.astype(np.float32) if angular_velocity is not None else random_angular_velocity().astype(np.float32)
        self.uuid = uuid if uuid else uuid4().hex
        self.name = name if name is not None else self.uuid[:6]
        self.unit_profile = unit_profile

    def to_dict(self):
        return {
            "mass": self.mass,
            "radius": self.radius,
            "coordinates": {"x": self.coordinates.x, "y": self.coordinates.y, "z": self.coordinates.z},
            "velocity": self.velocity.tolist(),
            "moi": self.moi,
            "angular_velocity": self.angular_velocity.tolist(),
            "uuid": self.uuid,
            "unit_profile": self.unit_profile.name.value,
        }

    def set_unit_profile(self, unit_profile: UnitProfile):
        self.unit_profile = unit_profile
    
    def __eq__(self, other: Object):
        return self.uuid == other.uuid
    
    def __repr__(self):
        return f"Object({self.to_dict()})"

    def position(self) -> np.ndarray:
        return self.coordinates.to_array()

    def lagrangian(self, system: Iterable[Object]) -> float:
        """Calculate the Lagrangian (kinetic - potential energy) of this object.

        Math:
            Kinetic Energy (T):  (1/2mv^2)
                T = 0.5 * m * v^2.
            Potential Energy (U):
                U = -G * m1 * m2 / r
            Lagrangian (L):
                L = T - U

            Where:
            - m is the mass of the object.
            - v is the velocity magnitude.
            - G is the gravitational constant (6.67430e-11 m^3 kg^-1 s^-2).
            - m1 and m2 are the masses of the two objects.
            - r is the distance between the centers of the masses.

        Note:
            The potential energy is summed over all other objects in the system.
        """
        # total kinetic energy = 0.5 * m * v^2 + 0.5 * I * ω^2
        # I = moment of inertia
        # ω = angular velocity
        # m = mass
        # v = velocity
        # https://en.wikipedia.org/wiki/Kinetic_energy#Rotation_in_systems
        T_trans = 0.5 * self.mass * np.linalg.norm(self.velocity)**2
        T_rot = 0.5 * self.moi * np.linalg.norm(self.angular_velocity)**2
        ke = T_trans + T_rot

        # Calculate potential energy from all other objects
        np_coordinates = self.coordinates.to_array()
        pe = 0
        for other in system:
            if other is not self:
                r = np.linalg.norm(np_coordinates - other.coordinates.to_array())
                pe += -self.unit_profile.G * self.mass * other.mass / r

        # Lagrangian
        return ke - pe

    def force_vector(self, other: Object) -> np.ndarray:
        """Calculate the gravitational force vector exerted on another object.

        Math:
            Newton's law of universal gravitation:
                F = G * m1 * m2 / r^2

            - F is the magnitude of the gravitational force.
            - G is the gravitational constant (6.67430e-11 m^3 kg^-1 s^-2).
            - m1 and m2 are the masses of the two objects.
            - r is the distance between the centers of the masses.

            The force is a vector pointing from self to 'other':
                r_vector = other.position - self.position
                force_direction = r_vector / |r_vector|
                force_vector = force_magnitude * force_direction
        
        Note:
        Note on Newton's Third Law:
            If two bodies exert forces on each other, these forces have the same magnitude but opposite directions.
            So obj1.force_vector(obj2) == -obj2.force_vector(obj1)
        """
        r_vector = other.coordinates.to_array() - self.coordinates.to_array()  # Vector from self to other
        distance = np.linalg.norm(r_vector)  # Euclidean distance between objects
        if distance == 0:
            return np.zeros(3)  # No force if objects occupy the same position
        force_magnitude = self.unit_profile.G * self.mass * other.mass / distance**2  # Scalar magnitude of force
        force_direction = r_vector / distance  # Unit vector in direction of force
        return force_magnitude * force_direction  # Force vector

    def update(self, acceleration: np.ndarray, dt: float) -> None:
        """Update the object's velocity and position using acceleration and time step.

        Math:
            Velocity update:
                v_new = v_old + a * dt

            Position update:
                x_new = x_old + v_new * dt

            - a is the acceleration vector (m/s^2).
            - dt is the time step (s).

        This uses a simple Euler integration scheme.
        """
        self.velocity += acceleration * dt
        new_pos = self.coordinates.to_array() + self.velocity * dt
        self.coordinates = Coordinates.from_iterable(new_pos)


def fragmentation_probability(obj1: Object, obj2: Object) -> float:
    """
    Compute the probability of fragmentation based on collision energy and masses.

    Uses a logistic function:
        p = 1 / (1 + exp(-k * (E_coll / E_thresh - 1)))
    Where:
        - E_coll: kinetic energy of collision
        - E_thresh: threshold energy for fragmentation (function of masses)
        - k: steepness parameter

    Returns:
        float: Probability between 0 and 1
    """
    # Relative velocity
    v_rel = np.linalg.norm(obj1.velocity - obj2.velocity)
    # Reduced mass
    mu = (obj1.mass * obj2.mass) / (obj1.mass + obj2.mass)
    # Collision kinetic energy
    E_coll = 0.5 * mu * v_rel**2
    # Threshold energy: proportional to combined mass (tunable)
    E_thresh = 0.5 * (obj1.mass + obj2.mass) * 1e3  # 1e3 is a tunable constant
    k = 5  # Steepness of transition (tunable)
    p = 1 / (1 + np.exp(-k * (E_coll / E_thresh - 1)))
    return p

def resolve_collision(obj1: Object, obj2: Object, collection: ObjectCollection):
    """
    Handle the outcome of a collision between obj1 and obj2.
    - If mass ratio is large, absorb the smaller object.
    - Otherwise, compute fragmentation probability and fragment if needed.
    - Else, perform elastic collision.
    """
    mass_ratio = max(obj1.mass, obj2.mass) / min(obj1.mass, obj2.mass)
    threshold = 10  # Arbitrary: if one object is 10x more massive

    if mass_ratio > threshold:
        # Absorption: smaller object is destroyed, mass added to larger
        larger, smaller = (obj1, obj2) if obj1.mass > obj2.mass else (obj2, obj1)
        larger.mass += smaller.mass
        larger.radius = (larger.radius**3 + smaller.radius**3)**(1/3)
        collection.remove(smaller)
    else:
        # Fragmentation probability
        p_frag = fragmentation_probability(obj1, obj2)
        if np.random.rand() < p_frag:
            # Fragment both objects into smaller pieces (not implemented here)
            collection.remove(obj1)
            collection.remove(obj2)
            # collection.extend(generate_fragments(obj1, obj2, ...))
            # You can implement generate_fragments to create debris objects
        else:
            # Default: elastic collision (already handled in handle_collisions)
            pass


def collide_spheres(obj1: Object, obj2: Object, restitution: float = 1.0):
    r1 = obj1.position()
    r2 = obj2.position()
    n = r1 - r2
    dist = np.linalg.norm(n)
    if dist == 0:
        return
    n /= dist

    m1, m2 = obj1.mass, obj2.mass
    v_rel = np.dot(obj1.velocity - obj2.velocity, n)
    if v_rel >= 0:
        return  # separating

    # 1D impulse along normal with restitution e
    # Coefficient of restitution e in [0,1]; impulse magnitude for 1D along n:
    # j = -(1+e) * v_rel / (1 / m1 + 1 / m2)
    m1_inv = 1. / m1
    m2_inv = 1. / m2
    e = float(np.clip(restitution, 0.0, 1.0))

    j = -(1 + e) * v_rel / (m1_inv + m2_inv)
    impulse = j * n
    obj1.velocity += impulse / m1
    obj2.velocity -= impulse / m2

    # Positional correction: push out of overlap proportionally to masses
    overlap = obj1.radius + obj2.radius - dist
    if overlap > 0:
        corr = overlap / (m1_inv + m2_inv)
        obj1.coordinates = Coordinates.from_iterable(r1 + n * (corr / m1))
        obj2.coordinates = Coordinates.from_iterable(r2 - n * (corr / m2))


def set_circular_orbit(primary: Object, secondary: Object, plane_normal=np.array([0., 0., 1.]), unit_profile: UnitProfile = STANDARD):
    """
    Sets velocities for a circular orbit of `secondary` around `primary`, and
    adjusts `primary` so total momentum is zero.
    """
    r = secondary.position() - primary.position()
    R = np.linalg.norm(r)
    if R == 0:
        raise ValueError("Bodies at same position.")

    # # Tangential direction t is perpendicular to radius vector in the chosen plane.
    t = np.cross(plane_normal / np.linalg.norm(plane_normal), r / R)
    if np.linalg.norm(t) < 1e-12:
        # Choose another plane if degenerate
        t = np.cross(np.array([0., 1., 0.]), r / R)
    t /= np.linalg.norm(t)


    # Circular orbital speed for reduced two-body about barycenter:
    v_mag = np.sqrt(unit_profile.G * (primary.mass + secondary.mass) / R)
    v2 = v_mag * t
    # Ensure zero total linear momentum: v1 = -(m2 / m1) * v2
    v1 = -(secondary.mass / primary.mass) * v2

    primary.velocity = v1
    secondary.velocity = v2


class ObjectCollection(object):
    """A collection of massive objects for interaction modeling.

    Attributes:
        objects (list[Object]): List of Object instances.
    """
    def __init__(self, objects: list[Object]):
        self.objects = objects
    
    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index):
        return self.objects[index]
    
    def __iter__(self):
        return iter(self.objects)

    def force_vector_map(self):
        """Compute the force_vector map of all objects on each other.

        Returns:
            dict: A mapping from object UUIDs to their net acceleration vectors.
        """
        force_vector_map = {obj.uuid: np.zeros(3) for obj in self.objects}
        for i, obj in enumerate(self.objects):
            for j, other in enumerate(self.objects):
                if i != j:
                    force = obj.force_vector(other)
                    # Acceleration = Force / mass
                    acceleration = force / obj.mass
                    force_vector_map[obj.uuid] += acceleration
        return force_vector_map
    
    def extend(self, new_objects: Iterable[Object]) -> None:
        """Add new objects to the collection."""
        self.objects.extend(new_objects)
    
    def append(self, new_object: Object) -> None:
        """Add a single new object to the collection."""
        self.objects.append(new_object)
    
    def pop(self, index: int = -1) -> Object:
        """Remove and return an object at a specific index."""
        return self.objects.pop(index)
    
    def remove(self, obj: Object) -> None:
        """Remove a specific object from the collection."""
        self.objects.remove(obj)

    def handle_collisions(self, restitution: float = 1.0, merge_on_capture: bool = False):
        N = len(self.objects)
        to_remove = []
        for i in range(N):
            oi = self.objects[i]
            for j in range(i+1, N):
                oj = self.objects[j]
                r = np.linalg.norm(oi.position() - oj.position())
                if r <= (oi.radius + oj.radius):
                    if merge_on_capture:
                        # Simple merge: conserve momentum; recompute radius by volume
                        m_new = oi.mass + oj.mass
                        v_new = (oi.mass * oi.velocity + oj.mass * oj.velocity) / m_new
                        # keep center at mass-weighted position
                        r_new = (oi.mass * oi.position() + oj.mass * oj.position()) / m_new
                        # radius by volume add (assume equal density spheres)
                        R_new = (oi.radius**3 + oj.radius**3)**(1/3)
                        oi.mass = m_new
                        oi.velocity = v_new
                        oi.coordinates = Coordinates.from_iterable(r_new)
                        oi.radius = R_new
                        to_remove.append(oj)
                    else:
                        collide_spheres(oi, oj, restitution=restitution)
        for obj in to_remove:
            self.remove(obj)

