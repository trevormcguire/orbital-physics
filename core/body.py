"""A massive body in 3D space."""
from __future__ import annotations

import math

from core.constants import STANDARD
from core.units import Unit, Meters, AU, Radians, Degrees, Kilograms, SolarMasses, Seconds, Days


G = STANDARD.G


class Body:
    """A body in 3D space.
    
    
    NOTE: planets use e, a, i, Ω, ϖ, L0
    while smaller bodies may use e, q, i, Ω, ω, T0.
    So varpi (ϖ) is typically used for planets, but not moons.
    ϖ = Ω + ω
     where ω is [argument of periapsis](https://en.wikipedia.org/wiki/Argument_of_periapsis)
     and ϖ is the [longitude of periapsis](https://en.wikipedia.org/wiki/Longitude_of_periapsis)
    
    See: https://en.wikipedia.org/wiki/Orbital_elements

    """
    def __init__(
        self,
        name: str,
        a: Meters | AU,
        e: float,
        I: Degrees | Radians,
        L: Degrees | Radians,  # mean longitude, defined as L = ϖ + M, where ϖ = Ω + ω, so L = Ω + ω + M)
        M: Degrees | Radians,  # mean anomaly, optional if L specified, can be derived
        long_peri: Degrees | Radians,  # ϖ, optional if ϖ specified, can be derived
        long_node: Degrees | Radians,  # Ω
        arg_peri: Degrees | Radians,  # ω, optional if ϖ specified, can be derived
        mass: Kilograms | SolarMasses,
        radius: Meters | AU,
        b: Meters | AU = None,  # semi-minor axis, optional
        fg: float = None,  # surface gravity, m/s^2
        T: Seconds | Days | float = None,  # orbital period in seconds, optional
        mu: float = None,  # standard gravitational parameter GM, optional
        parent: Body = None
    ):
        self.name = name
        self.a = a
        self.e = e
        self.I = I
        self.L = L 
        self.M = M
        self.long_peri = long_peri  # varpi
        self.long_node = long_node  # Omega
        self.arg_peri = arg_peri  # omega
        self.mass = mass
        self.radius = radius
        self.b = b
        self.fg = fg
        self.T = Seconds(T) if isinstance(T, float) else T  # orbital period
        self.parent = parent  # name of parent body, if any
        self.mu = mu  # standard gravitational parameter
        self.derive()

    def derive(self):
        """Derive missing orbital elements if possible."""
        # standard gravitational parameter
        if self.mu is None:
            self.mu = self.get_mu()

        # semi-minor axis
        if self.b is None:
            self.b = self.get_b()

        # longitude of periapsis and argument of periapsis are interchangeable
        if self.long_peri is None:
            assert self.arg_peri is not None, "Must provide either long_peri or arg_peri"
            self.long_peri = self.long_node + self.arg_peri
        elif self.arg_peri is None:
            assert self.long_peri is not None, "Must provide either long_peri or arg_peri"
            self.arg_peri = self.long_peri - self.long_node

        # mean anomaly and mean longitude are interchangeable
        if self.M is None:
            assert self.L is not None
            self.M = (self.L - self.long_peri)
        elif self.L is None:
            assert self.M is not None
            self.L = (self.long_peri + self.M)

        # compute surface gravity F = GM/r^2
        if self.fg is None:
            self.fg = self.get_fg()
        
        # compute orbital period T = 2π * sqrt(a^3 / GM), where M is mass of central body (parent)
        if self.T is None:
            self.T = self.get_T()

    def get_mu(self):
        """Standard gravitational parameter: G*M"""
        m = self.mass.to_kilograms() if isinstance(self.mass, SolarMasses) else self.mass
        return G * (m.value)

    def get_fg(self):
        r = self.radius.to_meters() if isinstance(self.radius, AU) else self.radius
        return self.mu / (r.value**2)

    def get_T(self):
        """Compute orbital period: 
        T = 2π * sqrt(a^3 / GM)
        where M is mass of central body (parent)
        """
        if self.parent is None:
            return None
        M = self.parent.mass
        M = M.to_kilograms() if isinstance(M, SolarMasses) else M
        a = self.a.to_meters() if isinstance(self.a, AU) else self.a
        return Seconds(2 * math.pi * math.sqrt((a.value**3) / (G * M.value)))

    def get_b(self) -> float:
        """compute the semi-minor axis (b) from the semi-major axis (a) and eccentricity (e)."""
        # semi-minor axis (b = a * sqrt(1 - e^2))
        a = self.a.to_meters() if isinstance(self.a, AU) else self.a
        return Meters(a.value * math.sqrt(1 - self.e**2))

    def to_dict(self):
        return {
            "name": self.name,
            "a": self.a,
            "e": self.e,
            "I": self.I,
            "L": self.L,
            "long_peri": self.long_peri,
            "long_node": self.long_node,
            "M": self.M,
            "arg_peri": self.arg_peri,
            "mass": self.mass,
            "radius": self.radius,
            "b": self.b,
            "mu": self.mu,
            "fg": self.fg,
            "T": self.T,
            "parent": self.parent.name if self.parent else ""
        }

    def to_json(self) -> dict:
        """Return a JSON-serializable dict."""
        json = {}
        for k, v in self.to_dict().items():
            if isinstance(v, Unit):
                json[k] = v.value
            else:
                json[k] = v
        return json

    def __repr__(self):
        return f"Body({self.to_dict()})"

    def mean_motion(self):
        """Mean motion (denoted 'n') is the angular speed required for a body to complete one orbit.
        
        See "Mean Motion and Kepler's Laws" https://en.wikipedia.org/wiki/Mean_motion
        for derivation.
        """
        a = (self.a.to_meters() if isinstance(self.a, AU) else self.a).value
        n = math.sqrt(self.mu / a**3)
        return n

    def get_state(self):
        """Return position and velocity vectors in meters and m/s.
        https://en.wikipedia.org/wiki/Kepler%27s_equation
        """
        from core.sol import solve_kepler
        if self.parent is None:
            return [0., 0., 0.], [0., 0., 0.]  # origin
            # raise ValueError("Cannot compute state for body without parent.")
        # E is the eccentric anomaly
        # it is useful to compute the position of a point moving in a Keplerian orbit.
        M = (self.M.to_radians() if isinstance(self.M, Degrees) else self.M).value
        a = (self.a.to_meters() if isinstance(self.a, AU) else self.a).value
        I = (self.I.to_radians() if isinstance(self.I, Degrees) else self.I).value
        Omega = (self.long_node.to_radians() if isinstance(self.long_node, Degrees) else self.long_node).value
        omega = (self.arg_peri.to_radians() if isinstance(self.arg_peri, Degrees) else self.arg_peri).value
        b = (self.b.to_meters() if isinstance(self.b, AU) else self.b).value

        E = solve_kepler(M, self.e)  # a point on the ellipse
        cos_E, sin_E = math.cos(E), math.sin(E)
        
        x_op = a * (cos_E - self.e)
        y_op = b * sin_E

        n = self.mean_motion()

        vx_op = -a * n * sin_E / (1 - self.e * cos_E)
        vy_op =  a * n * math.sqrt(1 - self.e**2) * cos_E / (1 - self.e * cos_E)

        # rotate by argument of periapsis, inclination, longitude of node
        cw, sw = math.cos(omega), math.sin(omega)
        ci, si = math.cos(I), math.sin(I)
        cO, sO = math.cos(Omega), math.sin(Omega)

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

        r = [R11*x_op + R12*y_op, R21*x_op + R22*y_op, R31*x_op + R32*y_op]
        v = [R11*vx_op + R12*vy_op, R21*vx_op + R22*vy_op, R31*vx_op + R32*vy_op]
        return r, v



class System:
    """A flattened collection of bodies."""
    def __init__(self, bodies: list[Body]):
        self.bodies = bodies
    
    def __getitem__(self, idx: int) -> Body:
        return self.bodies[idx]

#         self._names = [body.name.lower() for body in bodies]
    
#     def to_dict(self):
#         return {body.name: body.to_dict() for body in self.bodies}
    
#     def to_json(self):
#         return {body.name: body.to_json() for body in self.bodies}
    
#     def __repr__(self):
#         return f"System({self.bodies})"

#     def values(self):
#         return self.to_json()
    
#     def _get_index(self, name: str):
#         name = name.lower()
#         if name not in self._names:
#             return None
#         return self._names.index(name)

#     def get(self, name: str):
#         idx = self._get_index(name)
#         return self.bodies[idx] if idx is not None else None
    
#     def pop(self, name: str) -> dict | Body | None:
#         """Remove and return a body by name, or None if not found."""
#         idx = self._get_index(name)
#         return self.bodies.pop(idx) if idx is not None else None

#     def add(self, body: Body):
#         """Add a body to the system."""
#         if self._get_index(body.name) is not None:
#             raise ValueError(f"Body with name '{body.name}' already exists in system.")
#         self.bodies.append(body)
#         self._names.append(body.name.lower())
