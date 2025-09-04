from __future__ import annotations

from core.constants import STANDARD
from core.units import Unit, Meters, AU, Radians, Degrees, Kilograms, SolarMasses


G = STANDARD.G
# https://en.wikipedia.org/wiki/Epoch_(astronomy)#J2000
EPOCH = 2451545.0  # J2000


class Dataset(object):
    def __init__(
        self,
        data: list[dict],
        distance_unit: str = "meters",
        mass_unit: str = "kg",
        angle_unit: str = "radians"
    ):
        self.data = data
        self.distance_unit = distance_unit
        self.mass_unit = mass_unit
        self.angle_unit = angle_unit
    
    def set_distance_unit(self, unit: str):
        assert unit in ("meters", "au"), "Distance unit must be 'meters' or 'au'"
        self.distance_unit = unit

    def set_mass_unit(self, unit: str):
        assert unit in ("kilograms", "m_solar"), "Mass unit must be 'kilograms' or 'm_solar'"
        self.mass_unit = unit

    def set_angle_unit(self, unit: str):
        assert unit in ("radians", "degrees"), "Angle unit must be 'radians' or 'degrees'"
        self.angle_unit = unit

    def _convert(self, value):
        if not isinstance(value, Unit):
            return value
        if isinstance(value, Meters) and self.distance_unit == "au":
            return value.to_au()
        if isinstance(value, AU) and self.distance_unit == "meters":
            return value.to_meters()
        if isinstance(value, Radians) and self.angle_unit == "degrees":
            return value.to_degrees()
        if isinstance(value, Degrees) and self.angle_unit == "radians":
            return value.to_radians()
        if isinstance(value, Kilograms) and self.mass_unit == "m_solar":
            return value.to_solar_masses()
        if isinstance(value, SolarMasses) and self.mass_unit == "kilograms":
            return value.to_kilograms()
        return value

    def convert_types(self, distance_unit: str = None, mass_unit: str = None, angle_unit: str = None) -> Dataset:
        distance_unit = distance_unit or self.distance_unit
        mass_unit = mass_unit or self.mass_unit
        angle_unit = angle_unit or self.angle_unit
    
        standardized_data = []
        for row in self.data:
            standardized_data.append({k: self._convert(v) for k, v in row.items()})
        return Dataset(standardized_data, distance_unit=distance_unit, mass_unit=mass_unit, angle_unit=angle_unit)

    def values(self):
        return [{k: v.value if isinstance(v, Unit) else v for k, v in row.items()} for row in self.data]
    
    def get(self, name: str):
        for row in self.data:
            if row.get("name", "").lower() == name.lower():
                return row
        return None
    
    def pop(self, name: str) -> dict | Body | None:
        for i, row in enumerate(self.data):
            if row.get("name", "").lower() == name.lower():
                return self.data.pop(i)
        return None

class Body:
    """A body in 3D space.
    
    
    NOTE: planets use e, a, i, Ω, ϖ, L0
    while smaller bodies may use e, q, i, Ω, ω, T0.
    So varpi (ϖ) is used for planets, but not moons.
    ϖ = Ω + ω
    where ω is argument of periapsis and ϖ is the longitude of periapsis
    https://en.wikipedia.org/wiki/Argument_of_periapsis
    https://en.wikipedia.org/wiki/Longitude_of_periapsis
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
        fg: float = None,  # surface gravity, m/s^2
        parent: str = ""
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
        self.fg = fg
        self.derive()
        self.parent = parent  # name of parent body, if any

    def derive(self):
        """Derive missing orbital elements if possible."""
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

    def get_fg(self):
        # TODO: convert back if in opposite units?
        m = self.mass.to_kilograms() if isinstance(self.mass, SolarMasses) else self.mass.value
        r = self.radius.to_meters() if isinstance(self.radius, AU) else self.radius.value
        return G * m / (r**2)

    def to_dict(self):
        return {
            "name": self.name,
            "a": self.a,
            "e": self.e,
            "I": self.I,
            "L": self.L,
            "long.peri": self.long_peri,
            "long.node": self.long_node,
            "M": self.M,
            "arg.peri": self.arg_peri,
            "mass": self.mass,
            "radius": self.radius,
            "fg": self.fg,
            "parent": self.parent
        }

    def to_json(self):
        return {
            "name": self.name,
            "a": self.a.value,
            "e": self.e,
            "I": self.I.value,
            "L": self.L.value,
            "long.peri": self.long_peri.value,
            "long.node": self.long_node.value,
            "M": self.M.value,
            "arg.peri": self.arg_peri.value,
            "mass": self.mass.value,
            "radius": self.radius.value,
            "fg": self.fg,
            "parent": self.parent
        }

    def __repr__(self):
        return f"Body({self.to_dict()})"

    

# https://en.wikipedia.org/wiki/Orbital_elements
def _make_keplerian_element(a, e, I, L, long_peri, long_node):
    """ Helper to convert from classical elements to our OrbitalElements dataclass."""
    # note, we can derice ω and M from these, but we store as-is for clarity
    # M = (L - ϖ) % (2 * math.pi)  # mean anomaly
    # omega = (ϖ - Ω) % (2 * math.pi)  # argument of periapsis (ω)
    return {
        "a": AU(a),
        "e": e,
        "I": Degrees(I),
        "L": Degrees(L),  # mean longitude (defined as L = ϖ + M, where ϖ = Ω + ω, so L = Ω + ω + M)
        "long.peri": Degrees(long_peri),  # longitude of periapsis (closest point, defied as ϖ = Ω + ω), rename varpi?
        "long.node": Degrees(long_node),  # longitude of ascending node (defied as Ω),  rename Omega?
        # "M": Radians((L - long_peri) % (2 * math.pi)),  # mean anomaly
        # "omega": Radians((long_peri - long_node) % (2 * math.pi))  # argument of periapsis (ω)
    }

# Source: NASA/JPL: https://ssd.jpl.nasa.gov/planets/approx_pos.html
# with reference to to the mean ecliptic and equinox of J2000, valid for the time-interval 1800 AD - 2050 AD
def solar_keplerian_elements(**kwargs):
    return Dataset([
        {"name": "Mercury", **_make_keplerian_element(0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593)},
        {"name": "Venus", **_make_keplerian_element(0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255)},
        {"name": "Earth", **_make_keplerian_element(1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0)},  # Earth Moon Barycenter
        {"name": "Mars", **_make_keplerian_element(1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891)},
        {"name": "Jupiter", **_make_keplerian_element(5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909)},
        {"name": "Saturn", **_make_keplerian_element(9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448)},
        {"name": "Uranus", **_make_keplerian_element(19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503)},
        {"name": "Neptune", **_make_keplerian_element(30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574)},
        # https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=Ceres
        {"name": "Ceres", **_make_keplerian_element(2.766051, 0.0794, 10.588, 188.70268, 73.2734, 80.2522)},
        # https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=Pluto
        {"name": "Pluto", **_make_keplerian_element(39.5886, 0.2518, 17.1477, 38.68366, 113.709, 110.292)},
        # https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=Eris
        {"name": "Eris", **_make_keplerian_element(68.0506, 0.435675, 43.821, 211.032, 150.714, 36.0460)},
        # https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=20000%20Varuna
        {"name": "20000 Varuna", **_make_keplerian_element(43.1374, 0.053565, 17.1395, 114.900, 272.579, 97.21338)},
        # https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=makemake
        {"name": "Makemake", **_make_keplerian_element(45.4494, 0.16194, 29.03386, 168.8258, 296.95, 79.259)},
        # https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=28978%20Ixion
        {"name": "28978 Ixion", **_make_keplerian_element(39.3745, 0.2449, 19.6745, 293.546, 300.585, 71.099)},
    ], **kwargs)


# Source: NASA/JPL https://ssd.jpl.nasa.gov/planets/phys_par.html
def solar_physical_properties(**kwargs):
    return Dataset([
        {"name": "Sun", "mass": Kilograms(1.9885e30), "radius": Meters(6.9634e8), "fg": 274.},
        {"name": "Mercury", "mass": Kilograms(3.3011e23), "radius": Meters(2.4397e6), "fg": 3.70},
        {"name": "Venus", "mass": Kilograms(4.8675e24), "radius": Meters(6.0518e6), "fg": 8.87},
        {"name": "Earth", "mass": Kilograms(5.9722e24), "radius": Meters(6.371e6), "fg": 9.80},
        {"name": "Mars", "mass": Kilograms(6.4171e23), "radius": Meters(3.3895e6), "fg": 3.71},
        {"name": "Jupiter", "mass": Kilograms(1.8982e27), "radius": Meters(6.9911e7), "fg": 24.79},
        {"name": "Saturn", "mass": Kilograms(5.6834e26), "radius": Meters(5.8232e7), "fg": 10.44},
        {"name": "Uranus", "mass": Kilograms(8.6810e25), "radius": Meters(2.5362e7), "fg": 8.87},
        {"name": "Neptune", "mass": Kilograms(1.02413e26), "radius": Meters(2.4622e7), "fg": 11.15},
        {"name": "Pluto", "mass": Kilograms(13024.6e18), "radius": Meters(1188300), "fg": 0.62},
        {"name": "Ceres", "mass": Kilograms(938.416e18), "radius": Meters(469700), "fg": 0.27},
        {"name": "Eris", "mass": Kilograms(16600e18), "radius": Meters(1163000), "fg": 0.77},
        {"name": "20000 Varuna", "mass": Kilograms(3.698e20), "radius": Meters(334000), "fg": 0.15},
        {"name": "Makemake", "mass": Kilograms(3100e18), "radius": Meters(714000), "fg": 0.4},
        {"name": "28978 Ixion", "mass": Kilograms(2.773e17), "radius": Meters(355000), "fg": 0.45},
    ], **kwargs)


def solar_system(moons: bool = False, **kwargs):
    bodies = [
        Body(parent="", name="Sol", mass=Kilograms(1.9885e30), radius=Meters(6.9634e8), a=AU(0), e=0, I=Degrees(0), L=Degrees(0), long_peri=Degrees(0), long_node=Degrees(0), arg_peri=None, M=None),
        Body(parent="Sol", name="Mercury", mass=Kilograms(3.3011e23), radius=Meters(2.4397e6), a=AU(0.38709927), e=0.20563593, I=Degrees(7.00497902), L=Degrees(252.25032350), long_peri=Degrees(77.45779628), long_node=Degrees(48.33076593), M=None, arg_peri=None),
        Body(parent="Sol", name="Venus", mass=Kilograms(4.8675e24), radius=Meters(6.0518e6), a=AU(0.72333566), e=0.00677672, I=Degrees(3.39467605), L=Degrees(181.97909950), long_peri=Degrees(131.60246718), long_node=Degrees(76.67984255), M=None, arg_peri=None),
        Body(parent="Sol", name="Earth", mass=Kilograms(5.9722e24), radius=Meters(6.371e6), a=AU(1.00000261), e=0.01671123, I=Degrees(-0.00001531), L=Degrees(100.46457166), long_peri=Degrees(102.93768193), long_node=Degrees(0.0), M=None, arg_peri=None),
        Body(parent="Sol", name="Mars", mass=Kilograms(6.4171e23), radius=Meters(3.3895e6), a=AU(1.52371034), e=0.09339410, I=Degrees(1.84969142), L=Degrees(-4.55343205), long_peri=Degrees(-23.94362959), long_node=Degrees(49.55953891), M=None, arg_peri=None),
        Body(parent="Sol", name="Jupiter", mass=Kilograms(1.8982e27), radius=Meters(6.9911e7), a=AU(5.20288700), e=0.04838624, I=Degrees(1.30439695), L=Degrees(34.39644051), long_peri=Degrees(14.72847983), long_node=Degrees(100.47390909), M=None, arg_peri=None),
        Body(parent="Sol", name="Saturn", mass=Kilograms(5.6834e26), radius=Meters(5.8232e7), a=AU(9.53667594), e=0.05386179, I=Degrees(2.48599187), L=Degrees(49.95424423), long_peri=Degrees(92.59887831), long_node=Degrees(113.66242448), M=None, arg_peri=None),
        Body(parent="Sol", name="Uranus", mass=Kilograms(8.6810e25), radius=Meters(2.5362e7), a=AU(19.18916464), e=0.04725744, I=Degrees(0.77263783), L=Degrees(313.23810451), long_peri=Degrees(170.95427630), long_node=Degrees(74.01692503), M=None, arg_peri=None),
        Body(parent="Sol", name="Neptune", mass=Kilograms(1.02413e26), radius=Meters(2.4622e7), a=AU(30.06992276), e=0.00859048, I=Degrees(1.77004347), L=Degrees(-55.12002969), long_peri=Degrees(44.96476227), long_node=Degrees(131.78422574), M=None, arg_peri=None),
        Body(parent="Sol", name="Pluto", mass=Kilograms(13024.6e18), radius=Meters(1188300), a=AU(39.5886), e=0.2518, I=Degrees(17.1477), L=Degrees(38.68366), long_peri=Degrees(113.709), long_node=Degrees(110.292), M=None, arg_peri=None),
        Body(parent="Sol", name="Ceres", mass=Kilograms(938.416e18), radius=Meters(469700), a=AU(2.766051), e=0.0794, I=Degrees(10.588), L=Degrees(188.70268), long_peri=Degrees(73.2734), long_node=Degrees(80.2522), M=None, arg_peri=None),
        Body(parent="Sol", name="Eris", mass=Kilograms(16600e18), radius=Meters(1163000), a=AU(68.0506), e=0.435675, I=Degrees(43.821), L=Degrees(211.032), long_peri=Degrees(150.714), long_node=Degrees(36.0460), M=None, arg_peri=None),
        Body(parent="Sol", name="20000 Varuna", mass=Kilograms(3.698e20), radius=Meters(334000), a=AU(43.1374), e=0.053565, I=Degrees(17.1395), L=Degrees(114.900), long_peri=Degrees(272.579), long_node=Degrees(97.21338), M=None, arg_peri=None),
        Body(parent="Sol", name="Makemake", mass=Kilograms(3100e18), radius=Meters(714000), a=AU(45.4494), e=0.16194, I=Degrees(29.03386), L=Degrees(168.8258), long_peri=Degrees(296.95), long_node=Degrees(79.259), M=None, arg_peri=None),
        Body(parent="Sol", name="28978 Ixion", mass=Kilograms(3e20), radius=Meters(355000), a=AU(39.3745), e=0.2449, I=Degrees(19.6745), L=Degrees(293.546), long_peri=Degrees(300.585), long_node=Degrees(71.099), M=None, arg_peri=None),
    ]
    if moons:
        bodies += [
            Body(parent="Earth", name="Moon", mass=Kilograms(7.346e22), radius=Meters(1.7371e6), a=AU(0.00257), e=0.0549, I=Degrees(5.16), arg_peri=Degrees(318.15), M=Degrees(135.27), long_node=Degrees(125.08), long_peri=None, L=None),
        ]
    return Dataset([x.to_dict() for x in bodies], **kwargs)
