from __future__ import annotations

import math

AU_METERS = 1.495978707e11  # num meters in 1AU
KG_SOLAR = 1.98847e30  # num kg in 1 solar mass


class Unit:
    def __init__(self, value: float | int, unit: str):
        self.value = float(value)
        self.unit = unit
    
    def __repr__(self):
        return f"{self.unit.upper()}({self.value})"

class Radians(Unit):
    def __init__(self, value: float | int):
        super().__init__(value, "radians")
    
    def to_degrees(self) -> float:
        return Degrees(math.degrees(self.value))

class Degrees(Unit):
    def __init__(self, value: float | int):
        super().__init__(value, "degrees")
    
    def to_radians(self) -> float:
        return Radians(math.radians(self.value))

class Meters(Unit):
    def __init__(self, value: float | int):
        super().__init__(value, "meters")
    
    def to_au(self) -> float:
        return AU(self.value / AU_METERS)

class AU(Unit):
    def __init__(self, value: float | int):
        super().__init__(value, "au")
    
    def to_meters(self) -> float:
        return Meters(self.value * AU_METERS)

class Kilograms(Unit):
    def __init__(self, value: float | int):
        super().__init__(value, "kilograms")
    
    def to_solar_masses(self) -> float:
        return SolarMasses(self.value / KG_SOLAR)

class SolarMasses(Unit):
    def __init__(self, value: float | int):
        super().__init__(value, "m_solar")
    
    def to_kilograms(self) -> float:
        return Kilograms(self.value * KG_SOLAR)
    

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

    def convert_types(self) -> Dataset:
        standardized_data = []
        for row in self.data:
            standardized_data.append({k: self._convert(v) for k, v in row.items()})
        return Dataset(standardized_data, self.distance_unit, self.angle_unit)

    def values(self):
        return [{k: v.value if isinstance(v, Unit) else v for k, v in row.items()} for row in self.data]


def _make_keplerian_element(a, e, I, L, long_peri, long_node):
    """ Helper to convert from classical elements to our OrbitalElements dataclass."""
    return {
        "a": AU(a),
        "e": e,
        "I": Degrees(I),
        "L": Degrees(L),
        "long.peri": Degrees(long_peri),
        "long.node": Degrees(long_node)
    }

# Source: NASA/JPL: https://ssd.jpl.nasa.gov/planets/approx_pos.html
# with reference to to the mean ecliptic and equinox of J2000, valid for the time-interval 1800 AD - 2050 AD
def solar_keplerian_elements():
    return Dataset([
        {"name": "Mercury", **_make_keplerian_element(0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593)},
        {"name": "Venus", **_make_keplerian_element(0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255)},
        {"name": "EMB", **_make_keplerian_element(1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0)},  # Earth Moon Barycenter
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
        {"name": "Eris", **_make_keplerian_element(68.0506, 0.435675, 43.821, 211.032, 150.714, 36.0460)}
    ])


# Source: NASA/JPL https://ssd.jpl.nasa.gov/planets/phys_par.html
def solar_physical_properties():
    return Dataset([
        {"name": "Sun", "mass": Kilograms(1.9885e30), "radius": Meters(6.9634e8), "fg": 274.},
        {"name": "Mercury", "mass": Kilograms(3.3011e23), "radius": Meters(2.4397e6), "fg": 3.70},
        {"name": "Venus", "mass": Kilograms(4.8675e24), "radius": Meters(6.0518e6), "fg": 8.87},
        {"name": "Earth", "mass": Kilograms(5.9722e24), "radius": Meters(6.3710e6), "fg": 9.80},
        {"name": "Mars", "mass": Kilograms(6.4171e23), "radius": Meters(3.3895e6), "fg": 3.71},
        {"name": "Jupiter", "mass": Kilograms(1.8982e27), "radius": Meters(6.9911e7), "fg": 24.79},
        {"name": "Saturn", "mass": Kilograms(5.6834e26), "radius": Meters(5.8232e7), "fg": 10.44},
        {"name": "Uranus", "mass": Kilograms(8.6810e25), "radius": Meters(2.5362e7), "fg": 8.87},
        {"name": "Neptune", "mass": Kilograms(1.02413e26), "radius": Meters(2.4622e7), "fg": 11.15},
        {"name": "Pluto", "mass": Kilograms(13024.6e18), "radius": Meters(1188.3e4), "fg": 0.62},
        {"name": "Ceres", "mass": Kilograms(938.416e18), "radius": Meters(469.7e4), "fg": 0.27},
        {"name": "Eris", "mass": Kilograms(16600e18), "radius": Meters(1200.0e4), "fg": 0.77},
    ])
