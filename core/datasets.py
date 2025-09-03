from __future__ import annotations

import math

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
        return AU(self.value / 1.495978707e11)

class AU(Unit):
    def __init__(self, value: float | int):
        super().__init__(value, "au")
    
    def to_meters(self) -> float:
        return Meters(self.value * 1.495978707e11)


class Dataset(object):
    def __init__(self, data: list[dict], distance_unit: str = "meters", angle_unit: str = "radians"):
        self.data = data
        self.distance_unit = distance_unit
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
