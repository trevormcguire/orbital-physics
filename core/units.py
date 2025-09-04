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
        value = value % (2 * math.pi)  # normalize
        super().__init__(value, "radians")
    
    def to_degrees(self) -> float:
        return Degrees(math.degrees(self.value))

class Degrees(Unit):
    def __init__(self, value: float | int):
        value = value % 360  # normalize
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
