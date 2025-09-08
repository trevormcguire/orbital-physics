from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


AU = 1.495978707e11  # m
DAY = 86400.  # s
# The standard celestial coordinate frame:
# from https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/MPG%20Book/Release/Chapter7-OrbitalMechanics.pdf:
#   "it is possible to obtain a standard celestial coordinate frame that is fixed in space
#    by fixing the orientation of a chosen inertial coordinate frame at a specified instant,
#    called the standard epoch"
# The standard epoch is J2000, defined by the positions of the Earth's equator and equinox
# on Julian Day 2451545.0, or January 1, 2000 at 12:00:00.
JULIAN_DAY = 86400.  # s
J2000_JD = 2451545.  # Julian Date of J2000 epoch


class UnitSystem(str, Enum):
    ASTRO = "astro"  # AU, M_sun, day
    SI    = "si"  # m, kg, s

@dataclass(frozen=True)
class UnitProfile:
    name: UnitSystem
    G: float  # Gravatational constant for this unit system
    distance_unit: str
    mass_unit: str
    time_unit: str
    # Useful conversion anchors (identity in ASTRO)
    AU: float  # 1 AU in distance units
    M_SUN: float  # 1 M_sun in mass units
    DAY: float  # 1 day in time units

# Astronomical units
ASTRO = UnitProfile(
    name=UnitSystem.ASTRO,
    G=0.0002959122082855911,  # AU^3 / (M_sun * day^2)
    distance_unit="AU",
    mass_unit="M_sun",
    time_unit="day",
    AU=1.0,
    M_SUN=1.0,
    DAY=1.0,
)

# Standard units (general physics)
STANDARD = UnitProfile(
    name=UnitSystem.SI,
    G=6.67430e-11,  # m^3 / (kg * s^2)
    distance_unit="m",
    mass_unit="kg",
    time_unit="s",
    AU=1.495978707e11,  # meters
    M_SUN=1.98847e30,  # kg
    DAY=86400.0,  # seconds
)

@dataclass(frozen=True)  # frozen makes it immutable
class IntegratorParams:
    softening: float  # in *distance units* of the chosen profile
    dt: float  # time step in *time units*


# Defaults your engine will import
DEFAULT_STANDARD_INTEGRATOR = IntegratorParams(dt=60*60, softening=1.)  # 1 hour, 1 meter
DEFAULT_ASTRO_INTEGRATOR = IntegratorParams(dt=1.0, softening=1e-6)  # 1 day, 1 micro-AU


def get_unit_profile(name: str | UnitSystem) -> UnitProfile:
    """Get a UnitProfile by name."""
    if isinstance(name, str):
        name = UnitSystem(name.lower())
    if name == UnitSystem.ASTRO:
        return ASTRO
    elif name == UnitSystem.SI:
        return STANDARD
    else:
        raise ValueError(f"Unknown unit system: {name}")
