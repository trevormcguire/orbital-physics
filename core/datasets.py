from __future__ import annotations

from core.body import Body, System
from core.constants import J2000_JD, STANDARD
from core.units import Meters, AU, Degrees, Kilograms


G = STANDARD.G
# https://en.wikipedia.org/wiki/Epoch_(astronomy)#J2000
EPOCH = J2000_JD


def solar_system_v2(moons: bool = False, **kwargs):
    sol = Body(parent=None, name="Sol", mass=Kilograms(1.9885e30), radius=Meters(6.9634e8), a=AU(0), e=0, I=Degrees(0), L=Degrees(0), long_peri=Degrees(0), long_node=Degrees(0), arg_peri=None, M=None)
    earth = Body(parent=sol, name="Earth", mass=Kilograms(5.9722e24), radius=Meters(6.371e6), a=AU(1.00000261), e=0.01671123, I=Degrees(-0.00001531), L=Degrees(100.46457166), long_peri=Degrees(102.93768193), long_node=Degrees(0.0), M=None, arg_peri=None)
    jupiter = Body(parent=sol, name="Jupiter", mass=Kilograms(1.8982e27), radius=Meters(6.9911e7), a=AU(5.20288700), e=0.04838624, I=Degrees(1.30439695), L=Degrees(34.39644051), long_peri=Degrees(14.72847983), long_node=Degrees(100.47390909), M=None, arg_peri=None)
    saturn = Body(parent=sol, name="Saturn", mass=Kilograms(5.6834e26), radius=Meters(5.8232e7), a=AU(9.53667594), e=0.05386179, I=Degrees(2.48599187), L=Degrees(49.95424423), long_peri=Degrees(92.59887831), long_node=Degrees(113.66242448), M=None, arg_peri=None)
    uranus = Body(parent=sol, name="Uranus", mass=Kilograms(8.6810e25), radius=Meters(2.5362e7), a=AU(19.18916464), e=0.04725744, I=Degrees(0.77263783), L=Degrees(313.23810451), long_peri=Degrees(170.95427630), long_node=Degrees(74.01692503), M=None, arg_peri=None)
    neptune = Body(parent=sol, name="Neptune", mass=Kilograms(1.02413e26), radius=Meters(2.4622e7), a=AU(30.06992276), e=0.00859048, I=Degrees(1.77004347), L=Degrees(-55.12002969), long_peri=Degrees(44.96476227), long_node=Degrees(131.78422574), M=None, arg_peri=None)

    bodies = [
        sol,
        Body(parent=sol, name="Mercury", mass=Kilograms(3.3011e23), radius=Meters(2.4397e6), a=AU(0.38709927), e=0.20563593, I=Degrees(7.00497902), L=Degrees(252.25032350), long_peri=Degrees(77.45779628), long_node=Degrees(48.33076593), M=None, arg_peri=None),
        Body(parent=sol, name="Venus", mass=Kilograms(4.8675e24), radius=Meters(6.0518e6), a=AU(0.72333566), e=0.00677672, I=Degrees(3.39467605), L=Degrees(181.97909950), long_peri=Degrees(131.60246718), long_node=Degrees(76.67984255), M=None, arg_peri=None),
        earth,
        Body(parent=sol, name="Mars", mass=Kilograms(6.4171e23), radius=Meters(3.3895e6), a=AU(1.52371034), e=0.09339410, I=Degrees(1.84969142), L=Degrees(-4.55343205), long_peri=Degrees(-23.94362959), long_node=Degrees(49.55953891), M=None, arg_peri=None),
        jupiter,
        saturn,
        uranus,
        neptune,
        Body(parent=sol, name="Pluto", mass=Kilograms(13024.6e18), radius=Meters(1188300), a=AU(39.5886), e=0.2518, I=Degrees(17.1477), L=Degrees(38.68366), long_peri=Degrees(113.709), long_node=Degrees(110.292), M=None, arg_peri=None),
        Body(parent=sol, name="Ceres", mass=Kilograms(938.416e18), radius=Meters(469700), a=AU(2.766051), e=0.0794, I=Degrees(10.588), L=Degrees(188.70268), long_peri=Degrees(73.2734), long_node=Degrees(80.2522), M=None, arg_peri=None),
        Body(parent=sol, name="Eris", mass=Kilograms(16600e18), radius=Meters(1163000), a=AU(68.0506), e=0.435675, I=Degrees(43.821), L=Degrees(211.032), long_peri=Degrees(150.714), long_node=Degrees(36.0460), M=None, arg_peri=None),
        Body(parent=sol, name="20000 Varuna", mass=Kilograms(3.698e20), radius=Meters(334000), a=AU(43.1374), e=0.053565, I=Degrees(17.1395), L=Degrees(114.900), long_peri=Degrees(272.579), long_node=Degrees(97.21338), M=None, arg_peri=None),
        Body(parent=sol, name="Makemake", mass=Kilograms(3100e18), radius=Meters(714000), a=AU(45.4494), e=0.16194, I=Degrees(29.03386), L=Degrees(168.8258), long_peri=Degrees(296.95), long_node=Degrees(79.259), M=None, arg_peri=None),
        Body(parent=sol, name="28978 Ixion", mass=Kilograms(3e20), radius=Meters(355000), a=AU(39.3745), e=0.2449, I=Degrees(19.6745), L=Degrees(293.546), long_peri=Degrees(300.585), long_node=Degrees(71.099), M=None, arg_peri=None),
    ]
    if moons:
        # https://ssd.jpl.nasa.gov/sats/elem/
        bodies += [
            Body(parent=earth, name="Luna", mass=Kilograms(7.346e22), radius=Meters(1.7371e6), a=AU(0.00257), e=0.0549, I=Degrees(5.16), arg_peri=Degrees(318.15), M=Degrees(135.27), long_node=Degrees(125.08), long_peri=None, L=None),
            Body(parent=jupiter, name="Io", mass=Kilograms(8.93e22), radius=Meters(1_821_600), a=Meters(421_800_000).to_au(), e=0.004, I=Degrees(0.), arg_peri=Degrees(49.1), M=Degrees(330.9), long_node=Degrees(0.), long_peri=None, L=None),
            Body(parent=jupiter, name="Europa", mass=Kilograms(4.8e22), radius=Meters(1_560_800), a=Meters(671_100_000).to_au(), e=0.009, I=Degrees(0.5), arg_peri=Degrees(45.0), M=Degrees(345.4), long_node=Degrees(184.0), long_peri=None, L=None),
            Body(parent=jupiter, name="Ganymede", mass=Kilograms(1.4819e23), radius=Meters(2_634_100), a=Meters(1_070_400_000).to_au(), e=0.001, I=Degrees(0.2), arg_peri=Degrees(198.3), M=Degrees(324.8), long_node=Degrees(58.5), long_peri=None, L=None),
            Body(parent=jupiter, name="Callisto", mass=Kilograms(1.08e23), radius=Meters(1_560_800), a=Meters(1_882_700_000).to_au(), e=0.007, I=Degrees(0.3), arg_peri=Degrees(43.8), M=Degrees(87.4), long_node=Degrees(309.1), long_peri=None, L=None),
            # https://en.wikipedia.org/wiki/Moons_of_Saturn
            Body(parent=saturn, name="Titan", mass=Kilograms(1.345e23), radius=Meters(2_575_000), a=Meters(1_221_900_000).to_au(), e=0.029, I=Degrees(0.35), arg_peri=Degrees(78.3), M=Degrees(11.7), long_node=Degrees(78.6), long_peri=None, L=None),
            Body(parent=saturn, name="Enceladus", mass=Kilograms(1.08e20), radius=Meters(252_000), a=Meters(238_400_000).to_au(), e=0.005, I=Degrees(0.0), arg_peri=Degrees(119.5), M=Degrees(57.0), long_node=Degrees(0.0), long_peri=None, L=None),
            Body(parent=saturn, name="Rhea", mass=Kilograms(2.31e21), radius=Meters(763_800), a=Meters(527_200_000).to_au(), e=0.001, I=Degrees(0.3), arg_peri=Degrees(44.3), M=Degrees(31.5), long_node=Degrees(133.7), long_peri=None, L=None),
            Body(parent=saturn, name="Iapetus", mass=Kilograms(1.805e21), radius=Meters(734_400), a=Meters(3_561_700_000).to_au(), e=0.028, I=Degrees(7.6), arg_peri=Degrees(254.5), M=Degrees(74.8), long_node=Degrees(86.5), long_peri=None, L=None),
            # https://en.wikipedia.org/wiki/Moons_of_Neptune
            Body(parent=neptune, name="Triton", mass=Kilograms(2.14e22), radius=Meters(1_353_400), a=Meters(354_800_000).to_au(), e=0.0, I=Degrees(157.3), arg_peri=Degrees(0.0), M=Degrees(63.0), long_node=Degrees(178.1), long_peri=None, L=None),
            # https://en.wikipedia.org/wiki/Titania_(moon)
            Body(parent=uranus, name="Titania", mass=Kilograms(3.455e21), radius=Meters(788_400), a=Meters(436_298_000).to_au(), e=0.002, I=Degrees(0.1), arg_peri=Degrees(184.0), M=Degrees(68.1), long_node=Degrees(29.5), long_peri=None, L=None),
        ]
    return System(bodies, **kwargs)

solar_system = solar_system_v2  # backwards compatibility alias
