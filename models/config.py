from dataclasses import dataclass
import numpy as np


@dataclass
class DroneXZConfig:
    nx: int = 6
    nu: int = 2
    mass: float = 1.0
    l: float = 0.5
    gravity: float = 9.81

@dataclass
class DroneZConfig:
    nx: int = 2
    nu: int = 1
    mass: float = 1.0
    gravity: float = 9.81
    drag_coefficient: float = 0.5

@dataclass
class OmniBotXYConfig:
    nx: int = 4
    nu: int = 2
    safety_radius: float = 0.8


@dataclass
class BicycleXYConfig:
    nx: int = 3
    nu: int = 2
    lf: float = 0.5
    lr: float = 0.5
    safety_radius: float = 0.8