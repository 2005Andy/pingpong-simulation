"""
PingPong Simulation Package

A physics-based 3D ping-pong ball trajectory simulation system.
"""

__version__ = "1.0.0"
__author__ = "PingPong Simulation Team"

from . import constants
from . import ball_types
from . import physics
from . import simulation
from . import racket_control
from . import scenarios
from . import visualization

__all__ = [
    'constants',
    'ball_types',
    'physics',
    'simulation',
    'racket_control',
    'scenarios',
    'visualization',
]
