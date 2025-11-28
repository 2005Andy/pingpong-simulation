"""Physical constants and parameters for ball sports simulation.

This module contains all adjustable physical and numerical parameters
used in ball trajectory simulations. Parameters are based on official
regulations and scientific literature.

References:
- ITTF Official Rules: Table dimensions, net height, ball specifications
- Aerodynamic coefficients based on published research on ball dynamics
"""

from typing import Final, List, Tuple
import numpy as np

# =============================================================================
# Global physical parameters (SI units) - Based on ITTF regulations
# =============================================================================

# Air and gravity
AIR_DENSITY: Final[float] = 1.225  # kg/m^3 (standard atmosphere at 15°C)
GRAVITY: Final[np.ndarray] = np.array([0.0, 0.0, -9.81])  # m/s^2

# Ball properties (ITTF standard: 40mm diameter, 2.67-2.77g mass)
BALL_RADIUS: Final[float] = 0.020  # 20 mm = 2.0 cm
BALL_MASS: Final[float] = 0.0027  # 2.7 g (ITTF standard)
DRAG_COEFF: Final[float] = 0.40  # C_D for table tennis ball (range: 0.36-0.50)
MAGNUS_COEFF: Final[float] = 0.20  # C_Omega (lift due to spin)
BALL_INERTIA_FACTOR: Final[float] = 2.0 / 3.0  # I = k m R^2, thin spherical shell

# Reference speeds (for model validation)
U_MAX: Final[float] = 32.0  # typical maximum ball speed (m/s) - professional smash
U_INF: Final[float] = 10.0  # reference air speed magnitude (m/s)
WIND_VELOCITY: Final[np.ndarray] = np.array([0.0, 0.0, 0.0])  # background air (m/s)

# Table properties (ITTF standard dimensions)
TABLE_LENGTH: Final[float] = 2.74  # m (x direction)
TABLE_WIDTH: Final[float] = 1.525  # m (y direction)
TABLE_HEIGHT: Final[float] = 0.76  # m (table surface height from ground)
TABLE_RESTITUTION: Final[float] = 0.90  # normal restitution coefficient e
TABLE_FRICTION: Final[float] = 0.25  # tangential friction coefficient mu

# Net properties (ITTF standard: 15.25 cm high)
NET_HEIGHT: Final[float] = 0.1525  # m (15.25 cm above table surface)
NET_LENGTH: Final[float] = 1.83  # m (including posts, extends beyond table width)
NET_THICKNESS: Final[float] = 0.002  # m (2 mm, for collision detection)

# Racket properties (standard paddle: ~17 cm diameter blade)
RACKET_RADIUS: Final[float] = 0.085  # m (8.5 cm radius = 17 cm diameter)
RACKET_THICKNESS: Final[float] = 0.006  # m (~6 mm total thickness with rubber)

# Rubber surface properties (different for inverted/pimpled rubber)
# Inverted (smooth) rubber - most common
RUBBER_INVERTED_RESTITUTION: Final[float] = 0.88
RUBBER_INVERTED_FRICTION: Final[float] = 0.45  # high friction for spin generation

# Pimpled (short pips) rubber - less spin
RUBBER_PIMPLED_RESTITUTION: Final[float] = 0.85
RUBBER_PIMPLED_FRICTION: Final[float] = 0.30

# Anti-spin rubber - minimal spin
RUBBER_ANTISPIN_RESTITUTION: Final[float] = 0.80
RUBBER_ANTISPIN_FRICTION: Final[float] = 0.15

# Default racket surface type
DEFAULT_RUBBER_TYPE: Final[str] = "inverted"

# Player positions (typical standing positions relative to table center)
PLAYER_A_X: Final[float] = -1.8  # m (behind table on negative x side)
PLAYER_B_X: Final[float] = 1.8   # m (behind table on positive x side)
PLAYER_STRIKE_HEIGHT: Final[float] = 1.0  # m (typical striking height above ground)

# Numerical integration parameters
TIME_STEP: Final[float] = 5.0e-5  # s (50 microseconds for accuracy)
MAX_TIME: Final[float] = 5.0  # s
RECORD_INTERVAL: Final[int] = 20  # record every N steps to reduce data size

# Output / visualization parameters
DEFAULT_BALL_CSV: Final[str] = "ball_trajectory.csv"
DEFAULT_RACKET_CSV: Final[str] = "racket_trajectory.csv"
DEFAULT_ANIM_FILE: Final[str] = "trajectory.mp4"
ANIM_FPS: Final[int] = 60
ANIM_SKIP: Final[int] = 100  # use every Nth frame for animation

# Default scenario name
DEFAULT_SCENARIO: Final[str] = "serve"

# Default output directory
DEFAULT_OUTPUT_DIR: Final[str] = "./output"

# Default custom initial conditions (typical serve position with topspin)
CUSTOM_INITIAL_POSITION: Final[List[float]] = [
    -TABLE_LENGTH / 2 - 0.2,  # x: behind table
    0.0,                       # y: centered
    TABLE_HEIGHT + 0.25,       # z: above table
]
CUSTOM_INITIAL_VELOCITY: Final[List[float]] = [5.0, 0.0, 2.0]  # forward and upward
CUSTOM_INITIAL_OMEGA: Final[List[float]] = [0.0, 100.0, 0.0]   # topspin around y-axis

# Custom stroke sequence for Player A (each rally)
# Format: [(stroke_type, rubber_type), ...]
CUSTOM_STROKES_A: Final[List[Tuple[str, str]]] = [
    ("topspin", "inverted"),
    ("topspin", "inverted"),
    ("backspin", "inverted"),
    ("topspin", "inverted"),
    ("flat", "inverted"),
]

# Custom stroke sequence for Player B
CUSTOM_STROKES_B: Final[List[Tuple[str, str]]] = [
    ("topspin", "inverted"),
    ("backspin", "pimpled"),
    ("topspin", "inverted"),
    ("sidespin", "inverted"),
    ("topspin", "inverted"),
]
