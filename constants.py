"""Physical constants and parameters for ball sports simulation.

This module contains all adjustable physical and numerical parameters
used in ball trajectory simulations. Parameters are based on official
regulations and scientific literature.

References:
- ITTF Official Rules: Table dimensions, net height, ball specifications
- Aerodynamic coefficients based on published research on ball dynamics
"""

from typing import Final, List, Tuple, Dict, Any
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
TABLE_SURFACE_COLOR: Final[str] = "#003366"  # Standard ping-pong table dark blue
TABLE_STRIPE_COLOR: Final[str] = "#003366"  # White stripes on table
TABLE_EDGE_COLOR: Final[str] = "#ffffff"    # White edges
TABLE_LEG_COLOR: Final[str] = "#1d1e24"
TABLE_LEG_WIDTH: Final[float] = 0.04  # m
TABLE_LEG_HEIGHT: Final[float] = TABLE_HEIGHT  # m, leg height from ground to tabletop underside

# Center line rendering width (physical width in meters on table surface)
TABLE_CENTER_LINE_WIDTH: Final[float] = 0.015  # 15 mm wide white center line

# Net properties (ITTF standard: 15.25 cm high)
NET_HEIGHT: Final[float] = 0.1525  # m (15.25 cm above table surface)
NET_LENGTH: Final[float] = 1.83  # m (including posts, extends beyond table width)
NET_THICKNESS: Final[float] = 0.002  # m (2 mm, for collision detection)

# Racket properties (standard paddle: ~17 cm diameter blade)
RACKET_RADIUS: Final[float] = 0.085  # m (8.5 cm radius = 17 cm diameter)
RACKET_THICKNESS: Final[float] = 0.006  # m (~6 mm total thickness with rubber)
RACKET_MAX_SPEED: Final[float] = 1.0  # m/s, maximum allowed racket center speed
RACKET_BASE_SPEED: Final[float] = 0.6  # m/s, nominal center speed for planning
RACKET_MIN_MOVEMENT_DURATION: Final[float] = 0.08  # s, lower bound for easing duration

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

# Player positions (typical striking positions relative to table center)
PLAYER_A_X: Final[float] = -TABLE_LENGTH / 2  # m (near center on negative x side)
PLAYER_B_X: Final[float] = TABLE_LENGTH / 2   # m (near center on positive x side)
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
ANIM_SKIP: Final[int] = 10  # use every Nth frame for animation
DEFAULT_BALL_COLOR: Final[str] = "#f97306"
DEFAULT_BALL_SIZE: Final[float] = 18.0  # matplotlib marker size
DEFAULT_SCENE_MARGIN: Final[float] = 0.15  # meters of padding for visualization axes (tight view)

# Racket interception / planning parameters
RACKET_PREDICTION_DT_MULTIPLIER: Final[float] = 8.0
RACKET_PREDICTION_MIN_DT: Final[float] = 8.0e-4
RACKET_PREDICTION_MAX_TIME: Final[float] = 0.6  # s, prediction horizon after bounce
RACKET_STRIKE_HEIGHT_WINDOW: Final[float] = 0.22  # m, tolerance around desired strike height
RACKET_STRIKE_X_WINDOW: Final[float] = 0.30  # m, tolerance around target_x strike plane

# Default scenario name
DEFAULT_SCENARIO: Final[str] = "custom"
DEFAULT_SERVE_MODE: Final[str] = "fh_under"
DEFAULT_SERVER: Final[str] = "A"

# Default output directory
DEFAULT_OUTPUT_DIR: Final[str] = "./output"

# Default custom initial conditions (typical serve position with topspin)
CUSTOM_INITIAL_POSITION: Final[List[float]] = [
    -TABLE_LENGTH / 2 - 0.15,  # x: behind table
    0.4,                       # y: centered
    TABLE_HEIGHT + 0.20,       # z: above table
]
CUSTOM_INITIAL_VELOCITY: Final[List[float]] = [5.5, -0.5, -2.0]  # forward and upward
CUSTOM_INITIAL_OMEGA: Final[List[float]] = [0.0, 100.0, 0.0]   # topspin around y-axis

# Custom stroke sequence for Player A (each rally)
# Format: List[Dict[str, Any]] with keys: mode, rubber, overrides(optional)
CUSTOM_STROKES_A: Final[List[Dict[str, Any]]] = [
    {"mode": "flick", "rubber": "inverted"},
    {"mode": "flick", "rubber": "inverted"},
    {"mode": "drop_short", "rubber": "inverted"},
    {"mode": "counter_loop", "rubber": "inverted"},
    {"mode": "counter_loop", "rubber": "inverted"},
]

# Custom stroke sequence for Player B
CUSTOM_STROKES_B: Final[List[Dict[str, Any]]] = [
    {"mode": "flick", "rubber": "inverted"},
    {"mode": "flick", "rubber": "inverted"},
    {"mode": "counter_loop", "rubber": "inverted"},
    {"mode": "flick", "rubber": "inverted"},
    {"mode": "counter_loop", "rubber": "inverted"},
]
