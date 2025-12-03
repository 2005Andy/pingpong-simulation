"""3D ping-pong ball flight simulation with table, net, and dual racket collisions.

All adjustable physical and numerical parameters are collected at the top of this file
for easy modification. Parameters are based on ITTF official regulations and scientific
literature.

References:
- ITTF Official Rules: Table dimensions, net height, ball specifications
- Aerodynamic coefficients based on published research on table tennis ball dynamics
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.patches import Circle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =============================================================================
# Global physical parameters (SI units) - Based on ITTF regulations
# =============================================================================

# Air and gravity
AIR_DENSITY: float = 1.225  # kg/m^3 (standard atmosphere at 15°C)
GRAVITY: np.ndarray = np.array([0.0, 0.0, -9.81])  # m/s^2

# Ball properties (ITTF standard: 40mm diameter, 2.67-2.77g mass)
BALL_RADIUS: float = 0.020  # 20 mm = 2.0 cm
BALL_MASS: float = 0.0027  # 2.7 g (ITTF standard)
DRAG_COEFF: float = 0.40  # C_D for table tennis ball (range: 0.36-0.50)
MAGNUS_COEFF: float = 0.20  # C_Omega (lift due to spin)
BALL_INERTIA_FACTOR: float = 2.0 / 3.0  # I = k m R^2, thin spherical shell

# Reference speeds (for model validation)
U_MAX: float = 32.0  # typical maximum ball speed (m/s) - professional smash
U_INF: float = 10.0  # reference air speed magnitude (m/s)
WIND_VELOCITY: np.ndarray = np.array([0.0, 0.0, 0.0])  # background air (m/s)

# Table properties (ITTF standard dimensions)
TABLE_LENGTH: float = 2.74  # m (x direction)
TABLE_WIDTH: float = 1.525  # m (y direction)
TABLE_HEIGHT: float = 0.76  # m (table surface height from ground)
TABLE_RESTITUTION: float = 0.90  # normal restitution coefficient e
TABLE_FRICTION: float = 0.25  # tangential friction coefficient mu

# Net properties (ITTF standard: 15.25 cm high)
NET_HEIGHT: float = 0.1525  # m (15.25 cm above table surface)
NET_LENGTH: float = 1.83  # m (including posts, extends beyond table width)
NET_THICKNESS: float = 0.002  # m (2 mm, for collision detection)

# Racket properties (standard paddle: ~17 cm diameter blade)
RACKET_RADIUS: float = 0.085  # m (8.5 cm radius = 17 cm diameter)
RACKET_THICKNESS: float = 0.006  # m (~6 mm total thickness with rubber)

# Rubber surface properties (different for inverted/pimpled rubber)
# Inverted (smooth) rubber - most common
RUBBER_INVERTED_RESTITUTION: float = 0.88
RUBBER_INVERTED_FRICTION: float = 0.45  # high friction for spin generation

# Pimpled (short pips) rubber - less spin
RUBBER_PIMPLED_RESTITUTION: float = 0.85
RUBBER_PIMPLED_FRICTION: float = 0.30

# Anti-spin rubber - minimal spin
RUBBER_ANTISPIN_RESTITUTION: float = 0.80
RUBBER_ANTISPIN_FRICTION: float = 0.15

# Default racket surface type
DEFAULT_RUBBER_TYPE: str = "inverted"

# Player positions (typical standing positions relative to table center)
PLAYER_A_X: float = -1.8  # m (behind table on negative x side)
PLAYER_B_X: float = 1.8   # m (behind table on positive x side)
PLAYER_STRIKE_HEIGHT: float = 1.0  # m (typical striking height above ground)

# Numerical integration parameters
TIME_STEP: float = 5.0e-5  # s (50 microseconds for accuracy)
MAX_TIME: float = 5.0  # s
RECORD_INTERVAL: int = 20  # record every N steps to reduce data size

# Output / visualization parameters
DEFAULT_BALL_CSV: str = "ball_trajectory.csv"
DEFAULT_RACKET_CSV: str = "racket_trajectory.csv"
DEFAULT_ANIM_FILE: str = "trajectory.mp4"
ANIM_FPS: int = 60
ANIM_SKIP: int = 100  # use every Nth frame for animation

# Default scenario name
DEFAULT_SCENARIO: str = "serve"

# Default output directory
DEFAULT_OUTPUT_DIR: str = "./output"


# =============================================================================
# Enumerations
# =============================================================================


class RubberType(Enum):
    """Types of racket rubber surface."""
    INVERTED = auto()  # smooth, high spin
    PIMPLED = auto()   # short pips, medium spin
    ANTISPIN = auto()  # anti-spin, low spin


class EventType(Enum):
    """Types of collision/crossing events."""
    NONE = 0
    TABLE_BOUNCE = 1
    RACKET_A_HIT = 2
    RACKET_B_HIT = 3
    NET_HIT = 4
    NET_CROSS_SUCCESS = 5
    NET_CROSS_FAIL = 6  # ball went under net or hit net
    OUT_OF_BOUNDS = 7


class Player(Enum):
    """Player identifier."""
    A = auto()  # negative x side
    B = auto()  # positive x side


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class BallState:
    """State of the ping-pong ball at a given time."""
    position: np.ndarray  # shape (3,)
    velocity: np.ndarray  # shape (3,)
    omega: np.ndarray     # angular velocity vector, shape (3,)

    def copy(self) -> "BallState":
        return BallState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            omega=self.omega.copy(),
        )


@dataclass
class Table:
    """Table modeled as an axis-aligned rectangular plane."""
    height: float      # z coordinate of table surface
    length: float      # extent in x direction
    width: float       # extent in y direction
    restitution: float
    friction: float


@dataclass
class Net:
    """Net at the center of the table."""
    height: float      # height above table surface
    length: float      # extent in y direction
    thickness: float   # for collision detection
    x_position: float  # x coordinate (center of table = 0)


@dataclass
class RacketMovement:
    """State of racket movement for smooth transitions."""
    is_moving: bool = False
    target_position: Optional[np.ndarray] = None
    target_normal: Optional[np.ndarray] = None
    target_velocity: Optional[np.ndarray] = None
    start_position: Optional[np.ndarray] = None
    start_normal: Optional[np.ndarray] = None
    movement_time: float = 0.0  # total time for movement
    elapsed_time: float = 0.0   # time elapsed in movement
    racket_speed: float = 3.0   # m/s, speed at which racket moves to position


@dataclass
class RacketState:
    """State of a racket at a given time."""
    position: np.ndarray   # center position, shape (3,)
    normal: np.ndarray     # unit normal of striking face, shape (3,)
    velocity: np.ndarray   # translational velocity, shape (3,)
    radius: float          # radius of circular racket face
    rubber_type: RubberType
    restitution: float
    friction: float
    player: Player
    movement: RacketMovement = field(default_factory=RacketMovement)  # movement state

    def copy(self) -> "RacketState":
        return RacketState(
            position=self.position.copy(),
            normal=self.normal.copy(),
            velocity=self.velocity.copy(),
            radius=self.radius,
            rubber_type=self.rubber_type,
            restitution=self.restitution,
            friction=self.friction,
            player=self.player,
            movement=RacketMovement(
                is_moving=self.movement.is_moving,
                target_position=self.movement.target_position.copy() if self.movement.target_position is not None else None,
                target_normal=self.movement.target_normal.copy() if self.movement.target_normal is not None else None,
                target_velocity=self.movement.target_velocity.copy() if self.movement.target_velocity is not None else None,
                start_position=self.movement.start_position.copy() if self.movement.start_position is not None else None,
                start_normal=self.movement.start_normal.copy() if self.movement.start_normal is not None else None,
                movement_time=self.movement.movement_time,
                elapsed_time=self.movement.elapsed_time,
                racket_speed=self.movement.racket_speed,
            ),
        )


@dataclass
class StrokeParams:
    """Parameters for a single stroke (hit)."""
    target_x: float           # x position to intercept ball
    strike_height: float      # z height of strike point
    racket_angle: float       # angle of racket face (radians, 0 = vertical)
    swing_speed: float        # speed of racket swing (m/s)
    swing_direction: np.ndarray  # unit vector of swing direction
    rubber_type: RubberType   # rubber surface type for this stroke
    spin_intent: str          # "topspin", "backspin", "sidespin", "flat"


@dataclass
class RallyConfig:
    """Configuration for a rally (sequence of strokes)."""
    strokes_a: List[StrokeParams]  # strokes for player A
    strokes_b: List[StrokeParams]  # strokes for player B
    max_rallies: int = 10          # maximum number of hits


# =============================================================================
# Default stroke configurations
# =============================================================================


def get_rubber_properties(rubber_type: RubberType) -> Tuple[float, float]:
    """Get restitution and friction for a rubber type."""
    if rubber_type == RubberType.INVERTED:
        return RUBBER_INVERTED_RESTITUTION, RUBBER_INVERTED_FRICTION
    elif rubber_type == RubberType.PIMPLED:
        return RUBBER_PIMPLED_RESTITUTION, RUBBER_PIMPLED_FRICTION
    else:
        return RUBBER_ANTISPIN_RESTITUTION, RUBBER_ANTISPIN_FRICTION


def create_default_stroke(
    player: Player,
    stroke_type: str = "topspin",
    rubber_type: RubberType = RubberType.INVERTED,
) -> StrokeParams:
    """Create a default stroke configuration for a player."""
    # Determine x position based on player
    if player == Player.A:
        target_x = -TABLE_LENGTH * 0.3
        swing_dir = np.array([1.0, 0.0, 0.3])  # towards positive x with slight upward
    else:
        target_x = TABLE_LENGTH * 0.3
        swing_dir = np.array([-1.0, 0.0, 0.3])  # towards negative x with slight upward

    swing_dir = swing_dir / np.linalg.norm(swing_dir)

    # Adjust parameters based on stroke type
    if stroke_type == "topspin":
        angle = np.radians(15)  # slightly closed racket
        speed = 8.0
    elif stroke_type == "backspin":
        angle = np.radians(-20)  # open racket
        speed = 5.0
        swing_dir[2] = -0.2  # downward component
        swing_dir = swing_dir / np.linalg.norm(swing_dir)
    elif stroke_type == "sidespin":
        angle = np.radians(5)
        speed = 7.0
        swing_dir[1] = 0.5  # sideways component
        swing_dir = swing_dir / np.linalg.norm(swing_dir)
    else:  # flat
        angle = np.radians(0)
        speed = 10.0

    return StrokeParams(
        target_x=target_x,
        strike_height=TABLE_HEIGHT + 0.25,  # 25 cm above table
        racket_angle=angle,
        swing_speed=speed,
        swing_direction=swing_dir,
        rubber_type=rubber_type,
        spin_intent=stroke_type,
    )


# =============================================================================
# Physics utilities
# =============================================================================


def aerodynamic_acceleration(velocity: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Compute acceleration due to gravity, drag and Magnus effect.

    Based on the equation:
    m * dU/dt = m*g - 0.5*rho*U^2*S*C_D*U_hat + rho*R^3*C_Omega*(omega x U)

    Args:
        velocity: Ball velocity in world frame.
        omega: Ball angular velocity.

    Returns:
        Acceleration vector (m/s^2).
    """
    v_rel = velocity - WIND_VELOCITY
    speed = np.linalg.norm(v_rel)
    if speed < 1.0e-8:
        return GRAVITY.copy()

    area = np.pi * BALL_RADIUS**2

    # Drag opposite to relative velocity
    drag_dir = -v_rel / speed
    drag_mag = 0.5 * AIR_DENSITY * speed**2 * area * DRAG_COEFF / BALL_MASS
    drag = drag_mag * drag_dir

    # Magnus effect (lift due to spin)
    # F_magnus = rho * R^3 * C_Omega * (omega x v_rel)
    magnus = (
        AIR_DENSITY
        * (BALL_RADIUS**3)
        * MAGNUS_COEFF
        / BALL_MASS
        * np.cross(omega, v_rel)
    )

    return GRAVITY + drag + magnus


def rk4_step(state: BallState, dt: float) -> BallState:
    """Advance state by one step using 4th-order Runge-Kutta (spin held constant)."""

    def acceleration(v: np.ndarray) -> np.ndarray:
        return aerodynamic_acceleration(v, state.omega)

    x0 = state.position
    v0 = state.velocity

    a1 = acceleration(v0)
    k1_v = a1 * dt
    k1_x = v0 * dt

    a2 = acceleration(v0 + 0.5 * k1_v)
    k2_v = a2 * dt
    k2_x = (v0 + 0.5 * k1_v) * dt

    a3 = acceleration(v0 + 0.5 * k2_v)
    k3_v = a3 * dt
    k3_x = (v0 + 0.5 * k2_v) * dt

    a4 = acceleration(v0 + k3_v)
    k4_v = a4 * dt
    k4_x = (v0 + k3_v) * dt

    v_new = v0 + (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0
    x_new = x0 + (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x) / 6.0

    return BallState(position=x_new, velocity=v_new, omega=state.omega.copy())


def _orthonormal_basis_from_normal(
    normal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct orthonormal basis (t1, t2, n) from a given normal."""
    n = normal / np.linalg.norm(normal)
    if abs(n[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, ref)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2)
    return t1, t2, n


def handle_plane_collision(
    state: BallState,
    plane_point: np.ndarray,
    normal: np.ndarray,
    restitution: float,
    friction: float,
    surface_velocity: np.ndarray,
) -> bool:
    """Handle collision between the ball and a planar surface.

    Uses impulse-based collision with Coulomb friction model.
    Supports sliding/sticking/over-spin states.

    Args:
        state: Ball state to be updated in place.
        plane_point: A point on the plane.
        normal: Plane normal (will be normalized).
        restitution: Normal restitution coefficient e.
        friction: Tangential friction coefficient mu.
        surface_velocity: Translational velocity of the surface.

    Returns:
        True if a collision impulse was applied, False otherwise.
    """
    n = normal / np.linalg.norm(normal)
    d = np.dot(state.position - plane_point, n) - BALL_RADIUS
    v_rel = state.velocity - surface_velocity
    v_rel_n = np.dot(v_rel, n)

    if d > 0.0 or v_rel_n >= 0.0:
        return False

    # Move ball just outside the plane
    state.position -= d * n

    m = BALL_MASS
    k = BALL_INERTIA_FACTOR
    r = -BALL_RADIUS * n  # vector from center to contact point

    # Relative velocity at contact point (including rotation)
    u = v_rel + np.cross(state.omega, r)
    u_n_scalar = np.dot(u, n)
    u_n = u_n_scalar * n
    u_t = u - u_n

    # Normal impulse
    J_n_mag = -(1.0 + restitution) * m * v_rel_n

    # Tangential impulse with friction
    J_t = np.zeros(3)
    u_t_norm = np.linalg.norm(u_t)
    if u_t_norm > 1.0e-8:
        denom = 1.0 + 1.0 / k
        J_t_stick = -(m / denom) * u_t
        if np.linalg.norm(J_t_stick) <= friction * abs(J_n_mag):
            J_t = J_t_stick  # sticking/rolling
        else:
            J_t = -friction * abs(J_n_mag) * (u_t / u_t_norm)  # sliding

    J = J_n_mag * n + J_t

    # Update velocities
    state.velocity += J / m
    I_scalar = k * m * (BALL_RADIUS**2)
    state.omega += np.cross(r, J) / I_scalar

    return True


def handle_circular_racket_collision(
    ball_state: BallState,
    racket: RacketState,
) -> bool:
    """Handle collision between ball and circular racket.

    Args:
        ball_state: Ball state to be updated in place.
        racket: Racket state.

    Returns:
        True if collision occurred, False otherwise.
    """
    n = racket.normal / np.linalg.norm(racket.normal)

    # Distance from ball center to racket plane
    d = np.dot(ball_state.position - racket.position, n) - BALL_RADIUS
    v_rel = ball_state.velocity - racket.velocity
    v_rel_n = np.dot(v_rel, n)

    if d > 0.0 or v_rel_n >= 0.0:
        return False

    # Project ball position onto racket plane
    proj = ball_state.position - racket.position
    proj_on_plane = proj - np.dot(proj, n) * n
    dist_from_center = np.linalg.norm(proj_on_plane)

    # Check if within circular racket area
    if dist_from_center > racket.radius:
        return False

    # Apply collision
    return handle_plane_collision(
        state=ball_state,
        plane_point=racket.position,
        normal=n,
        restitution=racket.restitution,
        friction=racket.friction,
        surface_velocity=racket.velocity,
    )


def check_net_collision(
    old_pos: np.ndarray,
    new_pos: np.ndarray,
    net: Net,
    table_height: float,
) -> Tuple[bool, EventType]:
    """Check if ball trajectory crosses or hits the net.

    Args:
        old_pos: Ball position before step.
        new_pos: Ball position after step.
        net: Net configuration.
        table_height: Height of table surface.

    Returns:
        Tuple of (crossed_net, event_type).
    """
    # Check if ball crossed x = 0 (net position)
    if old_pos[0] * new_pos[0] > 0:
        # Did not cross x = 0
        return False, EventType.NONE

    # Ball crossed x = 0, interpolate position at crossing
    if abs(new_pos[0] - old_pos[0]) < 1e-10:
        return False, EventType.NONE

    t_cross = -old_pos[0] / (new_pos[0] - old_pos[0])
    cross_pos = old_pos + t_cross * (new_pos - old_pos)

    # Check if within net y-extent
    half_net_len = net.length / 2.0
    if abs(cross_pos[1]) > half_net_len:
        # Passed outside net posts
        return True, EventType.NET_CROSS_SUCCESS

    # Check height at crossing
    net_top = table_height + net.height
    ball_bottom = cross_pos[2] - BALL_RADIUS

    if ball_bottom < table_height:
        # Ball went under table level (impossible in real play, but check)
        return True, EventType.NET_CROSS_FAIL
    elif ball_bottom < net_top:
        # Ball hit the net
        return True, EventType.NET_HIT
    else:
        # Ball cleared the net
        return True, EventType.NET_CROSS_SUCCESS


def check_table_bounds(pos: np.ndarray, table: Table) -> bool:
    """Check if ball position is within table bounds (x-y plane)."""
    half_len = table.length / 2.0
    half_wid = table.width / 2.0
    return (-half_len <= pos[0] <= half_len) and (-half_wid <= pos[1] <= half_wid)


def start_racket_movement(racket: RacketState, target_state: RacketState) -> None:
    """Start smooth movement of racket to target position."""
    distance = np.linalg.norm(target_state.position - racket.position)
    if distance < 1e-6:
        # Already at target position
        racket.movement.is_moving = False
        return

    racket.movement.is_moving = True
    racket.movement.target_position = target_state.position.copy()
    racket.movement.target_normal = target_state.normal.copy()
    racket.movement.target_velocity = target_state.velocity.copy()
    racket.movement.start_position = racket.position.copy()
    racket.movement.start_normal = racket.normal.copy()
    racket.movement.movement_time = distance / racket.movement.racket_speed
    racket.movement.elapsed_time = 0.0


def update_racket_movement(racket: RacketState, dt: float) -> None:
    """Update racket position during smooth movement."""
    if not racket.movement.is_moving:
        return

    racket.movement.elapsed_time += dt

    if racket.movement.elapsed_time >= racket.movement.movement_time:
        # Movement complete
        racket.position = racket.movement.target_position.copy()
        racket.normal = racket.movement.target_normal.copy()
        racket.velocity = racket.movement.target_velocity.copy()
        racket.movement.is_moving = False
    else:
        # Interpolate position and orientation
        t = racket.movement.elapsed_time / racket.movement.movement_time
        # Smooth interpolation using ease-in-out
        t_smooth = t * t * (3.0 - 2.0 * t)

        racket.position = (1.0 - t_smooth) * racket.movement.start_position + t_smooth * racket.movement.target_position
        racket.normal = (1.0 - t_smooth) * racket.movement.start_normal + t_smooth * racket.movement.target_normal
        # Renormalize normal vector
        racket.normal = racket.normal / np.linalg.norm(racket.normal)

        # Velocity is towards target
        if racket.movement.target_position is not None and racket.movement.start_position is not None:
            direction = racket.movement.target_position - racket.movement.start_position
            dist = np.linalg.norm(direction)
            if dist > 1e-6:
                racket.velocity = (direction / dist) * racket.movement.racket_speed


# =============================================================================
# Racket control and stroke execution
# =============================================================================


def compute_racket_for_stroke(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    stroke: StrokeParams,
    player: Player,
    table_height: float,
) -> RacketState:
    """Compute racket state to execute a stroke.

    Args:
        ball_pos: Current ball position.
        ball_vel: Current ball velocity.
        stroke: Stroke parameters.
        player: Which player is hitting.
        table_height: Table surface height.

    Returns:
        RacketState configured for the stroke.
    """
    # Racket position: at the strike point
    if player == Player.A:
        racket_x = stroke.target_x
        # Normal points towards positive x (towards opponent)
        base_normal = np.array([1.0, 0.0, 0.0])
    else:
        racket_x = stroke.target_x
        base_normal = np.array([-1.0, 0.0, 0.0])

    # Apply racket angle (rotation around y-axis)
    angle = stroke.racket_angle
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    if player == Player.A:
        normal = np.array([cos_a, 0.0, sin_a])
    else:
        normal = np.array([-cos_a, 0.0, sin_a])
    normal = normal / np.linalg.norm(normal)

    # Position racket at strike height, centered in y
    racket_pos = np.array([racket_x, 0.0, stroke.strike_height])

    # Velocity is swing direction times speed
    velocity = stroke.swing_direction * stroke.swing_speed

    # Get rubber properties
    restitution, friction = get_rubber_properties(stroke.rubber_type)

    return RacketState(
        position=racket_pos,
        normal=normal,
        velocity=velocity,
        radius=RACKET_RADIUS,
        rubber_type=stroke.rubber_type,
        restitution=restitution,
        friction=friction,
        player=player,
    )


def should_player_hit(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    player: Player,
    stroke: StrokeParams,
) -> bool:
    """Determine if player should attempt to hit the ball now.

    Args:
        ball_pos: Current ball position.
        ball_vel: Current ball velocity.
        player: Which player.
        stroke: Stroke parameters.

    Returns:
        True if player should hit now.
    """
    if player == Player.A:
        # Player A hits when ball is moving towards them (negative x velocity)
        # and ball x position is near their strike zone
        return (
            ball_vel[0] < 0 and
            ball_pos[0] < stroke.target_x + 0.15 and
            ball_pos[0] > stroke.target_x - 0.15
        )
    else:
        # Player B hits when ball is moving towards them (positive x velocity)
        return (
            ball_vel[0] > 0 and
            ball_pos[0] > stroke.target_x - 0.15 and
            ball_pos[0] < stroke.target_x + 0.15
        )


# =============================================================================
# Scenario definitions
# =============================================================================


def create_table() -> Table:
    """Create a standard table-tennis table."""
    return Table(
        height=TABLE_HEIGHT,
        length=TABLE_LENGTH,
        width=TABLE_WIDTH,
        restitution=TABLE_RESTITUTION,
        friction=TABLE_FRICTION,
    )


def create_net() -> Net:
    """Create a standard table-tennis net."""
    return Net(
        height=NET_HEIGHT,
        length=NET_LENGTH,
        thickness=NET_THICKNESS,
        x_position=0.0,
    )


def create_serve_scenario() -> Tuple[BallState, List[StrokeParams], List[StrokeParams]]:
    """Create initial conditions for a serve scenario.

    Player A serves from their side with topspin.
    Ball is given initial velocity as if just hit by the racket.

    Returns:
        Tuple of (initial_ball_state, strokes_a, strokes_b).
    """
    # Ball starts with initial velocity (simulating just after serve contact)
    # Position: near server's end of table, above table height
    initial_position = np.array([
        -TABLE_LENGTH / 2 + 0.2,   # near server's end
        0.0,                        # centered
        TABLE_HEIGHT + 0.30,        # above table
    ])
    # Typical serve velocity: forward with slight upward arc
    initial_velocity = np.array([6.0, 0.0, 2.0])  # m/s
    # Topspin around y-axis (positive y = ball rotating forward)
    initial_omega = np.array([0.0, 120.0, 0.0])  # rad/s

    ball_state = BallState(initial_position, initial_velocity, initial_omega)

    # Return strokes for both players (for subsequent rallies)
    strokes_a = [create_default_stroke(Player.A, "topspin") for _ in range(5)]
    strokes_b = [create_default_stroke(Player.B, "topspin") for _ in range(5)]

    return ball_state, strokes_a, strokes_b


def create_smash_scenario() -> Tuple[BallState, List[StrokeParams], List[StrokeParams]]:
    """Create initial conditions for a smash scenario.

    Ball starts with high speed downward trajectory (simulating just after smash contact).

    Returns:
        Tuple of (initial_ball_state, strokes_a, strokes_b).
    """
    # Ball starts with smash velocity (simulating just after smash contact)
    initial_position = np.array([
        -TABLE_LENGTH / 4,          # near center, player A's side
        0.0,
        TABLE_HEIGHT + 0.45,        # above table
    ])
    # Smash velocity: fast forward with downward component
    initial_velocity = np.array([12.0, 0.0, -2.0])  # m/s, fast and downward
    # Strong topspin
    initial_omega = np.array([0.0, 180.0, 0.0])  # rad/s

    ball_state = BallState(initial_position, initial_velocity, initial_omega)

    # Return strokes for both players (for subsequent rallies)
    strokes_a = [create_default_stroke(Player.A, "topspin") for _ in range(5)]
    strokes_b = [create_default_stroke(Player.B, "topspin") for _ in range(5)]

    return ball_state, strokes_a, strokes_b


def create_custom_scenario(
    position: np.ndarray,
    velocity: np.ndarray,
    omega: np.ndarray,
    strokes_a: Optional[List[StrokeParams]] = None,
    strokes_b: Optional[List[StrokeParams]] = None,
) -> Tuple[BallState, List[StrokeParams], List[StrokeParams]]:
    """Create a custom scenario with user-defined initial conditions.

    Args:
        position: Initial ball position (x, y, z).
        velocity: Initial ball velocity (vx, vy, vz).
        omega: Initial ball angular velocity (wx, wy, wz).
        strokes_a: Optional list of strokes for Player A.
        strokes_b: Optional list of strokes for Player B.

    Returns:
        Tuple of (initial_ball_state, strokes_a, strokes_b).
    """
    ball_state = BallState(
        position=np.asarray(position, dtype=float),
        velocity=np.asarray(velocity, dtype=float),
        omega=np.asarray(omega, dtype=float),
    )

    if strokes_a is None:
        strokes_a = [create_default_stroke(Player.A, "topspin") for _ in range(5)]
    if strokes_b is None:
        strokes_b = [create_default_stroke(Player.B, "topspin") for _ in range(5)]

    return ball_state, strokes_a, strokes_b


# =============================================================================
# Custom scenario configuration (modify these for custom play)
# =============================================================================

# Default custom initial conditions (typical serve position with topspin)
CUSTOM_INITIAL_POSITION: List[float] = [
    -TABLE_LENGTH / 2 - 0.2,  # x: behind table
    0.0,                       # y: centered
    TABLE_HEIGHT + 0.25,       # z: above table
]
CUSTOM_INITIAL_VELOCITY: List[float] = [5.0, 0.0, 2.0]  # forward and upward
CUSTOM_INITIAL_OMEGA: List[float] = [0.0, 100.0, 0.0]   # topspin around y-axis

# Custom stroke sequence for Player A (each rally)
# Format: [(stroke_type, rubber_type), ...]
CUSTOM_STROKES_A: List[Tuple[str, str]] = [
    ("topspin", "inverted"),
    ("topspin", "inverted"),
    ("backspin", "inverted"),
    ("topspin", "inverted"),
    ("flat", "inverted"),
]

# Custom stroke sequence for Player B
CUSTOM_STROKES_B: List[Tuple[str, str]] = [
    ("topspin", "inverted"),
    ("backspin", "pimpled"),
    ("topspin", "inverted"),
    ("sidespin", "inverted"),
    ("topspin", "inverted"),
]


def parse_rubber_type(name: str) -> RubberType:
    """Parse rubber type from string."""
    name_lower = name.lower()
    if name_lower == "inverted":
        return RubberType.INVERTED
    elif name_lower == "pimpled":
        return RubberType.PIMPLED
    elif name_lower == "antispin":
        return RubberType.ANTISPIN
    else:
        raise ValueError(f"Unknown rubber type: {name}")


def create_custom_strokes(
    stroke_configs: List[Tuple[str, str]],
    player: Player,
) -> List[StrokeParams]:
    """Create stroke list from configuration tuples."""
    strokes = []
    for stroke_type, rubber_name in stroke_configs:
        rubber = parse_rubber_type(rubber_name)
        strokes.append(create_default_stroke(player, stroke_type, rubber))
    return strokes


# =============================================================================
# Simulation loop
# =============================================================================


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    ball_history: Dict[str, np.ndarray]
    racket_a_history: Dict[str, np.ndarray]
    racket_b_history: Dict[str, np.ndarray]
    events: List[Tuple[float, EventType, str]]  # (time, event_type, description)
    net_crossings: int
    table_bounces: int
    rally_count: int
    final_event: EventType


def simulate(
    initial_ball: BallState,
    strokes_a: List[StrokeParams],
    strokes_b: List[StrokeParams],
    table: Table,
    net: Net,
    dt: float = TIME_STEP,
    max_time: float = MAX_TIME,
    record_interval: int = RECORD_INTERVAL,
) -> SimulationResult:
    """Run the simulation with dual rackets and rally tracking.

    Args:
        initial_ball: Initial ball state.
        strokes_a: Stroke sequence for Player A.
        strokes_b: Stroke sequence for Player B.
        table: Table configuration.
        net: Net configuration.
        dt: Time step.
        max_time: Maximum simulation time.
        record_interval: Record state every N steps.

    Returns:
        SimulationResult with all trajectory data and events.
    """
    t = 0.0
    step_count = 0
    ball = initial_ball.copy()

    # Initialize racket states (will be updated when players hit)
    racket_a = RacketState(
        position=np.array([PLAYER_A_X, 0.0, PLAYER_STRIKE_HEIGHT]),
        normal=np.array([1.0, 0.0, 0.0]),
        velocity=np.zeros(3),
        radius=RACKET_RADIUS,
        rubber_type=RubberType.INVERTED,
        restitution=RUBBER_INVERTED_RESTITUTION,
        friction=RUBBER_INVERTED_FRICTION,
        player=Player.A,
    )
    racket_b = RacketState(
        position=np.array([PLAYER_B_X, 0.0, PLAYER_STRIKE_HEIGHT]),
        normal=np.array([-1.0, 0.0, 0.0]),
        velocity=np.zeros(3),
        radius=RACKET_RADIUS,
        rubber_type=RubberType.INVERTED,
        restitution=RUBBER_INVERTED_RESTITUTION,
        friction=RUBBER_INVERTED_FRICTION,
        player=Player.B,
    )

    # History storage
    ball_history: Dict[str, List[float]] = {
        "t": [], "x": [], "y": [], "z": [],
        "vx": [], "vy": [], "vz": [],
        "wx": [], "wy": [], "wz": [],
        "event": [],
    }
    racket_a_history: Dict[str, List[float]] = {
        "t": [], "x": [], "y": [], "z": [],
        "vx": [], "vy": [], "vz": [],
        "nx": [], "ny": [], "nz": [],
    }
    racket_b_history: Dict[str, List[float]] = {
        "t": [], "x": [], "y": [], "z": [],
        "vx": [], "vy": [], "vz": [],
        "nx": [], "ny": [], "nz": [],
    }

    events: List[Tuple[float, EventType, str]] = []
    net_crossings = 0
    table_bounces = 0
    rally_count = 0
    stroke_idx_a = 0
    stroke_idx_b = 0
    last_hitter: Optional[Player] = None
    final_event = EventType.NONE

    # Track which side of net the ball is on
    ball_side_positive = ball.position[0] > 0

    # Table collision parameters
    table_normal = np.array([0.0, 0.0, 1.0])
    table_point = np.array([0.0, 0.0, table.height])

    def record_state():
        """Record current state to history."""
        ball_history["t"].append(t)
        ball_history["x"].append(ball.position[0])
        ball_history["y"].append(ball.position[1])
        ball_history["z"].append(ball.position[2])
        ball_history["vx"].append(ball.velocity[0])
        ball_history["vy"].append(ball.velocity[1])
        ball_history["vz"].append(ball.velocity[2])
        ball_history["wx"].append(ball.omega[0])
        ball_history["wy"].append(ball.omega[1])
        ball_history["wz"].append(ball.omega[2])
        ball_history["event"].append(EventType.NONE.value)

        racket_a_history["t"].append(t)
        racket_a_history["x"].append(racket_a.position[0])
        racket_a_history["y"].append(racket_a.position[1])
        racket_a_history["z"].append(racket_a.position[2])
        racket_a_history["vx"].append(racket_a.velocity[0])
        racket_a_history["vy"].append(racket_a.velocity[1])
        racket_a_history["vz"].append(racket_a.velocity[2])
        racket_a_history["nx"].append(racket_a.normal[0])
        racket_a_history["ny"].append(racket_a.normal[1])
        racket_a_history["nz"].append(racket_a.normal[2])

        racket_b_history["t"].append(t)
        racket_b_history["x"].append(racket_b.position[0])
        racket_b_history["y"].append(racket_b.position[1])
        racket_b_history["z"].append(racket_b.position[2])
        racket_b_history["vx"].append(racket_b.velocity[0])
        racket_b_history["vy"].append(racket_b.velocity[1])
        racket_b_history["vz"].append(racket_b.velocity[2])
        racket_b_history["nx"].append(racket_b.normal[0])
        racket_b_history["ny"].append(racket_b.normal[1])
        racket_b_history["nz"].append(racket_b.normal[2])

    # Main simulation loop
    while t <= max_time:
        # Update racket movements
        update_racket_movement(racket_a, dt)
        update_racket_movement(racket_b, dt)

        # Record state at intervals
        if step_count % record_interval == 0:
            record_state()

        # Store old position for net crossing check
        old_pos = ball.position.copy()

        # Advance ball state
        new_ball = rk4_step(ball, dt)

        # Check for events
        current_event = EventType.NONE

        # 1. Check net crossing/collision
        crossed, net_event = check_net_collision(old_pos, new_ball.position, net, table.height)
        if crossed:
            if net_event == EventType.NET_HIT:
                events.append((t, net_event, "Ball hit the net"))
                final_event = net_event
                ball = new_ball
                break
            elif net_event == EventType.NET_CROSS_FAIL:
                events.append((t, net_event, "Ball failed to clear net"))
                final_event = net_event
                ball = new_ball
                break
            elif net_event == EventType.NET_CROSS_SUCCESS:
                net_crossings += 1
                ball_side_positive = new_ball.position[0] > 0
                events.append((t, net_event, f"Ball crossed net (crossing #{net_crossings})"))
                current_event = net_event

        # 2. Check table collision
        if check_table_bounds(new_ball.position, table):
            if handle_plane_collision(
                new_ball, table_point, table_normal,
                table.restitution, table.friction, np.zeros(3)
            ):
                table_bounces += 1
                events.append((t, EventType.TABLE_BOUNCE, f"Ball bounced on table (bounce #{table_bounces})"))
                current_event = EventType.TABLE_BOUNCE

        # 3. Check racket collisions
        # Player A hits when ball is on their side and moving towards them
        if stroke_idx_a < len(strokes_a) and not racket_a.movement.is_moving:
            stroke_a = strokes_a[stroke_idx_a]
            if should_player_hit(new_ball.position, new_ball.velocity, Player.A, stroke_a):
                # Start racket A movement to strike position
                target_racket_a = compute_racket_for_stroke(
                    new_ball.position, new_ball.velocity,
                    stroke_a, Player.A, table.height
                )
                start_racket_movement(racket_a, target_racket_a)

        # Player B hits when ball is on their side and moving towards them
        if stroke_idx_b < len(strokes_b) and not racket_b.movement.is_moving:
            stroke_b = strokes_b[stroke_idx_b]
            if should_player_hit(new_ball.position, new_ball.velocity, Player.B, stroke_b):
                # Start racket B movement to strike position
                target_racket_b = compute_racket_for_stroke(
                    new_ball.position, new_ball.velocity,
                    stroke_b, Player.B, table.height
                )
                start_racket_movement(racket_b, target_racket_b)

        # Check for collisions (rackets can be moving)
        if stroke_idx_a < len(strokes_a):
                if handle_circular_racket_collision(new_ball, racket_a):
                    rally_count += 1
                    stroke_idx_a += 1
                    last_hitter = Player.A
                # Stop racket movement after collision
                racket_a.movement.is_moving = False
                events.append((t, EventType.RACKET_A_HIT, f"Player A hit (rally #{rally_count})"))
                current_event = EventType.RACKET_A_HIT

        if stroke_idx_b < len(strokes_b):
                if handle_circular_racket_collision(new_ball, racket_b):
                    rally_count += 1
                    stroke_idx_b += 1
                    last_hitter = Player.B
                # Stop racket movement after collision
                racket_b.movement.is_moving = False
                events.append((t, EventType.RACKET_B_HIT, f"Player B hit (rally #{rally_count})"))
                current_event = EventType.RACKET_B_HIT

        # Update event in history
        if current_event != EventType.NONE and len(ball_history["event"]) > 0:
            ball_history["event"][-1] = current_event.value

        # 4. Check termination conditions
        # Ball fell below table
        if new_ball.position[2] < table.height - 0.5:
            events.append((t, EventType.OUT_OF_BOUNDS, "Ball fell below table"))
            final_event = EventType.OUT_OF_BOUNDS
            ball = new_ball
            break

        # Ball went out of bounds (x or y)
        if abs(new_ball.position[0]) > table.length + 1.0 or abs(new_ball.position[1]) > table.width + 0.5:
            events.append((t, EventType.OUT_OF_BOUNDS, "Ball out of bounds"))
            final_event = EventType.OUT_OF_BOUNDS
            ball = new_ball
            break

        ball = new_ball
        t += dt
        step_count += 1

    # Final record
    record_state()

    # Convert to numpy arrays
    ball_history_np = {k: np.asarray(v) for k, v in ball_history.items()}
    racket_a_history_np = {k: np.asarray(v) for k, v in racket_a_history.items()}
    racket_b_history_np = {k: np.asarray(v) for k, v in racket_b_history.items()}

    return SimulationResult(
        ball_history=ball_history_np,
        racket_a_history=racket_a_history_np,
        racket_b_history=racket_b_history_np,
        events=events,
        net_crossings=net_crossings,
        table_bounces=table_bounces,
        rally_count=rally_count,
        final_event=final_event,
    )


# =============================================================================
# Output utilities
# =============================================================================


def save_ball_history_to_csv(history: Dict[str, np.ndarray], filename: str) -> None:
    """Save ball trajectory to CSV file."""
    df = pd.DataFrame(history)
    df.to_csv(filename, index=False, float_format="%.6f")


def save_racket_history_to_csv(history: Dict[str, np.ndarray], filename: str) -> None:
    """Save racket trajectory to CSV file."""
    df = pd.DataFrame(history)
    df.to_csv(filename, index=False, float_format="%.6f")


def print_simulation_summary(result: SimulationResult) -> None:
    """Print a summary of the simulation results."""
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Total rallies: {result.rally_count}")
    print(f"Net crossings: {result.net_crossings}")
    print(f"Table bounces: {result.table_bounces}")
    print(f"Final event: {result.final_event.name}")
    print("\nEvent log:")
    for t, event, desc in result.events:
        print(f"  t={t:.4f}s: [{event.name}] {desc}")
    print("=" * 60 + "\n")


# =============================================================================
# Visualization utilities
# =============================================================================


def draw_table_3d(ax: Axes3D, table: Table, net: Net) -> None:
    """Draw the table and net on a 3D axis."""
    half_len = table.length / 2.0
    half_wid = table.width / 2.0
    z = table.height

    # Table surface
    table_verts = [
        [(-half_len, -half_wid, z), (half_len, -half_wid, z),
         (half_len, half_wid, z), (-half_len, half_wid, z)]
    ]
    table_poly = Poly3DCollection(table_verts, alpha=0.4, facecolor='darkgreen', edgecolor='white', linewidth=1)
    ax.add_collection3d(table_poly)

    # Center line
    ax.plot([0, 0], [-half_wid, half_wid], [z, z], 'w-', linewidth=1)

    # Net
    net_half_len = net.length / 2.0
    net_top = z + net.height
    net_verts = [
        [(0, -net_half_len, z), (0, net_half_len, z),
         (0, net_half_len, net_top), (0, -net_half_len, net_top)]
    ]
    net_poly = Poly3DCollection(net_verts, alpha=0.3, facecolor='gray', edgecolor='black', linewidth=0.5)
    ax.add_collection3d(net_poly)

    # Net posts
    ax.plot([0, 0], [-net_half_len, -net_half_len], [z, net_top], 'k-', linewidth=2)
    ax.plot([0, 0], [net_half_len, net_half_len], [z, net_top], 'k-', linewidth=2)


def draw_racket_3d(ax: Axes3D, position: np.ndarray, normal: np.ndarray, radius: float, color: str) -> None:
    """Draw a circular racket on a 3D axis."""
    # Create circle points in local coordinates
    theta = np.linspace(0, 2 * np.pi, 32)
    t1, t2, n = _orthonormal_basis_from_normal(normal)

    circle_points = []
    for th in theta:
        p = position + radius * (np.cos(th) * t1 + np.sin(th) * t2)
        circle_points.append(p)

    # Draw as polygon
    verts = [circle_points]
    poly = Poly3DCollection(verts, alpha=0.6, facecolor=color, edgecolor='black', linewidth=1)
    ax.add_collection3d(poly)


def plot_trajectory_3d(result: SimulationResult, table: Table, net: Net) -> plt.Figure:
    """Create a 3D plot of the ball trajectory with table, net, and rackets."""
    x = result.ball_history["x"]
    y = result.ball_history["y"]
    z = result.ball_history["z"]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw table and net
    draw_table_3d(ax, table, net)

    # Draw initial and final racket positions
    ra = result.racket_a_history
    rb = result.racket_b_history
    if len(ra["x"]) > 0:
        # Draw racket A at a few key positions
        for i in [0, len(ra["x"]) // 2, -1]:
            pos_a = np.array([ra["x"][i], ra["y"][i], ra["z"][i]])
            norm_a = np.array([ra["nx"][i], ra["ny"][i], ra["nz"][i]])
            draw_racket_3d(ax, pos_a, norm_a, RACKET_RADIUS, 'red')

    if len(rb["x"]) > 0:
        for i in [0, len(rb["x"]) // 2, -1]:
            pos_b = np.array([rb["x"][i], rb["y"][i], rb["z"][i]])
            norm_b = np.array([rb["nx"][i], rb["ny"][i], rb["nz"][i]])
            draw_racket_3d(ax, pos_b, norm_b, RACKET_RADIUS, 'blue')

    # Plot ball trajectory
    ax.plot(x, y, z, 'orange', linewidth=2, label='Ball trajectory')
    ax.scatter(x[0], y[0], z[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, marker='x', label='End')

    # Mark bounce points
    events = result.ball_history["event"]
    bounce_mask = events == EventType.TABLE_BOUNCE.value
    if np.any(bounce_mask):
        ax.scatter(x[bounce_mask], y[bounce_mask], z[bounce_mask],
                   color='yellow', s=50, marker='^', label='Bounce')

    # Axis settings
    half_len = table.length / 2.0
    half_wid = table.width / 2.0
    margin = 0.5

    ax.set_xlim(-half_len - margin - 1, half_len + margin + 1)
    ax.set_ylim(-half_wid - margin, half_wid + margin)
    ax.set_zlim(table.height - 0.1, max(np.max(z) + 0.3, table.height + 0.8))

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("3D Ping-Pong Ball Trajectory")
    ax.legend(loc='upper left')

    return fig


def animate_trajectory_3d(
    result: SimulationResult,
    table: Table,
    net: Net,
    filename: str,
    fps: int = ANIM_FPS,
    skip: int = ANIM_SKIP,
) -> None:
    """Create and save a 3D animation of the trajectory."""
    x = result.ball_history["x"][::skip]
    y = result.ball_history["y"][::skip]
    z = result.ball_history["z"][::skip]
    t_arr = result.ball_history["t"][::skip]

    ra = {k: v[::skip] for k, v in result.racket_a_history.items()}
    rb = {k: v[::skip] for k, v in result.racket_b_history.items()}

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    half_len = table.length / 2.0
    half_wid = table.width / 2.0

    # Draw static elements
    draw_table_3d(ax, table, net)

    # Initialize animated elements
    (traj_line,) = ax.plot([], [], [], 'orange', linewidth=2)
    (ball_point,) = ax.plot([], [], [], 'o', color='orange', markersize=10)

    ax.set_xlim(-half_len - 1.5, half_len + 1.5)
    ax.set_ylim(-half_wid - 0.5, half_wid + 0.5)
    ax.set_zlim(table.height - 0.1, max(np.max(z) + 0.3, table.height + 0.8))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    title = ax.set_title("")

    def init():
        traj_line.set_data([], [])
        traj_line.set_3d_properties([])
        ball_point.set_data([], [])
        ball_point.set_3d_properties([])
        return traj_line, ball_point

    def update(frame: int):
        traj_line.set_data(x[:frame + 1], y[:frame + 1])
        traj_line.set_3d_properties(z[:frame + 1])
        ball_point.set_data([x[frame]], [y[frame]])
        ball_point.set_3d_properties([z[frame]])
        title.set_text(f"t = {t_arr[frame]:.3f} s")
        return traj_line, ball_point

    interval_ms = 1000.0 / fps
    ani = animation.FuncAnimation(
        fig, update, frames=len(x),
        init_func=init, interval=interval_ms, blit=False
    )

    try:
        ani.save(filename, fps=fps, dpi=120)
        print(f"Animation saved to {filename}")
    except Exception as exc:
        print(f"Warning: failed to save animation to {filename}: {exc}")
        print("Make sure FFmpeg is installed for video export.")
    finally:
        plt.close(fig)


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "3D simulation of ping-pong ball with aerodynamic forces, "
            "table bounce, net detection, and dual racket interaction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pingpong_sim.py --scenario serve
  python pingpong_sim.py --scenario smash --duration 3.0
  python pingpong_sim.py --scenario custom --pos -1.2 0 0.9 --vel 8 0 3 --omega 0 150 0
        """
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=DEFAULT_SCENARIO,
        choices=["serve", "smash", "custom"],
        help="Scenario type (default: serve)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for data and video files (default: ./output)",
    )
    parser.add_argument(
        "--ball-csv",
        type=str,
        default=DEFAULT_BALL_CSV,
        help="Output CSV filename for ball trajectory",
    )
    parser.add_argument(
        "--racket-csv",
        type=str,
        default=DEFAULT_RACKET_CSV,
        help="Output CSV filename for racket trajectories",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=MAX_TIME,
        help="Total simulation time in seconds",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=TIME_STEP,
        help="Time step size in seconds",
    )
    parser.add_argument(
        "--no-animate",
        action="store_true",
        help="Do not save animation (only static plot)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not show interactive plot",
    )

    # Custom scenario parameters
    parser.add_argument(
        "--pos",
        type=float,
        nargs=3,
        default=CUSTOM_INITIAL_POSITION,
        metavar=("X", "Y", "Z"),
        help="Initial ball position (x, y, z) in meters",
    )
    parser.add_argument(
        "--vel",
        type=float,
        nargs=3,
        default=CUSTOM_INITIAL_VELOCITY,
        metavar=("VX", "VY", "VZ"),
        help="Initial ball velocity (vx, vy, vz) in m/s",
    )
    parser.add_argument(
        "--omega",
        type=float,
        nargs=3,
        default=CUSTOM_INITIAL_OMEGA,
        metavar=("WX", "WY", "WZ"),
        help="Initial ball angular velocity (wx, wy, wz) in rad/s",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    video_dir = output_dir / "video"

    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Data files will be saved to: {data_dir}")
    print(f"Video files will be saved to: {video_dir}")

    # Create table and net
    table = create_table()
    net = create_net()

    # Create scenario
    if args.scenario == "serve":
        initial_ball, strokes_a, strokes_b = create_serve_scenario()
    elif args.scenario == "smash":
        initial_ball, strokes_a, strokes_b = create_smash_scenario()
    else:  # custom
        strokes_a = create_custom_strokes(CUSTOM_STROKES_A, Player.A)
        strokes_b = create_custom_strokes(CUSTOM_STROKES_B, Player.B)
        initial_ball, strokes_a, strokes_b = create_custom_scenario(
            position=args.pos,
            velocity=args.vel,
            omega=args.omega,
            strokes_a=strokes_a,
            strokes_b=strokes_b,
        )

    print(f"Running simulation: {args.scenario} scenario")
    print(f"Initial position: {initial_ball.position}")
    print(f"Initial velocity: {initial_ball.velocity}")
    print(f"Initial spin: {initial_ball.omega}")

    # Run simulation
    result = simulate(
        initial_ball=initial_ball,
        strokes_a=strokes_a,
        strokes_b=strokes_b,
        table=table,
        net=net,
        dt=args.dt,
        max_time=args.duration,
    )

    # Print summary
    print_simulation_summary(result)

    # Save data
    ball_csv_path = data_dir / args.ball_csv
    save_ball_history_to_csv(result.ball_history, str(ball_csv_path))
    print(f"Saved ball trajectory to {ball_csv_path}")

    # Save racket trajectories
    racket_a_file = data_dir / args.racket_csv.replace(".csv", "_A.csv")
    racket_b_file = data_dir / args.racket_csv.replace(".csv", "_B.csv")
    save_racket_history_to_csv(result.racket_a_history, str(racket_a_file))
    save_racket_history_to_csv(result.racket_b_history, str(racket_b_file))
    print(f"Saved racket A trajectory to {racket_a_file}")
    print(f"Saved racket B trajectory to {racket_b_file}")

    # Visualization
    if not args.no_plot:
        fig = plot_trajectory_3d(result, table, net)
        plt.show()

    if not args.no_animate:
        anim_file = video_dir / DEFAULT_ANIM_FILE
        animate_trajectory_3d(result, table, net, str(anim_file))


if __name__ == "__main__":
    main()
