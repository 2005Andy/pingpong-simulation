"""Physics calculations for ball trajectory simulation.

This module contains all physics-related calculations including:
- Aerodynamic forces (drag and Magnus effect)
- Numerical integration (RK4)
- Collision detection and response
- Racket movement physics
"""

import numpy as np
from typing import Tuple, Optional

from constants import (
    AIR_DENSITY, GRAVITY, BALL_RADIUS, BALL_MASS, DRAG_COEFF, MAGNUS_COEFF,
    BALL_INERTIA_FACTOR, WIND_VELOCITY, RACKET_MAX_SPEED
)
from ball_types import BallState, Table, Net, RacketState, EventType


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


def start_racket_movement(
    racket: RacketState,
    target_state: RacketState,
    desired_arrival_time: Optional[float] = None,
) -> None:
    """Start instant movement of racket to target position for hitting."""
    # For hitting, move instantly to target position to ensure contact
    racket.position = target_state.position.copy()
    racket.normal = target_state.normal.copy()
    racket.velocity = target_state.velocity.copy()
    racket.movement.is_moving = False  # Mark as not moving since we're already there


def update_racket_movement(racket: RacketState, dt: float) -> None:
    """Update racket position during smooth movement."""
    if not racket.movement.is_moving:
        return

    if racket.movement.target_position is None or racket.movement.start_position is None:
        racket.movement.is_moving = False
        return

    racket.movement.elapsed_time += dt
    movement_time = max(racket.movement.movement_time, 1.0e-4)
    tau = np.clip(racket.movement.elapsed_time / movement_time, 0.0, 1.0)
    # Quintic polynomial (smoothstep) for zero velocity endpoints
    s = tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))
    ds_dt = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / movement_time

    start_pos = racket.movement.start_position
    end_pos = racket.movement.target_position
    path = end_pos - start_pos
    racket.position = start_pos + s * path

    start_normal = racket.movement.start_normal
    target_normal = racket.movement.target_normal
    if start_normal is not None and target_normal is not None:
        interp_normal = (1.0 - s) * start_normal + s * target_normal
        norm = np.linalg.norm(interp_normal)
        if norm > 1.0e-8:
            racket.normal = interp_normal / norm

    if racket.movement.target_velocity is not None and racket.movement.start_velocity is not None:
        racket.velocity = (1.0 - s) * racket.movement.start_velocity + s * racket.movement.target_velocity
    else:
        racket.velocity = path * ds_dt

    if tau >= 0.999:
        racket.position = end_pos.copy()
        if target_normal is not None:
            racket.normal = target_normal.copy()
        if racket.movement.target_velocity is not None:
            racket.velocity = racket.movement.target_velocity.copy()
        racket.movement.is_moving = False
