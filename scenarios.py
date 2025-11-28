"""Scenario definitions for ball sports simulation.

This module provides predefined scenarios and configurations for
different types of ball sports simulations, including table tennis.
"""

import numpy as np
from typing import List, Optional, Tuple

from constants import (
    TABLE_HEIGHT, TABLE_LENGTH, TABLE_WIDTH, TABLE_RESTITUTION, TABLE_FRICTION,
    NET_HEIGHT, NET_LENGTH, NET_THICKNESS
)
from ball_types import BallState, Table, Net, StrokeParams, Player, RubberType
from racket_control import create_default_stroke


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
