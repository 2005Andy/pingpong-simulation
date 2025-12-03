"""Scenario definitions for ball sports simulation.

This module provides predefined scenarios and configurations for
different types of ball sports simulations, including table tennis.
"""

import numpy as np
from typing import List, Optional, Tuple

from constants import (
    TABLE_HEIGHT, TABLE_LENGTH, TABLE_WIDTH, TABLE_RESTITUTION, TABLE_FRICTION,
    NET_HEIGHT, NET_LENGTH, NET_THICKNESS,
    CUSTOM_INITIAL_POSITION, CUSTOM_INITIAL_VELOCITY, CUSTOM_INITIAL_OMEGA, DEFAULT_SERVE_MODE
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


SERVE_MODE_PRESETS = {
    "fh_under": {
        "description": "Forehand underspin short serve",
        "position": np.array([
            -TABLE_LENGTH / 2 - 0.15,
            0.45,
            TABLE_HEIGHT + 0.2,
        ]),
        "velocity": np.array([4.5, -0.2, -2.0]),
        "omega": np.array([0.0, 220.0, 0.0]),
    },
    "fast_long": {
        "description": "Fast long drive serve",
        "position": np.array([
            -TABLE_LENGTH / 2 - 0.12,
            -0.10,
            TABLE_HEIGHT + 0.32,
        ]),
        "velocity": np.array([8.8, -0.1, 0.6]),
        "omega": np.array([0.0, 140.0, 25.0]),
    },
}

SERVE_MODE_CHOICES: Tuple[str, ...] = tuple(list(SERVE_MODE_PRESETS.keys()) + ["custom"])


def _mirror_for_player(vec: np.ndarray, player: Player) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    if player == Player.B:
        arr = np.array([-arr[0], -arr[1], arr[2]])
    return arr


def create_initial_ball_state(
    server: Player,
    serve_mode: str = DEFAULT_SERVE_MODE,
    position: Optional[np.ndarray] = None,
    velocity: Optional[np.ndarray] = None,
    omega: Optional[np.ndarray] = None,
) -> BallState:
    """Resolve the initial ball state for a serve mode or manual configuration."""
    if serve_mode != "custom":
        if serve_mode not in SERVE_MODE_PRESETS:
            raise ValueError(f"Unknown serve mode '{serve_mode}'")
        preset = SERVE_MODE_PRESETS[serve_mode]
        pos = _mirror_for_player(preset["position"], server)
        vel = _mirror_for_player(preset["velocity"], server)
        ang = _mirror_for_player(preset["omega"], server)
        return BallState(pos, vel, ang)

    pos_arr = np.asarray(position if position is not None else CUSTOM_INITIAL_POSITION, dtype=float)
    vel_arr = np.asarray(velocity if velocity is not None else CUSTOM_INITIAL_VELOCITY, dtype=float)
    omega_arr = np.asarray(omega if omega is not None else CUSTOM_INITIAL_OMEGA, dtype=float)
    return BallState(pos_arr, vel_arr, omega_arr)


def create_custom_scenario(
    position: Optional[np.ndarray] = None,
    velocity: Optional[np.ndarray] = None,
    omega: Optional[np.ndarray] = None,
    strokes_a: Optional[List[StrokeParams]] = None,
    strokes_b: Optional[List[StrokeParams]] = None,
    serve_mode: str = "custom",
    server: Player = Player.A,
) -> Tuple[BallState, List[StrokeParams], List[StrokeParams]]:
    """Create a custom scenario with user-defined initial conditions.

    Args:
        position: Initial ball position (x, y, z) for custom mode.
        velocity: Initial ball velocity (vx, vy, vz) for custom mode.
        omega: Initial ball angular velocity (wx, wy, wz) for custom mode.
        strokes_a: Optional list of strokes for Player A.
        strokes_b: Optional list of strokes for Player B.
        serve_mode: Serve preset name or "custom".
        server: Which player performs the serve.

    Returns:
        Tuple of (initial_ball_state, strokes_a, strokes_b).
    """
    ball_state = create_initial_ball_state(
        server=server,
        serve_mode=serve_mode,
        position=position,
        velocity=velocity,
        omega=omega,
    )

    if strokes_a is None:
        strokes_a = [create_default_stroke(Player.A, "topspin") for _ in range(5)]
    if strokes_b is None:
        strokes_b = [create_default_stroke(Player.B, "topspin") for _ in range(5)]

    return ball_state, strokes_a, strokes_b
