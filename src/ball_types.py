"""Data structures and type definitions for ball sports simulation.

This module defines all the core data structures used in ball trajectory
simulations, including ball state, table geometry, racket properties,
and simulation configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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
    DOUBLE_BOUNCE = 8


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
    start_velocity: Optional[np.ndarray] = None
    movement_time: float = 0.0  # total time for movement
    elapsed_time: float = 0.0   # time elapsed in movement
    racket_speed: float = 1.0   # m/s, speed at which racket moves to position (further reduced for smooth movement)
    min_duration: float = 0.02  # s, minimum duration for easing profile
    reaction_delay: float = 0.08  # s, shorter delay for faster reaction
    delay_elapsed: float = 0.0   # time elapsed in delay phase


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
                start_velocity=self.movement.start_velocity.copy() if self.movement.start_velocity is not None else None,
                movement_time=self.movement.movement_time,
                elapsed_time=self.movement.elapsed_time,
                racket_speed=self.movement.racket_speed,
                min_duration=self.movement.min_duration,
                reaction_delay=self.movement.reaction_delay,
                delay_elapsed=self.movement.delay_elapsed,
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
    mode: str = "custom"      # descriptive stroke mode (drop_short, flick, counter_loop, custom)
    contact_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))  # preferred hit point on ball (local coords)


@dataclass
class RallyConfig:
    """Configuration for a rally (sequence of strokes)."""
    strokes_a: List[StrokeParams]  # strokes for player A
    strokes_b: List[StrokeParams]  # strokes for player B
    max_rallies: int = 10          # maximum number of hits


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
    winner: Optional[Player] = None
    winner_reason: str = ""
