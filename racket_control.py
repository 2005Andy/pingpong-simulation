"""Racket control and stroke execution for ball sports simulation.

This module handles racket positioning, stroke parameters, and player
decision-making for when to hit the ball.
"""

import numpy as np
from typing import Tuple, List

from constants import (
    RACKET_RADIUS, RUBBER_INVERTED_RESTITUTION, RUBBER_INVERTED_FRICTION,
    RUBBER_PIMPLED_RESTITUTION, RUBBER_PIMPLED_FRICTION,
    RUBBER_ANTISPIN_RESTITUTION, RUBBER_ANTISPIN_FRICTION
)
from ball_types import RubberType, Player, StrokeParams, RacketState


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
    from constants import TABLE_LENGTH  # Import here to avoid circular imports

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
        strike_height=0.25,  # 25 cm above table (will be added to table height)
        racket_angle=angle,
        swing_speed=speed,
        swing_direction=swing_dir,
        rubber_type=rubber_type,
        spin_intent=stroke_type,
    )


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
    racket_pos = np.array([racket_x, 0.0, table_height + stroke.strike_height])

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
