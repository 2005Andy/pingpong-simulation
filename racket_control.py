"""Racket control and stroke execution for ball sports simulation.

This module handles racket positioning, stroke parameters, and player
decision-making for when to hit the ball.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional

from constants import (
    RACKET_RADIUS, RUBBER_INVERTED_RESTITUTION, RUBBER_INVERTED_FRICTION,
    RUBBER_PIMPLED_RESTITUTION, RUBBER_PIMPLED_FRICTION,
    RUBBER_ANTISPIN_RESTITUTION, RUBBER_ANTISPIN_FRICTION,
    TABLE_LENGTH, TABLE_HEIGHT, BALL_RADIUS,
    RACKET_STRIKE_HEIGHT_WINDOW, RACKET_STRIKE_X_WINDOW
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


STROKE_MODALITIES: Dict[str, Dict[str, Any]] = {
    "drop_short": {
        "angle_deg_range": (110.0, 130.0),
        "strike_height": 0.04,
        "swing_speed_range": (1.2, 1.8),
        "direction": (0.25, 0.0, -0.04),
        "spin": "backspin",
        "contact_offset": (0.0, 0.0, -0.015),
        "target_x_offset": 0.12,
    },
    "flick": {
        "angle_deg_range": (88.0, 94.0),  # Even more conservative angle range
        "strike_height": 0.15,  # Lower strike height
        "swing_speed_range": (0.04, 0.12),  # Even slower speed range
        "direction": (0.06, 0.0, 0.03),  # Very conservative direction
        "spin": "topspin",
        "contact_offset": (0.0, 0.0, -0.005),
        "target_x_offset": 0.16,  # Closer target for maximum safety
    },
    "counter_loop": {
        "angle_deg_range": (35.0, 55.0),  # Slightly higher angle for better control
        "strike_height": 0.25,  # Higher strike height
        "swing_speed_range": (7.0, 9.0),  # Reduced speed for better control
        "direction": (1.0, 0.0, -0.10),  # Less downward angle
        "spin": "topspin",
        "contact_offset": (0.0, 0.0, 0.015),
        "target_x_offset": 0.35,  # Further target position
    },
    # Legacy stroke names preserved for compatibility
    "topspin": {
        "angle_deg": 96.0,  # Slightly lower angle for good control
        "strike_height": 0.12,  # Lower strike height
        "swing_speed": 1.6,  # Moderate speed for better momentum without losing control
        "direction": (0.20, 0.0, 0.035),  # Slightly more momentum
        "spin": "topspin",
        "target_x_offset": 0.18,  # Balanced target distance
    },
    "backspin": {
        "angle_deg": 70.0,
        "strike_height": 0.20,
        "swing_speed": 5.0,
        "direction": (1.0, 0.0, -0.25),
        "spin": "backspin",
        "target_x_offset": 0.28,
    },
    "sidespin": {
        "angle_deg": 100.0,
        "strike_height": 0.24,
        "swing_speed": 7.0,
        "direction": (1.0, 0.5, 0.25),
        "spin": "sidespin",
        "target_x_offset": 0.30,
    },
    "flat": {
        "angle_deg": 90.0,
        "strike_height": 0.23,
        "swing_speed": 0.1,
        "direction": (0.1, 0.0, 0.05),
        "spin": "flat",
        "target_x_offset": 0.28,
    },
    "custom": {},  # fully user-specified
}


def _resolve_range(value: Optional[Tuple[float, float]], fallback: float) -> float:
    if value is None:
        return fallback
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return 0.5 * (value[0] + value[1])
    return float(value)


def _orient_vector(vec: Any, player: Player, normalize: bool = False) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    if player == Player.B:
        arr = np.array([-arr[0], -arr[1], arr[2]], dtype=float)
    if normalize:
        norm = np.linalg.norm(arr)
        if norm > 1.0e-8:
            arr = arr / norm
    return arr


def _deg_from_plane(angle_deg: float) -> float:
    """Convert paddle angle measured from table plane to radians relative to vertical."""
    return np.radians(angle_deg - 90.0)


def create_stroke_from_mode(
    player: Player,
    mode: str = "topspin",
    rubber_type: RubberType = RubberType.INVERTED,
    overrides: Optional[Dict[str, Any]] = None,
) -> StrokeParams:
    """Create a stroke that follows a given modality."""
    overrides = overrides or {}
    profile = STROKE_MODALITIES.get(mode)
    if profile is None:
        raise ValueError(f"Unknown stroke mode '{mode}'")

    angle_deg = overrides.get(
        "angle_deg",
        profile.get("angle_deg", _resolve_range(profile.get("angle_deg_range"), 90.0)),
    )
    strike_height = overrides.get("strike_height", profile.get("strike_height", 0.25))
    swing_speed = overrides.get(
        "swing_speed",
        profile.get("swing_speed", _resolve_range(profile.get("swing_speed_range"), 6.0)),
    )
    direction = overrides.get("direction", profile.get("direction", (1.0, 0.0, 0.3)))
    swing_direction = _orient_vector(direction, player, normalize=True)

    contact_offset = overrides.get("contact_offset", profile.get("contact_offset", (0.0, 0.0, 0.0)))
    contact_offset_vec = _orient_vector(contact_offset, player, normalize=False)

    target_x = overrides.get("target_x")
    if target_x is None:
        offset = profile.get("target_x_offset", 0.3 * TABLE_LENGTH)
        target_x = -offset if player == Player.A else offset

    spin_intent = overrides.get("spin_intent", profile.get("spin", mode))

    return StrokeParams(
        target_x=target_x,
        strike_height=float(strike_height),
        racket_angle=_deg_from_plane(float(angle_deg)),
        swing_speed=float(swing_speed),
        swing_direction=swing_direction,
        rubber_type=rubber_type,
        spin_intent=spin_intent,
        mode=mode,
        contact_offset=contact_offset_vec,
    )


def create_default_stroke(
    player: Player,
    stroke_type: str = "topspin",
    rubber_type: RubberType = RubberType.INVERTED,
) -> StrokeParams:
    """Backward-compatible wrapper that delegates to stroke modalities."""
    return create_stroke_from_mode(player, stroke_type, rubber_type)


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
    # Apply racket angle (rotation around y-axis)
    angle = stroke.racket_angle
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    if player == Player.A:
        normal = np.array([cos_a, 0.0, sin_a])
    else:
        normal = np.array([-cos_a, 0.0, sin_a])
    normal = normal / np.linalg.norm(normal)

    # Determine strike contact point relative to current ball state
    contact_point = np.array([
        ball_pos[0],
        ball_pos[1],
        max(ball_pos[2], table_height + BALL_RADIUS * 1.05),
    ])
    contact_point = contact_point + stroke.contact_offset
    racket_pos = contact_point - normal * BALL_RADIUS

    # Velocity emphasizes swing direction but biases toward opponent's side
    to_opponent = np.array([1.0, 0.0, 0.1]) if player == Player.A else np.array([-1.0, 0.0, 0.1])
    incoming = ball_vel
    incoming_norm = np.linalg.norm(incoming)
    if incoming_norm > 1.0e-6:
        incoming = incoming / incoming_norm
    blended_dir = (
        0.65 * stroke.swing_direction
        + 0.25 * to_opponent
        - 0.10 * incoming
    )
    blended_norm = np.linalg.norm(blended_dir)
    if blended_norm > 1.0e-8:
        swing_dir = blended_dir / blended_norm
    else:
        swing_dir = stroke.swing_direction
    velocity = swing_dir * stroke.swing_speed

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
    # Height gating: only hit when ball is near the desired strike height above table
    desired_height = TABLE_HEIGHT + stroke.strike_height
    height_window = RACKET_STRIKE_HEIGHT_WINDOW
    if abs(ball_pos[2] - desired_height) > height_window:
        return False

    # Side gating: player only hits when ball is on their half of the table
    if player == Player.A and ball_pos[0] > 0.0:
        return False
    if player == Player.B and ball_pos[0] < 0.0:
        return False

    # Once height and side conditions are met (and can_prepare_hit is True in the
    # simulation state machine), the player should attempt to hit.
    return True


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
    stroke_configs: List[Any],
    player: Player,
) -> List[StrokeParams]:
    """Create stroke list from configuration tuples or dictionaries."""
    strokes: List[StrokeParams] = []
    for config in stroke_configs:
        overrides: Dict[str, Any] = {}
        if isinstance(config, tuple):
            stroke_type, rubber_name = config
        elif isinstance(config, dict):
            stroke_type = config.get("mode") or config.get("stroke_type") or config.get("type") or "custom"
            rubber_name = config.get("rubber", "inverted")
            overrides = config.get("overrides", {})
        else:
            raise ValueError(f"Unsupported stroke configuration: {config}")

        rubber = parse_rubber_type(rubber_name)
        strokes.append(create_stroke_from_mode(player, stroke_type, rubber, overrides))
    return strokes
