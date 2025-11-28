"""Core simulation engine for ball sports trajectory simulation.

This module contains the main simulation loop and result handling
for ball trajectory simulations with physics-based interactions.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from constants import (
    PLAYER_A_X, PLAYER_B_X, PLAYER_STRIKE_HEIGHT, TIME_STEP, MAX_TIME,
    RECORD_INTERVAL, RACKET_RADIUS, RUBBER_INVERTED_RESTITUTION,
    RUBBER_INVERTED_FRICTION
)
from ball_types import (
    BallState, Table, Net, RacketState, StrokeParams, Player, RubberType,
    EventType, SimulationResult
)
from physics import (
    rk4_step, handle_plane_collision, handle_circular_racket_collision,
    check_net_collision, check_table_bounds, update_racket_movement
)
from racket_control import compute_racket_for_stroke, should_player_hit


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
                from physics import start_racket_movement
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
                from physics import start_racket_movement
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
