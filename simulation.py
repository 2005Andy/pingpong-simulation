"""Core simulation engine for ball sports trajectory simulation.

This module contains the main simulation loop and result handling
for ball trajectory simulations with physics-based interactions.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from constants import (
    PLAYER_A_X, PLAYER_B_X, PLAYER_STRIKE_HEIGHT, TIME_STEP, MAX_TIME,
    RECORD_INTERVAL, RACKET_RADIUS, RUBBER_INVERTED_RESTITUTION,
    RUBBER_INVERTED_FRICTION, BALL_RADIUS,
    RACKET_PREDICTION_DT_MULTIPLIER, RACKET_PREDICTION_MIN_DT, RACKET_PREDICTION_MAX_TIME
)
from ball_types import (
    BallState, Table, Net, RacketState, StrokeParams, Player, RubberType,
    EventType, SimulationResult
)
from physics import (
    rk4_step, handle_plane_collision, handle_circular_racket_collision,
    check_net_collision, check_table_bounds, update_racket_movement,
    start_racket_movement
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
    server: Player = Player.A,
    double_bounce_limit: int = 2,
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
        server: Player that last touched the ball before simulation starts.
        double_bounce_limit: Number of bounces allowed on the receiver's table before point ends.

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
    last_hitter: Optional[Player] = server
    final_event = EventType.NONE
    print(f"Simulation started. Server: Player {server.name}, Last hitter: {last_hitter.name if last_hitter else 'None'}")
    print(f"Player A strokes: {len(strokes_a)}")
    print(f"Player B strokes: {len(strokes_b)}")
    print(f"Table dimensions: {table.length:.1f}m x {table.width:.1f}m x {table.height:.1f}m")
    print(f"Net height: {net.height:.1f}m")
    print(f"Ball radius: {BALL_RADIUS:.1f}m")
    print(f"Racket radius: {RACKET_RADIUS:.1f}m")
    print(f"Time step: {dt:.0e}s, Max time: {max_time:.1f}s")
    print(f"Initial ball position: {ball.position}")
    print(f"Initial ball velocity: {ball.velocity}")
    print(f"Initial ball spin: {ball.omega}")
    winner: Optional[Player] = None
    winner_reason = ""
    bounce_counts = {
        Player.A: 0,
        Player.B: 0,
    }
    print(f"Initial bounce counts: A={bounce_counts[Player.A]}, B={bounce_counts[Player.B]}")
    awaiting_first_bounce = {
        Player.A: False,
        Player.B: False,
    }
    can_prepare_hit = {
        Player.A: False,
        Player.B: False,
    }

    # Serve state: ball is initially served by `server`
    is_serving_phase = True
    server_first_bounce_done = False

    # Table collision parameters
    table_normal = np.array([0.0, 0.0, 1.0])
    table_point = np.array([0.0, 0.0, table.height])

    def opponent(player: Player) -> Player:
        return Player.B if player == Player.A else Player.A

    def court_side_from_x(pos_x: float) -> Player:
        return Player.B if pos_x >= 0.0 else Player.A

    def reset_bounce_counts() -> None:
        bounce_counts[Player.A] = 0
        bounce_counts[Player.B] = 0

    def _crossed_plane(x0: float, x1: float, plane_x: float, player: Player) -> bool:
        if player == Player.A:
            return (x0 - plane_x) >= 0.0 and (x1 - plane_x) <= 0.0
        return (x0 - plane_x) <= 0.0 and (x1 - plane_x) >= 0.0

    def predict_contact_state(
        state: BallState,
        player: Player,
        stroke: StrokeParams,
        dt_predict: float,
        max_prediction_time: float = RACKET_PREDICTION_MAX_TIME,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Predict when the ball will intersect the player's strike plane."""
        direction_ok = (
            state.velocity[0] < -0.1 if player == Player.A else state.velocity[0] > 0.1
        )
        if not direction_ok:
            return None

        plane_x = stroke.target_x
        sim_state = state.copy()
        elapsed = 0.0
        while elapsed < max_prediction_time:
            next_state = rk4_step(sim_state, dt_predict)
            elapsed += dt_predict
            if _crossed_plane(sim_state.position[0], next_state.position[0], plane_x, player):
                denom = next_state.position[0] - sim_state.position[0]
                alpha = 0.0 if abs(denom) < 1.0e-8 else (plane_x - sim_state.position[0]) / denom
                alpha = np.clip(alpha, 0.0, 1.0)
                contact_pos = sim_state.position + alpha * (next_state.position - sim_state.position)
                contact_vel = sim_state.velocity + alpha * (next_state.velocity - sim_state.velocity)
                contact_pos[2] = max(contact_pos[2], table.height + BALL_RADIUS * 1.05)
                contact_time = elapsed - dt_predict + alpha * dt_predict
                return contact_pos, contact_vel, contact_time
            sim_state = next_state
        return None

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

    prediction_step = max(dt * RACKET_PREDICTION_DT_MULTIPLIER, RACKET_PREDICTION_MIN_DT)

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
                fault_player = last_hitter or server
                winner = opponent(fault_player)
                winner_reason = f"Player {fault_player.name} hit the net"
                final_event = net_event
                ball = new_ball
                break
            elif net_event == EventType.NET_CROSS_FAIL:
                events.append((t, net_event, "Ball failed to clear net"))
                fault_player = last_hitter or server
                winner = opponent(fault_player)
                winner_reason = "Ball failed to clear the net"
                final_event = net_event
                ball = new_ball
                break
            elif net_event == EventType.NET_CROSS_SUCCESS:
                net_crossings += 1
                events.append((t, net_event, f"Ball crossed net (crossing #{net_crossings})"))

                # Serve rule: during the very first crossing the ball must have
                # bounced once on the server's court; otherwise it is a serve fault.
                if is_serving_phase and net_crossings == 1:
                    if not server_first_bounce_done:
                        fault_player = server
                        winner = opponent(fault_player)
                        winner_reason = "Serve fault: ball did not bounce on server's court before crossing net"
                        final_event = net_event
                        ball = new_ball
                        break
                    # Valid serve: serving phase ends once ball leaves server side
                    is_serving_phase = False

                reset_bounce_counts()
                receiver = court_side_from_x(new_ball.position[0])
                awaiting_first_bounce[receiver] = True
                can_prepare_hit[receiver] = False
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
                bounce_side = court_side_from_x(new_ball.position[0])
                if is_serving_phase:
                    # Serve phase: ball is still under the initial serve from `server`
                    if bounce_side == server:
                        # First legal bounce on server's own court
                        if not server_first_bounce_done:
                            server_first_bounce_done = True
                        else:
                            # Second bounce on server side before crossing net -> serve fault
                            desc = f"Serve fault: ball bounced twice on Player {server.name}'s court"
                            events.append((t, EventType.DOUBLE_BOUNCE, desc))
                            final_event = EventType.DOUBLE_BOUNCE
                            winner = opponent(server)
                            winner_reason = desc
                            ball = new_ball
                            break
                    else:
                        # Defensive: bounce on receiver side while still in serving phase
                        # Treat as invalid serve sequence
                        desc = "Serve fault: invalid bounce sequence during serve"
                        events.append((t, EventType.DOUBLE_BOUNCE, desc))
                        final_event = EventType.DOUBLE_BOUNCE
                        winner = opponent(server)
                        winner_reason = desc
                        ball = new_ball
                        break
                else:
                    # Regular rally phase
                    if last_hitter is None:
                        last_hitter = server

                    if bounce_side == last_hitter:
                        # After a hit, the ball must not bounce on the hitter's own court.
                        desc = f"Ball bounced on Player {last_hitter.name}'s own court after hit"
                        events.append((t, EventType.DOUBLE_BOUNCE, desc))
                        final_event = EventType.DOUBLE_BOUNCE
                        winner = opponent(last_hitter)
                        winner_reason = desc
                        ball = new_ball
                        break

                    # Bounce on opponent's court
                    receiver = bounce_side
                    bounce_counts[receiver] += 1
                    if bounce_counts[receiver] == 1 and awaiting_first_bounce[receiver]:
                        # First legal bounce on receiver side -> they may prepare to hit
                        can_prepare_hit[receiver] = True
                        awaiting_first_bounce[receiver] = False
                    if bounce_counts[receiver] >= double_bounce_limit:
                        # Double bounce on receiver court -> receiver loses, last hitter wins
                        desc = f"Ball bounced twice on Player {receiver.name}'s court"
                        events.append((t, EventType.DOUBLE_BOUNCE, desc))
                        final_event = EventType.DOUBLE_BOUNCE
                        winner = last_hitter
                        winner_reason = desc
                        ball = new_ball
                        break

        # 3. Check racket collisions
        # Debug info: print every 1000 steps to avoid spam
        if step_count % 1000 == 0:
            print(f"t={t:.3f}s: Ball at {new_ball.position}, vel={new_ball.velocity}")
            print(f"  Awaiting bounce: A={awaiting_first_bounce[Player.A]}, B={awaiting_first_bounce[Player.B]}")
            print(f"  Can prepare hit: A={can_prepare_hit[Player.A]}, B={can_prepare_hit[Player.B]}")
            print(f"  Bounce counts: A={bounce_counts[Player.A]}, B={bounce_counts[Player.B]}")
            print(f"  Racket A pos: {racket_a.position}, moving: {racket_a.movement.is_moving}")
            print(f"  Racket B pos: {racket_b.position}, moving: {racket_b.movement.is_moving}")

        if stroke_idx_a < len(strokes_a) and can_prepare_hit[Player.A] and not racket_a.movement.is_moving:
            stroke_a = strokes_a[stroke_idx_a]
            # For hitting, just move to current ball position if it's in strike zone
            if should_player_hit(new_ball.position, new_ball.velocity, Player.A, stroke_a):
                target_racket_a = compute_racket_for_stroke(
                    new_ball.position, new_ball.velocity,
                    stroke_a, Player.A, table.height
                )
                start_racket_movement(racket_a, target_racket_a, None)  # Instant move
                can_prepare_hit[Player.A] = False  # Prevent further movement until next rally
                print(f"  Player A moving to hit ball at {new_ball.position}")

        # Player B hits when ball is on their side and moving towards them
        if stroke_idx_b < len(strokes_b) and can_prepare_hit[Player.B] and not racket_b.movement.is_moving:
            stroke_b = strokes_b[stroke_idx_b]
            # For hitting, just move to current ball position if it's in strike zone
            if should_player_hit(new_ball.position, new_ball.velocity, Player.B, stroke_b):
                target_racket_b = compute_racket_for_stroke(
                    new_ball.position, new_ball.velocity,
                    stroke_b, Player.B, table.height
                )
                start_racket_movement(racket_b, target_racket_b, None)  # Instant move
                can_prepare_hit[Player.B] = False  # Prevent further movement until next rally
                print(f"  Player B moving to hit ball at {new_ball.position}")

        # Check for collisions (rackets can be moving)
        if stroke_idx_a < len(strokes_a):
            if handle_circular_racket_collision(new_ball, racket_a):
                rally_count += 1
                stroke_idx_a += 1
                last_hitter = Player.A
                # Stop racket movement after collision
                racket_a.movement.is_moving = False
                reset_bounce_counts()
                awaiting_first_bounce[Player.B] = True  # Next rally, B awaits first bounce
                can_prepare_hit[Player.A] = False
                can_prepare_hit[Player.B] = False
                events.append((t, EventType.RACKET_A_HIT, f"Player A hit (rally #{rally_count})"))
                current_event = EventType.RACKET_A_HIT
                print(f"  Player A successfully hit the ball! Rally #{rally_count}")
                print(f"    Ball after hit: pos={new_ball.position}, vel={new_ball.velocity}")

        if stroke_idx_b < len(strokes_b):
            if handle_circular_racket_collision(new_ball, racket_b):
                rally_count += 1
                stroke_idx_b += 1
                last_hitter = Player.B
                # Stop racket movement after collision
                racket_b.movement.is_moving = False
                reset_bounce_counts()
                awaiting_first_bounce[Player.A] = True  # Next rally, A awaits first bounce
                can_prepare_hit[Player.A] = False
                can_prepare_hit[Player.B] = False
                events.append((t, EventType.RACKET_B_HIT, f"Player B hit (rally #{rally_count})"))
                current_event = EventType.RACKET_B_HIT
                print(f"  Player B successfully hit the ball! Rally #{rally_count}")
                print(f"    Ball after hit: pos={new_ball.position}, vel={new_ball.velocity}")

        # Update event in history
        if current_event != EventType.NONE and len(ball_history["event"]) > 0:
            ball_history["event"][-1] = current_event.value

        # 4. Check termination conditions
        # Ball fell below table
        if new_ball.position[2] < table.height - 0.5:
            events.append((t, EventType.OUT_OF_BOUNDS, "Ball fell below table"))
            fault_player = last_hitter or server
            winner = opponent(fault_player)
            winner_reason = "Ball fell below the table surface"
            final_event = EventType.OUT_OF_BOUNDS
            ball = new_ball
            break

        # Ball went out of bounds (x or y)
        if abs(new_ball.position[0]) > table.length + 1.0 or abs(new_ball.position[1]) > table.width + 0.5:
            events.append((t, EventType.OUT_OF_BOUNDS, "Ball out of bounds"))
            fault_player = last_hitter or server
            winner = opponent(fault_player)
            winner_reason = "Ball left the bounds of the table"
            final_event = EventType.OUT_OF_BOUNDS
            ball = new_ball
            break

        ball = new_ball
        t += dt
        step_count += 1

    # Final record
    record_state()
    if final_event != EventType.NONE and len(ball_history["event"]) > 0:
        ball_history["event"][-1] = final_event.value

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
        winner=winner,
        winner_reason=winner_reason,
    )
