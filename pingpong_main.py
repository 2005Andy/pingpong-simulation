"""Main entry point for ball sports simulation.

This module provides the command-line interface for running ball trajectory
simulations with various scenarios and output options.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from constants import (
    DEFAULT_SCENARIO, DEFAULT_OUTPUT_DIR, DEFAULT_BALL_CSV, DEFAULT_RACKET_CSV,
    DEFAULT_ANIM_FILE, TIME_STEP, MAX_TIME, CUSTOM_INITIAL_POSITION,
    CUSTOM_INITIAL_VELOCITY, CUSTOM_INITIAL_OMEGA, CUSTOM_STROKES_A, CUSTOM_STROKES_B
)
from scenarios import create_table, create_net, create_serve_scenario, create_smash_scenario, create_custom_scenario
from simulation import simulate
from visualization import (
    print_simulation_summary, save_ball_history_to_csv, save_racket_history_to_csv,
    plot_trajectory_3d, animate_trajectory_3d
)
from racket_control import create_custom_strokes
from ball_types import Player

import matplotlib.pyplot as plt


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
  python pingpong_main.py --scenario serve
  python pingpong_main.py --scenario smash --duration 3.0
  python pingpong_main.py --scenario custom --pos -1.2 0 0.9 --vel 8 0 3 --omega 0 150 0
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
