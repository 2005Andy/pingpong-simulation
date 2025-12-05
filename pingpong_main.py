"""Main entry point for ball sports simulation.

This module provides the command-line interface for running ball trajectory
simulations with various scenarios and output options.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.constants import (
    DEFAULT_OUTPUT_DIR, DEFAULT_BALL_CSV, DEFAULT_RACKET_CSV,
    DEFAULT_ANIM_FILE, TIME_STEP, MAX_TIME, CUSTOM_INITIAL_POSITION,
    CUSTOM_INITIAL_VELOCITY, CUSTOM_INITIAL_OMEGA, CUSTOM_STROKES_A, CUSTOM_STROKES_B,
    DEFAULT_SERVE_MODE, DEFAULT_SERVER, DEFAULT_BALL_COLOR, DEFAULT_BALL_SIZE, DEFAULT_SCENE_MARGIN
)
from src.scenarios import create_table, create_net, create_custom_scenario, SERVE_MODE_CHOICES
from src.simulation import simulate
from src.visualization import (
    print_simulation_summary, save_ball_history_to_csv, save_racket_history_to_csv,
    plot_trajectory_3d, animate_trajectory_3d
)
from src.racket_control import create_custom_strokes
from src.ball_types import Player

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
  python pingpong_main.py --serve-mode fh_under
  python pingpong_main.py --serve-mode fast_long --server B --duration 3.0
  python pingpong_main.py --serve-mode custom --pos -1.2 0 0.9 --vel 8 0 3 --omega 0 150 0
        """
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
        "--serve-mode",
        type=str,
        choices=SERVE_MODE_CHOICES,
        default=DEFAULT_SERVE_MODE,
        help="Serve modality preset (fh_under, fast_long, custom)",
    )
    parser.add_argument(
        "--server",
        type=str,
        choices=["A", "B"],
        default=DEFAULT_SERVER,
        help="Which player serves first (A on negative x, B on positive x)",
    )
    parser.add_argument(
        "--ball-color",
        type=str,
        default=DEFAULT_BALL_COLOR,
        help="Matplotlib color for rendering the ball",
    )
    parser.add_argument(
        "--ball-size",
        type=float,
        default=DEFAULT_BALL_SIZE,
        help="Marker/ball size used in plots and animations",
    )
    parser.add_argument(
        "--scene-margin",
        type=float,
        default=DEFAULT_SCENE_MARGIN,
        help="Extra margin around table when framing plots/animations",
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

    server_player = Player[args.server]
    strokes_a = create_custom_strokes(CUSTOM_STROKES_A, Player.A)
    strokes_b = create_custom_strokes(CUSTOM_STROKES_B, Player.B)
    initial_ball, strokes_a, strokes_b = create_custom_scenario(
        position=args.pos,
        velocity=args.vel,
        omega=args.omega,
        strokes_a=strokes_a,
        strokes_b=strokes_b,
        serve_mode=args.serve_mode,
        server=server_player,
    )

    print(f"Running simulation with serve mode: {args.serve_mode} (server = Player {server_player.name})")
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
        server=server_player,
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
        fig = plot_trajectory_3d(
            result,
            table,
            net,
            ball_color=args.ball_color,
            ball_size=args.ball_size,
            scene_margin=args.scene_margin,
        )
        plt.show()

    if not args.no_animate:
        anim_file = video_dir / DEFAULT_ANIM_FILE
        animate_trajectory_3d(
            result,
            table,
            net,
            str(anim_file),
            ball_color=args.ball_color,
            ball_size=args.ball_size,
            scene_margin=args.scene_margin,
        )


if __name__ == "__main__":
    main()
