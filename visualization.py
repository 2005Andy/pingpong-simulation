"""Visualization and output utilities for ball sports simulation.

This module provides functions for plotting trajectories, creating animations,
and saving simulation results to files.
"""

from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from constants import RACKET_RADIUS, ANIM_FPS, ANIM_SKIP
from ball_types import SimulationResult, Table, Net, EventType
from physics import _orthonormal_basis_from_normal

import numpy as np


def save_ball_history_to_csv(history: Dict[str, np.ndarray], filename: str) -> None:
    """Save ball trajectory to CSV file."""
    df = pd.DataFrame(history)
    df.to_csv(filename, index=False, float_format="%.6f")


def save_racket_history_to_csv(history: Dict[str, np.ndarray], filename: str) -> None:
    """Save racket trajectory to CSV file."""
    df = pd.DataFrame(history)
    df.to_csv(filename, index=False, float_format="%.6f")


def print_simulation_summary(result: SimulationResult) -> None:
    """Print a summary of the simulation results."""
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Total rallies: {result.rally_count}")
    print(f"Net crossings: {result.net_crossings}")
    print(f"Table bounces: {result.table_bounces}")
    print(f"Final event: {result.final_event.name}")
    print("\nEvent log:")
    for t, event, desc in result.events:
        print(f"  t={t:.4f}s: [{event.name}] {desc}")
    print("=" * 60 + "\n")


def draw_table_3d(ax: Axes3D, table: Table, net: Net) -> None:
    """Draw the table and net on a 3D axis."""
    half_len = table.length / 2.0
    half_wid = table.width / 2.0
    z = table.height

    # Table surface
    table_verts = [
        [(-half_len, -half_wid, z), (half_len, -half_wid, z),
         (half_len, half_wid, z), (-half_len, half_wid, z)]
    ]
    table_poly = Poly3DCollection(table_verts, alpha=0.4, facecolor='darkgreen', edgecolor='white', linewidth=1)
    ax.add_collection3d(table_poly)

    # Center line
    ax.plot([0, 0], [-half_wid, half_wid], [z, z], 'w-', linewidth=1)

    # Net
    net_half_len = net.length / 2.0
    net_top = z + net.height
    net_verts = [
        [(0, -net_half_len, z), (0, net_half_len, z),
         (0, net_half_len, net_top), (0, -net_half_len, net_top)]
    ]
    net_poly = Poly3DCollection(net_verts, alpha=0.3, facecolor='gray', edgecolor='black', linewidth=0.5)
    ax.add_collection3d(net_poly)

    # Net posts
    ax.plot([0, 0], [-net_half_len, -net_half_len], [z, net_top], 'k-', linewidth=2)
    ax.plot([0, 0], [net_half_len, net_half_len], [z, net_top], 'k-', linewidth=2)


def draw_racket_3d(ax: Axes3D, position: np.ndarray, normal: np.ndarray, radius: float, color: str) -> None:
    """Draw a circular racket on a 3D axis."""
    # Create circle points in local coordinates
    theta = np.linspace(0, 2 * np.pi, 32)
    t1, t2, n = _orthonormal_basis_from_normal(normal)

    circle_points = []
    for th in theta:
        p = position + radius * (np.cos(th) * t1 + np.sin(th) * t2)
        circle_points.append(p)

    # Draw as polygon
    verts = [circle_points]
    poly = Poly3DCollection(verts, alpha=0.6, facecolor=color, edgecolor='black', linewidth=1)
    ax.add_collection3d(poly)


def plot_trajectory_3d(result: SimulationResult, table: Table, net: Net) -> plt.Figure:
    """Create a 3D plot of the ball trajectory with table, net, and rackets."""
    x = result.ball_history["x"]
    y = result.ball_history["y"]
    z = result.ball_history["z"]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw table and net
    draw_table_3d(ax, table, net)

    # Draw initial and final racket positions
    ra = result.racket_a_history
    rb = result.racket_b_history
    if len(ra["x"]) > 0:
        # Draw racket A at a few key positions
        for i in [0, len(ra["x"]) // 2, -1]:
            pos_a = np.array([ra["x"][i], ra["y"][i], ra["z"][i]])
            norm_a = np.array([ra["nx"][i], ra["ny"][i], ra["nz"][i]])
            draw_racket_3d(ax, pos_a, norm_a, RACKET_RADIUS, 'red')

    if len(rb["x"]) > 0:
        for i in [0, len(rb["x"]) // 2, -1]:
            pos_b = np.array([rb["x"][i], rb["y"][i], rb["z"][i]])
            norm_b = np.array([rb["nx"][i], rb["ny"][i], rb["nz"][i]])
            draw_racket_3d(ax, pos_b, norm_b, RACKET_RADIUS, 'blue')

    # Plot ball trajectory
    ax.plot(x, y, z, 'orange', linewidth=2, label='Ball trajectory')
    ax.scatter(x[0], y[0], z[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, marker='x', label='End')

    # Mark bounce points
    events = result.ball_history["event"]
    bounce_mask = events == EventType.TABLE_BOUNCE.value
    if np.any(bounce_mask):
        ax.scatter(x[bounce_mask], y[bounce_mask], z[bounce_mask],
                   color='yellow', s=50, marker='^', label='Bounce')

    # Axis settings
    half_len = table.length / 2.0
    half_wid = table.width / 2.0
    margin = 0.5

    ax.set_xlim(-half_len - margin - 1, half_len + margin + 1)
    ax.set_ylim(-half_wid - margin, half_wid + margin)
    ax.set_zlim(table.height - 0.1, max(np.max(z) + 0.3, table.height + 0.8))

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("3D Ping-Pong Ball Trajectory")
    ax.legend(loc='upper left')

    return fig


def animate_trajectory_3d(
    result: SimulationResult,
    table: Table,
    net: Net,
    filename: str,
    fps: int = ANIM_FPS,
    skip: int = ANIM_SKIP,
) -> None:
    """Create and save a 3D animation of the trajectory."""
    x = result.ball_history["x"][::skip]
    y = result.ball_history["y"][::skip]
    z = result.ball_history["z"][::skip]
    t_arr = result.ball_history["t"][::skip]

    ra = {k: v[::skip] for k, v in result.racket_a_history.items()}
    rb = {k: v[::skip] for k, v in result.racket_b_history.items()}

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    half_len = table.length / 2.0
    half_wid = table.width / 2.0

    # Draw static elements
    draw_table_3d(ax, table, net)

    # Initialize animated elements
    (traj_line,) = ax.plot([], [], [], 'orange', linewidth=2)
    (ball_point,) = ax.plot([], [], [], 'o', color='orange', markersize=10)

    ax.set_xlim(-half_len - 1.5, half_len + 1.5)
    ax.set_ylim(-half_wid - 0.5, half_wid + 0.5)
    ax.set_zlim(table.height - 0.1, max(np.max(z) + 0.3, table.height + 0.8))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    title = ax.set_title("")

    def init():
        traj_line.set_data([], [])
        traj_line.set_3d_properties([])
        ball_point.set_data([], [])
        ball_point.set_3d_properties([])
        return traj_line, ball_point

    def update(frame: int):
        traj_line.set_data(x[:frame + 1], y[:frame + 1])
        traj_line.set_3d_properties(z[:frame + 1])
        ball_point.set_data([x[frame]], [y[frame]])
        ball_point.set_3d_properties([z[frame]])
        title.set_text(f"t = {t_arr[frame]:.3f} s")
        return traj_line, ball_point

    interval_ms = 1000.0 / fps
    ani = animation.FuncAnimation(
        fig, update, frames=len(x),
        init_func=init, interval=interval_ms, blit=False
    )

    try:
        ani.save(filename, fps=fps, dpi=120)
        print(f"Animation saved to {filename}")
    except Exception as exc:
        print(f"Warning: failed to save animation to {filename}: {exc}")
        print("Make sure FFmpeg is installed for video export.")
    finally:
        plt.close(fig)
