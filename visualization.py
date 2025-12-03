"""Visualization and output utilities for ball sports simulation.

This module provides functions for plotting trajectories, creating animations,
and saving simulation results to files.
"""

from typing import Dict, Tuple, List
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from constants import (
    RACKET_RADIUS, ANIM_FPS, ANIM_SKIP, DEFAULT_BALL_COLOR, DEFAULT_BALL_SIZE,
    DEFAULT_SCENE_MARGIN, TABLE_SURFACE_COLOR, TABLE_STRIPE_COLOR, TABLE_EDGE_COLOR,
    TABLE_LEG_COLOR, TABLE_LEG_WIDTH, TABLE_CENTER_LINE_WIDTH
)
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
    if result.winner is not None:
        print(f"Winner: Player {result.winner.name}")
        if result.winner_reason:
            print(f"Reason: {result.winner_reason}")
    else:
        print("Winner: undecided")
    print("\nEvent log:")
    for t, event, desc in result.events:
        print(f"  t={t:.4f}s: [{event.name}] {desc}")
    print("=" * 60 + "\n")


def _racket_vertices(position: np.ndarray, normal: np.ndarray, radius: float, segments: int = 32) -> List[np.ndarray]:
    t1, t2, _ = _orthonormal_basis_from_normal(normal)
    theta = np.linspace(0, 2 * np.pi, segments)
    circle_points = [
        position + radius * (np.cos(th) * t1 + np.sin(th) * t2) for th in theta
    ]
    return [circle_points]


def _draw_table_legs(ax: Axes3D, table: Table) -> None:
    half_len = table.length / 2.0
    half_wid = table.width / 2.0
    z_top = table.height  # tabletop
    z_bottom = 0.0        # ground plane
    offsets = [
        (half_len - TABLE_LEG_WIDTH, half_wid - TABLE_LEG_WIDTH),
        (-half_len + TABLE_LEG_WIDTH, half_wid - TABLE_LEG_WIDTH),
        (half_len - TABLE_LEG_WIDTH, -half_wid + TABLE_LEG_WIDTH),
        (-half_len + TABLE_LEG_WIDTH, -half_wid + TABLE_LEG_WIDTH),
    ]
    for ox, oy in offsets:
        verts = [
            [
                (ox - TABLE_LEG_WIDTH, oy - TABLE_LEG_WIDTH, z_bottom),
                (ox + TABLE_LEG_WIDTH, oy - TABLE_LEG_WIDTH, z_bottom),
                (ox + TABLE_LEG_WIDTH, oy + TABLE_LEG_WIDTH, z_bottom),
                (ox - TABLE_LEG_WIDTH, oy + TABLE_LEG_WIDTH, z_bottom),
            ],
            [
                (ox - TABLE_LEG_WIDTH, oy - TABLE_LEG_WIDTH, z_top),
                (ox + TABLE_LEG_WIDTH, oy - TABLE_LEG_WIDTH, z_top),
                (ox + TABLE_LEG_WIDTH, oy + TABLE_LEG_WIDTH, z_top),
                (ox - TABLE_LEG_WIDTH, oy + TABLE_LEG_WIDTH, z_top),
            ],
        ]
        faces = [
            [verts[0][0], verts[0][1], verts[1][1], verts[1][0]],
            [verts[0][1], verts[0][2], verts[1][2], verts[1][1]],
            [verts[0][2], verts[0][3], verts[1][3], verts[1][2]],
            [verts[0][3], verts[0][0], verts[1][0], verts[1][3]],
            verts[0],
            verts[1],
        ]
        leg_poly = Poly3DCollection(faces, facecolor=TABLE_LEG_COLOR, edgecolor='black', linewidth=0.5, alpha=0.9)
        ax.add_collection3d(leg_poly)


def _draw_table_pattern(ax: Axes3D, table: Table) -> None:
    half_len = table.length / 2.0
    half_wid = table.width / 2.0
    z = table.height + 1e-3
    stripe_count = 6
    stripe_width = table.length / stripe_count
    for i in range(stripe_count):
        if i % 2 == 0:
            continue
        x0 = -half_len + i * stripe_width
        x1 = x0 + stripe_width
        verts = [[
            (x0, -half_wid, z),
            (x1, -half_wid, z),
            (x1, half_wid, z),
            (x0, half_wid, z),
        ]]
        stripe_poly = Poly3DCollection(verts, alpha=0.18, facecolor=TABLE_STRIPE_COLOR, edgecolor=None)
        ax.add_collection3d(stripe_poly)


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
    table_poly = Poly3DCollection(
        table_verts,
        alpha=0.9,
        facecolor=TABLE_SURFACE_COLOR,
        edgecolor=TABLE_EDGE_COLOR,
        linewidth=1.5,
    )
    ax.add_collection3d(table_poly)

    # Center line along x-axis at y=0 with configurable physical width
    half_center_wid = TABLE_CENTER_LINE_WIDTH / 2.0
    center_verts = [[
        (-half_len, -half_center_wid, z),
        (half_len, -half_center_wid, z),
        (half_len, half_center_wid, z),
        (-half_len, half_center_wid, z),
    ]]
    center_poly = Poly3DCollection(
        center_verts,
        alpha=0.95,
        facecolor=TABLE_STRIPE_COLOR,
        edgecolor=None,
    )
    ax.add_collection3d(center_poly)

    # Boundary lines (table edges)
    ax.plot([-half_len, half_len], [-half_wid, -half_wid], [z, z], color=TABLE_EDGE_COLOR, linewidth=2)
    ax.plot([-half_len, half_len], [half_wid, half_wid], [z, z], color=TABLE_EDGE_COLOR, linewidth=2)
    ax.plot([-half_len, -half_len], [-half_wid, half_wid], [z, z], color=TABLE_EDGE_COLOR, linewidth=2)
    ax.plot([half_len, half_len], [-half_wid, half_wid], [z, z], color=TABLE_EDGE_COLOR, linewidth=2)

    # Pattern accents
    _draw_table_pattern(ax, table)
    _draw_table_legs(ax, table)

    # Net
    net_half_len = net.length / 2.0
    net_top = z + net.height
    net_verts = [
        [(0, -net_half_len, z), (0, net_half_len, z),
         (0, net_half_len, net_top), (0, -net_half_len, net_top)]
    ]
    net_poly = Poly3DCollection(net_verts, alpha=0.35, facecolor='gray', edgecolor='black', linewidth=0.5)
    ax.add_collection3d(net_poly)

    # Net posts
    ax.plot([0, 0], [-net_half_len, -net_half_len], [z, net_top], 'k-', linewidth=2)
    ax.plot([0, 0], [net_half_len, net_half_len], [z, net_top], 'k-', linewidth=2)


def draw_racket_3d(ax: Axes3D, position: np.ndarray, normal: np.ndarray, radius: float, color: str) -> None:
    """Draw a circular racket on a 3D axis."""
    verts = _racket_vertices(position, normal, radius)
    poly = Poly3DCollection(verts, alpha=0.6, facecolor=color, edgecolor='black', linewidth=1)
    ax.add_collection3d(poly)


def _compute_scene_limits(
    table: Table,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    margin: float,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    half_len = table.length / 2.0
    half_wid = table.width / 2.0
    x_limits = (
        min(np.min(x), -half_len) - margin,
        max(np.max(x), half_len) + margin,
    )
    y_limits = (
        min(np.min(y), -half_wid) - margin,
        max(np.max(y), half_wid) + margin,
    )
    z_floor = 0.0  # ground plane
    z_ceiling = table.height + 0.8
    z_limits = (
        min(np.min(z), z_floor) - margin * 0.5,
        max(np.max(z), z_ceiling) + margin,
    )
    return x_limits, y_limits, z_limits


def plot_trajectory_3d(
    result: SimulationResult,
    table: Table,
    net: Net,
    ball_color: str = DEFAULT_BALL_COLOR,
    ball_size: float = DEFAULT_BALL_SIZE,
    scene_margin: float = DEFAULT_SCENE_MARGIN,
) -> plt.Figure:
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
        sample_idx = np.linspace(0, len(ra["x"]) - 1, num=min(len(ra["x"]), 5), dtype=int)
        for i in sample_idx:
            pos_a = np.array([ra["x"][i], ra["y"][i], ra["z"][i]])
            norm_a = np.array([ra["nx"][i], ra["ny"][i], ra["nz"][i]])
            draw_racket_3d(ax, pos_a, norm_a, RACKET_RADIUS, 'red')

    if len(rb["x"]) > 0:
        sample_idx = np.linspace(0, len(rb["x"]) - 1, num=min(len(rb["x"]), 5), dtype=int)
        for i in sample_idx:
            pos_b = np.array([rb["x"][i], rb["y"][i], rb["z"][i]])
            norm_b = np.array([rb["nx"][i], rb["ny"][i], rb["nz"][i]])
            draw_racket_3d(ax, pos_b, norm_b, RACKET_RADIUS, 'blue')

    # Plot ball trajectory
    ax.plot(x, y, z, color=ball_color, linewidth=2, label='Ball trajectory')
    marker_scale = max(ball_size, 6.0) * 5.0
    ax.scatter(x[0], y[0], z[0], color='green', s=marker_scale, marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=marker_scale, marker='x', label='End')

    # Mark bounce points
    events = result.ball_history["event"]
    bounce_mask = events == EventType.TABLE_BOUNCE.value
    if np.any(bounce_mask):
        ax.scatter(x[bounce_mask], y[bounce_mask], z[bounce_mask],
                   color='yellow', s=50, marker='^', label='Bounce')

    # Axis settings
    x_lim, y_lim, z_lim = _compute_scene_limits(table, x, y, z, scene_margin)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)

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
    ball_color: str = DEFAULT_BALL_COLOR,
    ball_size: float = DEFAULT_BALL_SIZE,
    scene_margin: float = DEFAULT_SCENE_MARGIN,
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

    # Draw static elements
    draw_table_3d(ax, table, net)

    # Initialize animated elements
    (traj_line,) = ax.plot([], [], [], color=ball_color, linewidth=2)
    (ball_point,) = ax.plot([], [], [], 'o', color=ball_color, markersize=ball_size)
    # Start with a degenerate (radius=0) disc at the table center to avoid empty-verts errors
    dummy_pos = np.array([0.0, 0.0, table.height + 0.05])
    dummy_norm = np.array([0.0, 0.0, 1.0])
    racket_a_patch = Poly3DCollection(_racket_vertices(dummy_pos, dummy_norm, 0.0),
                                      alpha=0.6, facecolor='red', edgecolor='black', linewidth=1)
    racket_b_patch = Poly3DCollection(_racket_vertices(dummy_pos, dummy_norm, 0.0),
                                      alpha=0.6, facecolor='blue', edgecolor='black', linewidth=1)
    ax.add_collection3d(racket_a_patch)
    ax.add_collection3d(racket_b_patch)

    x_lim, y_lim, z_lim = _compute_scene_limits(table, result.ball_history["x"], result.ball_history["y"], result.ball_history["z"], scene_margin)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    title = ax.set_title("")

    def init():
        traj_line.set_data([], [])
        traj_line.set_3d_properties([])
        ball_point.set_data([], [])
        ball_point.set_3d_properties([])
        # Keep degenerate discs; no need to reset verts to empty
        return traj_line, ball_point, racket_a_patch, racket_b_patch

    def update(frame: int):
        traj_line.set_data(x[:frame + 1], y[:frame + 1])
        traj_line.set_3d_properties(z[:frame + 1])
        ball_point.set_data([x[frame]], [y[frame]])
        ball_point.set_3d_properties([z[frame]])
        title.set_text(f"t = {t_arr[frame]:.3f} s")
        if len(ra["x"]) > frame:
            pos_a = np.array([ra["x"][frame], ra["y"][frame], ra["z"][frame]])
            norm_a = np.array([ra["nx"][frame], ra["ny"][frame], ra["nz"][frame]])
            racket_a_patch.set_verts(_racket_vertices(pos_a, norm_a, RACKET_RADIUS))
        else:
            # Keep as tiny disc at dummy position (effectively invisible)
            racket_a_patch.set_verts(_racket_vertices(dummy_pos, dummy_norm, 0.0))
        if len(rb["x"]) > frame:
            pos_b = np.array([rb["x"][frame], rb["y"][frame], rb["z"][frame]])
            norm_b = np.array([rb["nx"][frame], rb["ny"][frame], rb["nz"][frame]])
            racket_b_patch.set_verts(_racket_vertices(pos_b, norm_b, RACKET_RADIUS))
        else:
            racket_b_patch.set_verts(_racket_vertices(dummy_pos, dummy_norm, 0.0))
        return traj_line, ball_point, racket_a_patch, racket_b_patch

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
