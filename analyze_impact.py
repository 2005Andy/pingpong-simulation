import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Import from project framework
from constants import (
    TABLE_LENGTH, TABLE_HEIGHT, BALL_RADIUS, TABLE_WIDTH,
    DEFAULT_OUTPUT_DIR
)
from ball_types import BallState
from scenarios import create_table, create_net
from pingpong_sim import simulate, SimulationResult
from visualization import animate_trajectory_3d

def run_analysis(speed: float, angle: float, spin: list, output_dir: str):
    """
    Run a trajectory analysis with specified launch angle.
    
    Args:
        speed (float): Initial speed in m/s.
        angle (float): Launch angle in degrees.
        spin (list): Initial angular velocity [wx, wy, wz] in rad/s.
        output_dir (str): Directory to save results.
    """
    print(f"Initializing Analysis:")
    print(f"  - Speed: {speed} m/s")
    print(f"  - Angle: {angle} degrees")
    print(f"  - Spin: {spin} rad/s")
    
    # 1. Setup Initial Conditions
    angle_rad = np.radians(angle)
    
    # Velocity components
    # Launching towards +x direction
    vx = speed * np.cos(angle_rad)
    vy = 0.0 # Planar trajectory in X-Z plane ideally, unless side spin
    vz = speed * np.sin(angle_rad)
    
    # Position: -x side (edge), y=0, z=table_height
    # We add BALL_RADIUS to z to place the ball *on* the surface, not inside it.
    start_x = -TABLE_LENGTH / 2
    start_y = 0.0
    start_z = TABLE_HEIGHT + 0.3 # Small epsilon to avoid interpenetration
    
    pos = np.array([start_x, start_y, start_z])
    vel = np.array([vx, vy, vz])
    omega = np.array(spin)
    
    initial_ball = BallState(pos, vel, omega)
    
    # 2. Setup Environment
    table = create_table()
    net = create_net()
    
    # No strokes (free flight + bounce)
    strokes_a = []
    strokes_b = []
    
    # 3. Run Simulation
    # We'll run for enough time to see the full trajectory (e.g. 3 seconds)
    print("Running simulation...")
    result = simulate(
        initial_ball=initial_ball,
        strokes_a=strokes_a,
        strokes_b=strokes_b,
        table=table,
        net=net,
        max_time=3.0
    )
    
    # 4. Process Outputs
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename prefix based on parameters
    # Format: traj_v{speed}_ang{angle}_spin{wx}_{wy}_{wz}
    spin_str = f"{int(spin[0])}_{int(spin[1])}_{int(spin[2])}"
    base_name = f"traj_v{speed:.1f}_ang{angle:.1f}_spin{spin_str}"
    
    # (1) X-Z Trajectory Curve
    png_path_xz = out_path / f"{base_name}_xz.png"
    plot_xz_curve(result, table, net, png_path_xz, angle)
    print(f"Saved trajectory plot to {png_path_xz}")

    # (2) X-Y Trajectory Curve (Top View)
    png_path_xy = out_path / f"{base_name}_xy.png"
    plot_xy_curve(result, table, net, png_path_xy, angle)
    print(f"Saved trajectory plot to {png_path_xy}")
    
    # (3) Video Generation
    video_file = out_path / f"{base_name}.gif"
    # Note: If .gif fails (no ImageMagick), fallback to .mp4
    try:
        print(f"Generating video {video_file}...")
        animate_trajectory_3d(result, table, net, str(video_file))
    except Exception as e:
        print(f"GIF generation failed ({e}), falling back to MP4...")
        video_file = out_path / f"{base_name}.mp4"
        animate_trajectory_3d(result, table, net, str(video_file))
        
    print(f"Saved video to {video_file}")

def plot_xz_curve(result: SimulationResult, table, net, filename, angle):
    """Plot the trajectory in the X-Z plane with Z=0 at table height."""
    x = result.ball_history["x"]
    # Shift Z coordinates so table surface is at 0
    z = result.ball_history["z"] - table.height
    
    plt.figure(figsize=(12, 6))
    
    # Plot Trajectory
    plt.plot(x, z, label=f"Ball Trajectory ({angle}°)", linewidth=2, color='blue')
    
    # Start/End points (shifted)
    plt.scatter([x[0]], [z[0]], color='green', label='Start')
    plt.scatter([x[-1]], [z[-1]], color='red', label='End')
    
    # Draw Table (Side View) at Z=0
    table_x_min = -table.length / 2
    table_x_max = table.length / 2
    plt.plot([table_x_min, table_x_max], [0, 0], 
             color='black', linewidth=3, label='Table Surface')
    
    # Draw Net (Side View) relative to table
    net_x = 0
    net_top = net.height # Net height is already relative to table surface
    plt.plot([net_x, net_x], [0, net_top], 
             color='gray', linestyle='-', linewidth=2, label='Net')
    
    # Formatting
    plt.title(f"Ball Trajectory (Side View, Angle={angle}°)")
    plt.xlabel("Length (X) [m]")
    plt.ylabel("Height above Table (Z) [m]")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal') # Important to see real arc
    
    # Ensure we see the whole context
    # Since table is at 0, we might want to see a bit below it if ball falls off
    plt.ylim(bottom=min(np.min(z), -0.2))
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_xy_curve(result: SimulationResult, table, net, filename, angle):
    """Plot the trajectory in the X-Y plane (Top View)."""
    x = result.ball_history["x"]
    y = result.ball_history["y"]
    
    plt.figure(figsize=(8, 12))
    
    # Plot Trajectory
    plt.plot(x, y, label=f"Ball Trajectory", linewidth=2, color='blue')
    
    # Start/End points
    plt.scatter([x[0]], [y[0]], color='green', label='Start')
    plt.scatter([x[-1]], [y[-1]], color='red', label='End')
    
    # Draw Table Boundary
    half_len = table.length / 2
    half_wid = table.width / 2
    
    # Table outline rectangle
    table_x = [-half_len, half_len, half_len, -half_len, -half_len]
    table_y = [-half_wid, -half_wid, half_wid, half_wid, -half_wid]
    plt.plot(table_x, table_y, color='black', linewidth=2, label='Table Edge')
    
    # Draw Center Line (y=0)
    plt.plot([-half_len, half_len], [0, 0], color='lightgray', linestyle='--', linewidth=1)
    
    # Draw Net
    net_half_len = net.length / 2
    plt.plot([0, 0], [-net_half_len, net_half_len], 
             color='gray', linewidth=3, label='Net')
    
    # Formatting
    plt.title(f"Ball Trajectory (Top View, Angle={angle}°)")
    plt.xlabel("Length (X) [m]")
    plt.ylabel("Width (Y) [m]")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    
    # Set limits to include table and some margin
    margin = 0.5
    plt.xlim(-half_len - margin, half_len + margin)
    plt.ylim(-half_wid - margin, half_wid + margin)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ping-pong ball trajectory with configurable launch.")
    parser.add_argument("--speed", type=float, default=5.0, help="Initial speed (magnitude) in m/s")
    parser.add_argument("--angle", type=float, default=45.0, help="Launch angle in degrees (default: 45.0)")
    parser.add_argument("--spin", type=float, nargs=3, default=[0.0, 50.0, 0.0], 
                        help="Initial spin [wx, wy, wz] in rad/s (default: topspin)")
    parser.add_argument("--out", type=str, default="output/analysis", help="Output directory")
    
    args = parser.parse_args()
    
    run_analysis(args.speed, args.angle, args.spin, args.out)
