# User Guide

## Quick Start

### Environment Requirements

- Python 3.8+
- NumPy 1.21+
- Pandas 1.3+
- Matplotlib 3.5+

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Serve simulation
python pingpong_main.py --serve-mode fh_under

# Custom initial conditions
python pingpong_main.py --serve-mode custom --pos -1.2 0 0.9 --vel 8 0 3 --omega 0 150 0

# Trajectory analysis tool
python analyze_impact.py --speed 5.0 --angle 30.0 --spin 0 150 0
```

## Command Line Parameters Details

### pingpong_main.py Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--serve-mode` | str | fh_under | Serve mode (fh_under, fast_long, custom) |
| `--output-dir` | str | ./output | Output directory |
| `--duration` | float | 10.0 | Simulation duration (seconds) |
| `--dt` | float | 5e-5 | Time step (seconds) |
| `--pos X Y Z` | float×3 | - | Custom initial position |
| `--vel VX VY VZ` | float×3 | - | Custom initial velocity |
| `--omega WX WY WZ` | float×3 | - | Custom initial angular velocity |

### analyze_impact.py Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--speed` | float | 5.0 | Initial speed (m/s) |
| `--angle` | float | 45.0 | Launch angle (degrees) |
| `--spin WX WY WZ` | float×3 | 0 50 0 | Initial spin (rad/s) |
| `--out` | str | output/analysis | Output directory |

## Serve Mode Description

### fh_under (Forehand underspin serve)
- **Characteristics**: Low arc, downward spin
- **Usage**: Control pace, opening tactics

### fast_long (Fast long serve)
- **Characteristics**: High speed, slight topspin
- **Usage**: Aggressive tactics, pressuring opponent

### custom (Custom mode)
- **Characteristics**: Fully customizable initial conditions
- **Usage**: Research specific stroke effects

## Output Files Description

### CSV Data Files

#### ball_trajectory.csv
| Column | Description | Unit |
|--------|-------------|------|
| t | Time | s |
| x, y, z | Position coordinates | m |
| vx, vy, vz | Velocity components | m/s |
| wx, wy, wz | Angular velocity components | rad/s |
| event | Event code | - |

#### racket_trajectory_A/B.csv
| Column | Description | Unit |
|--------|-------------|------|
| t | Time | s |
| x, y, z | Racket center position | m |
| vx, vy, vz | Racket velocity | m/s |
| nx, ny, nz | Racket normal vector | - |

### Event Code Definitions

| Code | Event Type | Description |
|------|------------|-------------|
| 0 | NONE | No event |
| 1 | TABLE_BOUNCE | Table bounce |
| 2 | RACKET_A_HIT | Player A hit |
| 3 | RACKET_B_HIT | Player B hit |
| 4 | NET_HIT | Net touch |
| 5 | NET_CROSS_SUCCESS | Successful net crossing |
| 6 | NET_CROSS_FAIL | Net crossing failure |
| 7 | OUT_OF_BOUNDS | Out of bounds |

## Visualization Output

### 3D Trajectory Plot (trajectory.mp4)
- Shows complete ball motion trajectory
- Marks start/end points and bounce positions
- Real-time display of racket position and movement

### Trajectory Analysis Plots (analyze_impact.py)
- **XZ plane plot**: Side view, shows trajectory arc
- **XY plane plot**: Top view, shows planar trajectory

## Advanced Usage

### Custom Stroke Sequences

Modify stroke parameters in `scenarios.py`:

```python
CUSTOM_STROKES_A = [
    ("flick", "inverted"),      # Light push, inverted rubber
    ("topspin", "inverted"),    # Topspin, inverted rubber
    ("backspin", "pimpled"),    # Backspin, pimpled rubber
]

CUSTOM_STROKES_B = [
    ("counter_loop", "inverted"),  # Counter loop
    ("smash", "inverted"),         # Smash
]
```

### Physics Parameter Adjustment

Modify parameters in `constants.py`:

```python
# Adjust air drag
DRAG_COEFF = 0.35  # Reduce drag, increase flight distance

# Adjust Magnus effect
MAGNUS_COEFF = 0.25  # Enhance spin effects

# Adjust table properties
TABLE_RESTITUTION = 0.95  # Increase bounce height
```

### Batch Analysis

Use scripts for parameter sweeps:

```bash
# Different angle analysis
for angle in 20 30 45 60; do
    python analyze_impact.py --angle $angle --out results/angle_$angle
done

# Different speed analysis
for speed in 3 5 8 12; do
    python analyze_impact.py --speed $speed --out results/speed_$speed
done
```

## Troubleshooting

### Common Issues

1. **Animation generation failed**
   - Install FFmpeg: `conda install ffmpeg` or `pip install ffmpeg-python`

2. **Out of memory**
   - Reduce `RECORD_INTERVAL` value
   - Shorten simulation time `MAX_TIME`

3. **Numerical instability**
   - Reduce `TIME_STEP` value
   - Check initial conditions reasonableness

### Debug Mode

Enable verbose output:
```bash
python pingpong_main.py --serve-mode custom --dt 1e-4
```

View intermediate states:
```python
# Add debug output in code
print(f"t={t:.3f}s: pos={ball.position}, vel={ball.velocity}")
```

## Performance Optimization

### Computational Efficiency
- **Time step**: Default 5e-5s, can increase while maintaining accuracy
- **Record interval**: RECORD_INTERVAL=20, adjustable as needed
- **Simulation duration**: Set MAX_TIME according to research needs

### Memory Management
- Large-scale simulations recommend HDF5 format for data storage
- Process large trajectory datasets in batches

### Parallel Processing
System supports multi-core parallel processing for parameter sweep studies.
