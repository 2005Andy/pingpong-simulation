# PingPong 3D Flight Simulation System Architecture Documentation

## Project Overview

This project is a physics-based 3D ping-pong ball trajectory simulation system that implements a complete table tennis physics model, including aerodynamics, collision detection, and racket interaction. The system uses modular design and supports multi-scenario simulation with detailed data output.

## Core Architecture

### Modular Design Principles

The project adopts a strict modular architecture where each module has a single responsibility, making it easy to maintain, test, and extend:

```
PingPong Simulation System/
├── constants.py          # Physical constants and parameter definitions
├── ball_types.py         # Data structures and type definitions
├── physics.py           # Physics calculation engine
├── simulation.py        # Simulation control engine
├── racket_control.py    # Racket control logic
├── scenarios.py         # Scenario configuration management
├── visualization.py     # Data visualization and output
└── pingpong_main.py     # Command-line interface
```

### Module Detailed Description

#### 1. constants.py - Physical Constants Definition

**Function**: Centralized management of all physical constants and system parameters
**Key Constants**:

- **Aerodynamic Parameters**:
  - `AIR_DENSITY = 1.225` (kg/m³, standard atmospheric density)
  - `GRAVITY = [0, 0, -9.81]` (m/s², gravitational acceleration)
  - `DRAG_COEFF = 0.40` (drag coefficient)
  - `MAGNUS_COEFF = 0.20` (Magnus effect coefficient)

- **Ball Parameters (ITTF Standard)**:
  - `BALL_RADIUS = 0.020` (m, ball radius)
  - `BALL_MASS = 0.0027` (kg, ball mass)
  - `BALL_INERTIA_FACTOR = 2/3` (rotational inertia factor)

- **Table Parameters (ITTF Standard)**:
  - `TABLE_LENGTH = 2.74` (m, table length)
  - `TABLE_WIDTH = 1.525` (m, table width)
  - `TABLE_HEIGHT = 0.76` (m, table height)
  - `NET_HEIGHT = 0.1525` (m, net height)

- **Racket Parameters**:
  - `RACKET_RADIUS = 0.085` (m, racket radius)
  - Different rubber type physical properties (inverted, pimpled, anti-spin)

- **Simulation Parameters**:
  - `TIME_STEP = 5e-5` (s, time step)
  - `MAX_TIME = 10.0` (s, maximum simulation time)

#### 2. ball_types.py - Data Structure Definition

**Core Data Classes**:

- **BallState**: Ball state (position, velocity, angular velocity)
- **Table**: Table geometry (dimensions, material properties)
- **Net**: Net definition (height, length)
- **RacketState**: Racket state (position, normal vector, material properties)
- **StrokeParams**: Stroke parameters (target position, racket angle, swing speed)
- **SimulationResult**: Simulation results (trajectory history, event log)

**Enumeration Types**:
- **RubberType**: Rubber types (INVERTED, PIMPLED, ANTISPIN)
- **EventType**: Event types (TABLE_BOUNCE, RACKET_HIT, NET_COLLISION, etc.)
- **Player**: Player identifiers (A, B)

#### 3. physics.py - Physics Calculation Engine

**Aerodynamic Model**:
```python
def aerodynamic_acceleration(velocity, omega):
    # Drag: F_drag = -0.5 * ρ * v² * S * C_d * v̂
    # Magnus force: F_magnus = ρ * R³ * C_Ω * (ω × v)
    return gravity + drag + magnus_force
```

**Collision Detection System**:
- **Plane Collision**: Table collision using impulse-momentum method
- **Circular Racket Collision**: Racket-ball interaction
- **Net Detection**: Trajectory crossing detection

**Numerical Integration**:
- **RK4 Method**: 4th-order Runge-Kutta integration for numerical stability

#### 4. simulation.py - Simulation Control Engine

**Main Simulation Loop**:
```python
while t <= max_time:
    # 1. Update racket movement
    # 2. Record state
    # 3. Detect events (net collision, table collision, racket collision)
    # 4. Integrate ball motion
    # 5. Check termination conditions
```

**Event-Driven Mechanism**:
- **Net Crossing Events**: Success/failure crossing detection
- **Table Bounce Events**: Record bounce position and physics
- **Racket Hit Events**: Record hits and player alternation
- **Out-of-Bounds Events**: Ball leaves valid area

#### 5. racket_control.py - Racket Control Logic

**Stroke Mode Definitions**:
- **flick**: Light push (control-oriented stroke)
- **topspin**: Topspin (offensive stroke)
- **backspin**: Backspin (defensive stroke)
- **flat**: Flat hit (fast stroke)

**Intelligent Decision System**:
```python
def should_player_hit(ball_pos, ball_vel, player, stroke):
    # Determine if player should hit based on ball position, velocity, stroke parameters
    # Consider racket reaction time and hitting window
```

#### 6. scenarios.py - Scenario Configuration Management

**Predefined Scenarios**:
- **serve**: Serve scenarios (various serve types)
- **custom**: Custom initial conditions
- **trajectory**: Trajectory analysis (no racket interaction)

**Parameterized Configuration**:
- Flexible setting of position, velocity, angular velocity
- Custom stroke sequence configuration

#### 7. visualization.py - Visualization Engine

**Output Formats**:
- **CSV Export**: Trajectory data (position, velocity, events)
- **3D Visualization**: Matplotlib 3D plotting
- **Animation Generation**: MP4/GIF formats (requires FFmpeg)

**Visualization Components**:
- 3D rendering of table and net
- Racket position and trajectory display
- Event markers (bounce points, hit points)

## Physics Model Details

### Aerodynamics

The ball motion in air follows a simplified form of the Navier-Stokes equations:

```
m dv/dt = m g - 0.5 ρ v² S C_d v̂ + ρ R³ C_Ω (ω × v)
```

Where:
- **Drag term**: Proportional to velocity squared
- **Magnus force**: Rotation-induced lift, proportional to cross product of angular velocity and velocity

### Collision Model

**Table Collision**:
- **Normal restitution**: ε = 0.90 (ITTF standard)
- **Tangential friction**: μ = 0.25 (considering sliding/sticking/over-spin states)

**Racket Collision**:
- Different rubber types have different physical properties
- Supports complex spin transfer mechanisms

### Table Tennis Rules Implementation

The system fully implements ITTF table tennis rules:
- **Service rules**: Must bounce on server's side first
- **Scoring rules**: Double bounce, net touch, out-of-bounds judgment
- **Rally alternation**: Switch serve after successful hits

## Technical Features

### Performance Optimization
- **Adaptive time stepping**: Adjust calculation precision based on motion intensity
- **Event-driven**: Detailed calculations only at critical moments
- **Vectorized computation**: Efficient numerical calculations using NumPy

### Extensibility
- **Modular design**: Easy to add new sports
- **Parameterized configuration**: Support for different ball types and rules
- **Plugin architecture**: Extensible with new physics models

### Data Integrity
- **Trajectory recording**: Complete position, velocity, angular velocity history
- **Event logging**: Detailed records of all collisions and rule events
- **Metadata**: Complete records of simulation parameters and environmental conditions

## Quality Assurance

### Test Coverage
- **Unit tests**: Independent functionality testing for each module
- **Integration tests**: Complete simulation process validation
- **Physics validation**: Comparison with analytical solutions and experimental data

### Code Quality
- **Type hints**: Complete type annotations
- **Docstrings**: Detailed function and class documentation
- **Code standards**: Following Google Python style guide
