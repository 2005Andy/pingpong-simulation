# API Reference Documentation

## Core Modules

### simulation Module

#### simulate() Function

```python
def simulate(
    initial_ball: BallState,
    strokes_a: List[StrokeParams],
    strokes_b: List[StrokeParams],
    table: Table,
    net: Net,
    dt: float = TIME_STEP,
    max_time: float = MAX_TIME,
    record_interval: int = RECORD_INTERVAL,
) -> SimulationResult
```

**Parameters**:
- `initial_ball`: Initial ball state
- `strokes_a`: Stroke sequence for Player A
- `strokes_b`: Stroke sequence for Player B
- `table`: Table configuration
- `net`: Net configuration
- `dt`: Time step (seconds)
- `max_time`: Maximum simulation time (seconds)
- `record_interval`: Record interval (steps)

**Returns**: SimulationResult object containing complete trajectory data

### physics Module

#### aerodynamic_acceleration()

```python
def aerodynamic_acceleration(velocity: np.ndarray, omega: np.ndarray) -> np.ndarray
```

Calculates aerodynamic acceleration including gravity, drag, and Magnus force.

**Parameters**:
- `velocity`: Ball velocity vector (m/s)
- `omega`: Ball angular velocity vector (rad/s)

**Returns**: Acceleration vector (m/s²)

#### rk4_step()

```python
def rk4_step(state: BallState, dt: float) -> BallState
```

Advances state by one step using 4th-order Runge-Kutta method.

**Parameters**:
- `state`: Current ball state
- `dt`: Time step

**Returns**: New ball state

#### handle_plane_collision()

```python
def handle_plane_collision(
    state: BallState,
    plane_point: np.ndarray,
    normal: np.ndarray,
    restitution: float,
    friction: float,
    surface_velocity: np.ndarray,
) -> bool
```

Handles collision between ball and planar surface.

**Parameters**:
- `state`: Ball state (will be modified)
- `plane_point`: A point on the plane
- `normal`: Plane normal vector
- `restitution`: Normal restitution coefficient
- `friction`: Tangential friction coefficient
- `surface_velocity`: Surface velocity

**Returns**: Whether collision impulse was applied

### racket_control Module

#### create_stroke_from_mode()

```python
def create_stroke_from_mode(
    player: Player,
    mode: str = "topspin",
    rubber_type: RubberType = RubberType.INVERTED,
    overrides: Optional[Dict[str, Any]] = None,
) -> StrokeParams
```

Creates stroke parameters from predefined modes.

**Parameters**:
- `player`: Player (A or B)
- `mode`: Stroke mode ("flick", "topspin", "backspin", etc.)
- `rubber_type`: Rubber type
- `overrides`: Parameter override dictionary

**Returns**: StrokeParams object

#### compute_racket_for_stroke()

```python
def compute_racket_for_stroke(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    stroke: StrokeParams,
    player: Player,
    table_height: float,
) -> RacketState
```

Computes racket state required for executing a stroke.

**Parameters**:
- `ball_pos`: Ball position
- `ball_vel`: Ball velocity
- `stroke`: Stroke parameters
- `player`: Hitting player
- `table_height`: Table height

**Returns**: Racket state configuration

### scenarios Module

#### create_table()

```python
def create_table() -> Table
```

Creates standard ITTF table tennis table configuration.

**Returns**: Table object

#### create_net()

```python
def create_net() -> Net
```

Creates standard ITTF table tennis net configuration.

**Returns**: Net object

#### create_custom_scenario()

```python
def create_custom_scenario(
    position: Optional[np.ndarray] = None,
    velocity: Optional[np.ndarray] = None,
    omega: Optional[np.ndarray] = None,
    strokes_a: Optional[List[StrokeParams]] = None,
    strokes_b: Optional[List[StrokeParams]] = None,
) -> Tuple[BallState, List[StrokeParams], List[StrokeParams]]
```

Creates custom simulation scenario.

**Parameters**:
- `position`: Initial position
- `velocity`: Initial velocity
- `omega`: Initial angular velocity
- `strokes_a`: Stroke sequence for Player A
- `strokes_b`: Stroke sequence for Player B

**Returns**: (ball state, A stroke sequence, B stroke sequence)

### visualization Module

#### plot_trajectory_3d()

```python
def plot_trajectory_3d(
    result: SimulationResult,
    table: Table,
    net: Net,
    ball_color: str = DEFAULT_BALL_COLOR,
    ball_size: float = DEFAULT_BALL_SIZE,
    scene_margin: float = DEFAULT_SCENE_MARGIN,
) -> plt.Figure
```

Creates 3D trajectory plot.

**Parameters**:
- `result`: Simulation results
- `table`: Table configuration
- `net`: Net configuration
- `ball_color`: Ball color
- `ball_size`: Ball size
- `scene_margin`: Scene margin

**Returns**: matplotlib Figure object

#### animate_trajectory_3d()

```python
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
) -> None
```

Creates 3D trajectory animation.

**Parameters**:
- `result`: Simulation results
- `table`: Table configuration
- `net`: Net configuration
- `filename`: Output filename
- `fps`: Frames per second
- `skip`: Frame skip interval

### Data Structures

#### BallState

```python
@dataclass
class BallState:
    position: np.ndarray  # (x, y, z) position
    velocity: np.ndarray  # (vx, vy, vz) velocity
    omega: np.ndarray     # (ωx, ωy, ωz) angular velocity
```

#### Table

```python
@dataclass
class Table:
    height: float      # table surface height
    length: float      # length (x direction)
    width: float       # width (y direction)
    restitution: float # restitution coefficient
    friction: float    # friction coefficient
```

#### StrokeParams

```python
@dataclass
class StrokeParams:
    target_x: float           # target x position for stroke
    strike_height: float      # strike height
    racket_angle: float       # racket angle (radians)
    swing_speed: float        # swing speed
    swing_direction: np.ndarray # swing direction
    rubber_type: RubberType   # rubber type
    spin_intent: str          # spin intent
    mode: str = "custom"      # stroke mode
```

#### SimulationResult

```python
@dataclass
class SimulationResult:
    ball_history: Dict[str, np.ndarray]    # ball trajectory history
    racket_a_history: Dict[str, np.ndarray] # Player A racket history
    racket_b_history: Dict[str, np.ndarray] # Player B racket history
    events: List[Tuple[float, EventType, str]] # event list
    net_crossings: int     # net crossing count
    table_bounces: int     # table bounce count
    rally_count: int       # rally count
    final_event: EventType # final event
    winner: Optional[Player] = None      # winner
    winner_reason: str = ""              # win reason
```

## Enumeration Types

### RubberType
- `INVERTED`: Inverted rubber (high spin, offensive)
- `PIMPLED`: Pimpled rubber (medium spin, control)
- `ANTISPIN`: Anti-spin rubber (low spin, defensive)

### EventType
- `NONE = 0`: No event
- `TABLE_BOUNCE = 1`: Table bounce
- `RACKET_A_HIT = 2`: Player A hit
- `RACKET_B_HIT = 3`: Player B hit
- `NET_HIT = 4`: Net touch
- `NET_CROSS_SUCCESS = 5`: Successful net crossing
- `NET_CROSS_FAIL = 6`: Net crossing failure
- `OUT_OF_BOUNDS = 7`: Out of bounds
- `DOUBLE_BOUNCE = 8`: Double bounce

### Player
- `A`: Player A (negative x side)
- `B`: Player B (positive x side)
