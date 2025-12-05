# API 参考文档

## 核心模块

### simulation 模块

#### simulate() 函数

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

**参数**:
- `initial_ball`: 初始球体状态
- `strokes_a`: A选手击球序列
- `strokes_b`: B选手击球序列
- `table`: 球桌配置
- `net`: 球网配置
- `dt`: 时间步长 (秒)
- `max_time`: 最大仿真时间 (秒)
- `record_interval`: 记录间隔 (步数)

**返回值**: SimulationResult 对象，包含完整轨迹数据

### physics 模块

#### aerodynamic_acceleration()

```python
def aerodynamic_acceleration(velocity: np.ndarray, omega: np.ndarray) -> np.ndarray
```

计算空气动力学加速度，包括重力、阻力和马格努斯力。

**参数**:
- `velocity`: 球体速度向量 (m/s)
- `omega`: 球体角速度向量 (rad/s)

**返回值**: 加速度向量 (m/s²)

#### rk4_step()

```python
def rk4_step(state: BallState, dt: float) -> BallState
```

使用四阶龙格-库塔方法积分一步。

**参数**:
- `state`: 当前球体状态
- `dt`: 时间步长

**返回值**: 新的球体状态

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

处理球体与平面的碰撞。

**参数**:
- `state`: 球体状态 (会被修改)
- `plane_point`: 平面上一点
- `normal`: 平面法向量
- `restitution`: 法向恢复系数
- `friction`: 切向摩擦系数
- `surface_velocity`: 表面速度

**返回值**: 是否发生碰撞

### racket_control 模块

#### create_stroke_from_mode()

```python
def create_stroke_from_mode(
    player: Player,
    mode: str = "topspin",
    rubber_type: RubberType = RubberType.INVERTED,
    overrides: Optional[Dict[str, Any]] = None,
) -> StrokeParams
```

从预定义模式创建击球参数。

**参数**:
- `player`: 选手 (A 或 B)
- `mode`: 击球模式 ("flick", "topspin", "backspin"等)
- `rubber_type`: 胶皮类型
- `overrides`: 参数覆盖字典

**返回值**: StrokeParams 对象

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

计算击球所需的球拍状态。

**参数**:
- `ball_pos`: 球体位置
- `ball_vel`: 球体速度
- `stroke`: 击球参数
- `player`: 击球选手
- `table_height`: 球桌高度

**返回值**: 球拍状态配置

### scenarios 模块

#### create_table()

```python
def create_table() -> Table
```

创建标准ITTF球桌配置。

**返回值**: Table 对象

#### create_net()

```python
def create_net() -> Net
```

创建标准ITTF球网配置。

**返回值**: Net 对象

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

创建自定义仿真场景。

**参数**:
- `position`: 初始位置
- `velocity`: 初始速度
- `omega`: 初始角速度
- `strokes_a`: A选手击球序列
- `strokes_b`: B选手击球序列

**返回值**: (球体状态, A击球序列, B击球序列)

### visualization 模块

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

创建3D轨迹图。

**参数**:
- `result`: 仿真结果
- `table`: 球桌配置
- `net`: 球网配置
- `ball_color`: 球体颜色
- `ball_size`: 球体大小
- `scene_margin`: 场景边距

**返回值**: matplotlib Figure 对象

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

创建3D轨迹动画。

**参数**:
- `result`: 仿真结果
- `table`: 球桌配置
- `net`: 球网配置
- `filename`: 输出文件名
- `fps`: 帧率
- `skip`: 帧跳跃间隔

### 数据结构

#### BallState

```python
@dataclass
class BallState:
    position: np.ndarray  # (x, y, z) 位置
    velocity: np.ndarray  # (vx, vy, vz) 速度
    omega: np.ndarray     # (ωx, ωy, ωz) 角速度
```

#### Table

```python
@dataclass
class Table:
    height: float      # 桌面高度
    length: float      # 长度 (x方向)
    width: float       # 宽度 (y方向)
    restitution: float # 恢复系数
    friction: float    # 摩擦系数
```

#### StrokeParams

```python
@dataclass
class StrokeParams:
    target_x: float           # 击球目标x位置
    strike_height: float      # 击球高度
    racket_angle: float       # 球拍角度 (弧度)
    swing_speed: float        # 挥拍速度
    swing_direction: np.ndarray # 挥拍方向
    rubber_type: RubberType   # 胶皮类型
    spin_intent: str          # 旋转意图
    mode: str = "custom"      # 击球模式
```

#### SimulationResult

```python
@dataclass
class SimulationResult:
    ball_history: Dict[str, np.ndarray]    # 球体轨迹历史
    racket_a_history: Dict[str, np.ndarray] # A选手球拍历史
    racket_b_history: Dict[str, np.ndarray] # B选手球拍历史
    events: List[Tuple[float, EventType, str]] # 事件列表
    net_crossings: int     # 过网次数
    table_bounces: int     # 桌反弹次数
    rally_count: int       # 回合数
    final_event: EventType # 最终事件
    winner: Optional[Player] = None      # 获胜者
    winner_reason: str = ""              # 获胜原因
```

## 枚举类型

### RubberType
- `INVERTED`: 反胶 (高旋转，进攻性)
- `PIMPLED`: 生胶 (中等旋转，控制性)
- `ANTISPIN`: 防弧胶 (低旋转，防守性)

### EventType
- `NONE = 0`: 无事件
- `TABLE_BOUNCE = 1`: 桌反弹
- `RACKET_A_HIT = 2`: A选手击球
- `RACKET_B_HIT = 3`: B选手击球
- `NET_HIT = 4`: 触网
- `NET_CROSS_SUCCESS = 5`: 成功过网
- `NET_CROSS_FAIL = 6`: 过网失败
- `OUT_OF_BOUNDS = 7`: 出界
- `DOUBLE_BOUNCE = 8`: 双反弹

### Player
- `A`: 选手A (负x侧)
- `B`: 选手B (正x侧)
