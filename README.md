# 乒乓球三维飞行仿真 (3D Ping-Pong Ball Flight Simulation)

基于真实物理参数的乒乓球三维运动轨迹仿真程序，包含空气动力学（阻力+马格努斯效应）、球桌碰撞、球网检测、双侧球拍交互等完整功能。

## 物理参数说明（基于ITTF官方规定和科学文献）

| 参数 | 数值 | 来源 |
|------|------|------|
| 球桌高度 | 0.76 m | ITTF规定 |
| 球桌长度 | 2.74 m | ITTF规定 |
| 球桌宽度 | 1.525 m | ITTF规定 |
| 球网高度 | 0.1525 m (15.25 cm) | ITTF规定 |
| 球网长度 | 1.83 m（含网柱） | ITTF规定 |
| 乒乓球直径 | 40 mm | ITTF规定 |
| 乒乓球质量 | 2.7 g | ITTF规定 |
| 球拍直径 | ~17 cm | 标准球拍 |
| 阻力系数 C_D | 0.40 | 科学文献 |
| 马格努斯系数 C_Ω | 0.20 | 科学文献 |
| 恢复系数（球-桌） | 0.90 | 实验测量 |
| 恢复系数（球-拍） | 0.85-0.93 | 取决于胶皮类型 |

## 功能特性

1. **完整的物理模型**
   - 重力、空气阻力、马格努斯力（旋转升力）
   - 球与平面碰撞（含法向恢复和切向摩擦）
   - 滑动-粘着-过旋三种碰撞状态

2. **球网建模**
   - 真实高度 15.25 cm
   - 过网/下网/触网检测
   - 自动判断球是否成功越过球网

3. **双侧球拍系统**
   - 圆形球拍建模（直径 17 cm）
   - 三种胶皮类型：正胶(inverted)、生胶(pimpled)、防弧(antispin)
   - 不同胶皮具有不同的摩擦系数和恢复系数
   - 支持多回合击球追踪

4. **多场景支持**
   - `serve`: 发球场景
   - `smash`: 扣杀场景
   - `custom`: 自定义初始条件

5. **数据输出**
   - 乒乓球轨迹 CSV（位置、速度、角速度、事件）
   - 球拍轨迹 CSV（位置、速度、法向量）

6. **3D可视化**
   - 球桌、球网、球拍完整渲染
   - 静态轨迹图
   - 动态 MP4 动画

## 安装

```bash
pip install -r requirements.txt
```

注意：生成 MP4 动画需要安装 FFmpeg。

## 使用方法

### 基本用法

```bash
# 发球场景
python pingpong_sim.py --scenario serve

# 扣杀场景
python pingpong_sim.py --scenario smash

# 自定义场景
python pingpong_sim.py --scenario custom --pos -1.2 0 0.9 --vel 8 0 3 --omega 0 150 0
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--scenario` | 场景类型 (serve/smash/custom) | serve |
| `--ball-csv` | 球轨迹输出文件 | ball_trajectory.csv |
| `--racket-csv` | 球拍轨迹输出文件 | racket_trajectory.csv |
| `--duration` | 仿真时长（秒） | 5.0 |
| `--dt` | 时间步长（秒） | 5e-5 |
| `--no-animate` | 不生成动画 | False |
| `--no-plot` | 不显示交互图 | False |
| `--pos X Y Z` | 自定义初始位置 | - |
| `--vel VX VY VZ` | 自定义初始速度 | - |
| `--omega WX WY WZ` | 自定义初始角速度 | - |

### 自定义击球序列

在代码顶部修改以下数组来自定义每回合的击球方式：

```python
# 玩家A的击球序列
CUSTOM_STROKES_A = [
    ("topspin", "inverted"),   # 上旋，正胶
    ("backspin", "inverted"),  # 下旋，正胶
    ("flat", "inverted"),      # 平击，正胶
]

# 玩家B的击球序列
CUSTOM_STROKES_B = [
    ("topspin", "pimpled"),    # 上旋，生胶
    ("sidespin", "inverted"),  # 侧旋，正胶
]
```

击球类型：`topspin`（上旋）、`backspin`（下旋）、`sidespin`（侧旋）、`flat`（平击）

胶皮类型：`inverted`（正胶/反胶）、`pimpled`（生胶）、`antispin`（防弧）

## 输出文件说明

### ball_trajectory.csv

| 列名 | 说明 |
|------|------|
| t | 时间 (s) |
| x, y, z | 位置 (m) |
| vx, vy, vz | 速度 (m/s) |
| wx, wy, wz | 角速度 (rad/s) |
| event | 事件代码 (0=无, 1=球桌弹跳, 2=A击球, 3=B击球, 4=触网, 5=过网, 6=下网, 7=出界) |

### racket_trajectory_A.csv / racket_trajectory_B.csv

| 列名 | 说明 |
|------|------|
| t | 时间 (s) |
| x, y, z | 球拍中心位置 (m) |
| vx, vy, vz | 球拍速度 (m/s) |
| nx, ny, nz | 球拍法向量 |

## 坐标系说明

- **x轴**：沿球桌长度方向，玩家A在负x侧，玩家B在正x侧
- **y轴**：沿球桌宽度方向
- **z轴**：垂直向上，球桌表面在 z = 0.76 m

## 物理模型

### 飞行控制方程

$$\frac{d\mathbf{U}}{dt} = \mathbf{g} - \frac{\rho S C_D}{2M} U \mathbf{U} + \frac{\rho R^3 C_\Omega}{M} \boldsymbol{\Omega} \times \mathbf{U}$$

其中：
- $\rho$ = 空气密度 (1.225 kg/m³)
- $S = \pi R^2$ = 迎风面积
- $C_D$ = 阻力系数 (0.40)
- $C_\Omega$ = 马格努斯系数 (0.20)
- $\boldsymbol{\Omega}$ = 角速度矢量

### 碰撞模型

- **法向碰撞**：$v_{n,out} = -e \cdot v_{n,in}$
- **切向摩擦**：基于Coulomb摩擦模型，支持滑动/粘着/过旋状态
- **角速度更新**：通过冲量-动量定理更新

## 许可证

MIT License
