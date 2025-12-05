# 用户指南

## 快速开始

### 环境要求

- Python 3.8+
- NumPy 1.21+
- Pandas 1.3+
- Matplotlib 3.5+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本运行

```bash
# 发球仿真
python pingpong_main.py --serve-mode fh_under

# 自定义初始条件
python pingpong_main.py --serve-mode custom --pos -1.2 0 0.9 --vel 8 0 3 --omega 0 150 0

# 轨迹分析工具
python analyze_impact.py --speed 5.0 --angle 30.0 --spin 0 150 0
```

## 命令行参数详解

### pingpong_main.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--serve-mode` | str | fh_under | 发球模式 (fh_under, fast_long, custom) |
| `--output-dir` | str | ./output | 输出目录 |
| `--duration` | float | 10.0 | 仿真时长(秒) |
| `--dt` | float | 5e-5 | 时间步长(秒) |
| `--pos X Y Z` | float×3 | - | 自定义初始位置 |
| `--vel VX VY VZ` | float×3 | - | 自定义初始速度 |
| `--omega WX WY WZ` | float×3 | - | 自定义初始角速度 |

### analyze_impact.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--speed` | float | 5.0 | 初始速度(m/s) |
| `--angle` | float | 45.0 | 发射角度(度) |
| `--spin WX WY WZ` | float×3 | 0 50 0 | 初始旋转(rad/s) |
| `--out` | str | output/analysis | 输出目录 |

## 发球模式说明

### fh_under (正手underspin发球)
- **特点**: 低弧线，旋转向下
- **适用**: 控制节奏，开局战术

### fast_long (快速长球发球)
- **特点**: 高速直线，稍带上旋
- **适用**: 抢攻战术，压制对手

### custom (自定义模式)
- **特点**: 完全自定义初始条件
- **适用**: 研究特定击球效果

## 输出文件说明

### CSV 数据文件

#### ball_trajectory.csv
| 列名 | 说明 | 单位 |
|------|------|------|
| t | 时间 | s |
| x, y, z | 位置坐标 | m |
| vx, vy, vz | 速度分量 | m/s |
| wx, wy, wz | 角速度分量 | rad/s |
| event | 事件代码 | - |

#### racket_trajectory_A/B.csv
| 列名 | 说明 | 单位 |
|------|------|------|
| t | 时间 | s |
| x, y, z | 球拍中心位置 | m |
| vx, vy, vz | 球拍速度 | m/s |
| nx, ny, nz | 球拍法向量 | - |

### 事件代码定义

| 代码 | 事件类型 | 说明 |
|------|----------|------|
| 0 | NONE | 无事件 |
| 1 | TABLE_BOUNCE | 球桌反弹 |
| 2 | RACKET_A_HIT | A选手击球 |
| 3 | RACKET_B_HIT | B选手击球 |
| 4 | NET_HIT | 触网 |
| 5 | NET_CROSS_SUCCESS | 成功过网 |
| 6 | NET_CROSS_FAIL | 过网失败 |
| 7 | OUT_OF_BOUNDS | 出界 |

## 可视化输出

### 3D轨迹图 (trajectory.mp4)
- 显示球体完整运动轨迹
- 标记起点、终点和反弹位置
- 实时显示球拍位置和运动

### 轨迹分析图 (analyze_impact.py)
- **XZ平面图**: 侧视图，显示轨迹弧线
- **XY平面图**: 俯视图，显示平面轨迹

## 高级用法

### 自定义击球序列

修改 `scenarios.py` 中的击球参数：

```python
CUSTOM_STROKES_A = [
    ("flick", "inverted"),      # 轻拉，反胶
    ("topspin", "inverted"),    # 上旋，反胶
    ("backspin", "pimpled"),    # 下旋，生胶
]

CUSTOM_STROKES_B = [
    ("counter_loop", "inverted"),  # 接发球上旋
    ("smash", "inverted"),         # 扣杀
]
```

### 物理参数调整

修改 `constants.py` 中的参数：

```python
# 调整空气阻力
DRAG_COEFF = 0.35  # 减小阻力，增加飞行距离

# 调整马格努斯效应
MAGNUS_COEFF = 0.25  # 增强旋转效果

# 调整球桌属性
TABLE_RESTITUTION = 0.95  # 增加反弹高度
```

### 批量分析

使用脚本进行参数扫描：

```bash
# 不同角度分析
for angle in 20 30 45 60; do
    python analyze_impact.py --angle $angle --out results/angle_$angle
done

# 不同速度分析
for speed in 3 5 8 12; do
    python analyze_impact.py --speed $speed --out results/speed_$speed
done
```

## 故障排除

### 常见问题

1. **动画生成失败**
   - 安装FFmpeg: `conda install ffmpeg` 或 `pip install ffmpeg-python`

2. **内存不足**
   - 减小 `RECORD_INTERVAL` 值
   - 缩短仿真时间 `MAX_TIME`

3. **数值不稳定**
   - 减小 `TIME_STEP` 值
   - 检查初始条件合理性

### 调试模式

启用详细输出：
```bash
python pingpong_main.py --serve-mode custom --dt 1e-4
```

查看中间状态：
```python
# 在代码中添加调试输出
print(f"t={t:.3f}s: pos={ball.position}, vel={ball.velocity}")
```

## 性能优化

### 计算效率
- **时间步长**: 默认5e-5s，在保证精度的前提下可适当增大
- **记录间隔**: RECORD_INTERVAL=20，可根据需要调整
- **仿真时长**: 根据研究需要合理设置MAX_TIME

### 内存管理
- 大规模仿真建议使用HDF5格式存储数据
- 可分批处理大量轨迹数据

### 并行处理
系统支持多核并行，可用于参数扫描研究。
