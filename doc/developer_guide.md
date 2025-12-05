# 开发指南

## 项目结构

```
乒乓球仿真系统/
├── doc/                    # 文档目录
│   ├── architecture.md     # 架构文档
│   ├── physics_model.md    # 物理模型详解
│   ├── user_guide.md       # 用户指南
│   ├── api_reference.md    # API参考
│   └── developer_guide.md  # 开发指南
├── src/                    # 源代码目录 (推荐)
│   ├── __init__.py
│   ├── constants.py
│   ├── ball_types.py
│   ├── physics.py
│   ├── simulation.py
│   ├── racket_control.py
│   ├── scenarios.py
│   └── visualization.py
├── tests/                  # 测试目录
│   ├── test_physics.py
│   ├── test_simulation.py
│   └── conftest.py
├── examples/               # 示例脚本
│   ├── basic_simulation.py
│   └── parameter_study.py
├── pingpong_main.py        # 主入口脚本
├── analyze_impact.py       # 分析工具
├── requirements.txt        # 依赖列表
├── setup.py               # 安装脚本
├── LICENSE                # 许可证
├── README.md              # 项目说明
├── CONTRIBUTING.md        # 贡献指南
└── .gitignore            # 忽略文件
```

## 开发环境设置

### 环境要求

- Python 3.8+
- pip 或 conda 包管理器

### 安装开发依赖

```bash
# 克隆项目
git clone https://github.com/username/pingpong-simulation.git
cd pingpong-simulation

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 安装开发依赖
pip install -e .[dev]
```

### 开发工具配置

推荐使用以下工具：

- **代码编辑器**: VSCode, PyCharm
- **版本控制**: Git
- **代码格式化**: Black
- **代码检查**: Flake8, MyPy
- **测试框架**: pytest
- **文档生成**: Sphinx

## 代码规范

### 命名约定

- **变量和函数**: snake_case
- **类**: PascalCase
- **常量**: UPPER_CASE
- **模块**: snake_case

### 类型提示

所有新代码必须包含完整的类型提示：

```python
from typing import List, Optional, Tuple
import numpy as np

def calculate_trajectory(
    initial_pos: np.ndarray,
    initial_vel: np.ndarray,
    time_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """计算轨迹"""
    pass
```

### 文档字符串

使用Google风格的文档字符串：

```python
def aerodynamic_acceleration(
    velocity: np.ndarray,
    omega: np.ndarray
) -> np.ndarray:
    """计算空气动力学加速度。

    基于阻力定律和马格努斯效应计算球体加速度。

    Args:
        velocity: 球体速度向量 (m/s)
        omega: 球体角速度向量 (rad/s)

    Returns:
        加速度向量 (m/s²)

    Examples:
        >>> vel = np.array([5.0, 0.0, 2.0])
        >>> omega = np.array([0.0, 100.0, 0.0])
        >>> acc = aerodynamic_acceleration(vel, omega)
        >>> acc.shape
        (3,)
    """
```

## 测试开发

### 测试结构

```python
# tests/test_physics.py
import numpy as np
import pytest
from src.physics import aerodynamic_acceleration

class TestAerodynamicAcceleration:
    def test_zero_velocity(self):
        """测试静止球体的加速度"""
        vel = np.zeros(3)
        omega = np.zeros(3)
        acc = aerodynamic_acceleration(vel, omega)

        # 应该只有重力
        expected = np.array([0.0, 0.0, -9.81])
        np.testing.assert_array_almost_equal(acc, expected)

    def test_drag_force(self):
        """测试阻力效应"""
        vel = np.array([10.0, 0.0, 0.0])  # 10 m/s 水平速度
        omega = np.zeros(3)
        acc = aerodynamic_acceleration(vel, omega)

        # 应该有阻力分量 (负x方向)
        assert acc[0] < 0.0  # 阻力减速
        assert acc[2] == -9.81  # 重力不变
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_physics.py::TestAerodynamicAcceleration::test_zero_velocity

# 带覆盖率报告
pytest --cov=src --cov-report=html
```

### 测试最佳实践

1. **单元测试**: 每个函数独立测试
2. **集成测试**: 模块间交互测试
3. **性能测试**: 大规模仿真性能验证
4. **物理验证**: 与解析解对比

## 性能优化

### 分析性能瓶颈

```python
import cProfile
import pstats

# 性能分析
profiler = cProfile.Profile()
profiler.enable()

# 运行仿真
result = simulate(...)

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(20)  # 显示前20个耗时函数
```

### 优化策略

1. **向量化计算**: 使用NumPy数组操作
2. **减少内存分配**: 重用数组对象
3. **自适应时间步长**: 根据运动剧烈程度调整dt
4. **事件驱动**: 避免不必要的计算

### 内存优化

```python
# 使用内存映射数组处理大数据
import numpy as np

def save_large_trajectory(filename: str, trajectory: np.ndarray):
    """高效保存大数据轨迹"""
    fp = np.memmap(filename, dtype='float64',
                   mode='w+', shape=trajectory.shape)
    fp[:] = trajectory[:]
    fp.flush()
```

## 扩展开发

### 添加新物理模型

```python
# src/physics.py
def advanced_aerodynamic_model(
    velocity: np.ndarray,
    omega: np.ndarray,
    temperature: float = 20.0,
    humidity: float = 0.5
) -> np.ndarray:
    """高级空气动力学模型，考虑温度和湿度"""
    # 空气密度随温度变化
    rho = 1.225 * (273.15 / (temperature + 273.15))

    # 湿度影响
    # ... 实现细节

    return acceleration
```

### 添加新球类运动

```python
# src/sports/tennis.py
from ..ball_types import BallState, Table
from ..constants import TENNIS_BALL_RADIUS, TENNIS_BALL_MASS

class TennisBall:
    def __init__(self):
        self.radius = TENNIS_BALL_RADIUS
        self.mass = TENNIS_BALL_MASS

    def simulate_rally(self, initial_state: BallState) -> SimulationResult:
        # 网球特有的仿真逻辑
        pass
```

### 自定义可视化

```python
# src/visualization/advanced_plotting.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_velocity_field(result: SimulationResult):
    """绘制速度场可视化"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 实现速度向量场绘制
    # ...

    return fig
```

## 版本控制

### Git 工作流

```bash
# 创建功能分支
git checkout -b feature/new-physics-model

# 提交原子化更改
git add -p
git commit -m "feat: add advanced aerodynamic model

- Implement temperature-dependent air density
- Add humidity effects on drag
- Update tests for new model"

# 推送到远程
git push origin feature/new-physics-model

# 创建Pull Request
```

### 提交信息规范

```
type(scope): description

[body]

[footer]
```

类型包括：
- `feat`: 新功能
- `fix`: 修复
- `docs`: 文档更新
- `style`: 代码格式
- `refactor`: 重构
- `test`: 测试
- `chore`: 杂项

## 发布流程

### 版本号管理

遵循语义化版本 (Semantic Versioning):

- **MAJOR**: 不兼容的API变更
- **MINOR**: 向后兼容的新功能
- **PATCH**: 向后兼容的修复

### 发布步骤

1. **更新版本号**
   ```python
   # setup.py 或 pyproject.toml
   version = "1.2.0"
   ```

2. **更新变更日志**
   ```
   # CHANGELOG.md
   ## [1.2.0] - 2024-01-15
   ### Added
   - Advanced aerodynamic model
   - Tennis simulation support

   ### Fixed
   - Collision detection precision
   ```

3. **创建发布标签**
   ```bash
   git tag -a v1.2.0 -m "Release version 1.2.0"
   git push origin v1.2.0
   ```

4. **构建分发包**
   ```bash
   python setup.py sdist bdist_wheel
   twine upload dist/*
   ```

## 质量保证

### 代码审查清单

- [ ] 类型提示完整
- [ ] 文档字符串规范
- [ ] 单元测试覆盖
- [ ] 性能测试通过
- [ ] 向后兼容性
- [ ] 代码格式检查

### 持续集成

推荐的CI配置 (.github/workflows/ci.yml):

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest --cov=src --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## 故障排除

### 常见开发问题

1. **导入错误**
   ```python
   # 错误
   from ..physics import rk4_step

   # 正确 (添加__init__.py)
   from src.physics import rk4_step
   ```

2. **类型检查失败**
   ```bash
   mypy src/ --ignore-missing-imports
   ```

3. **性能问题**
   ```bash
   python -m cProfile -s cumulative script.py
   ```

### 调试技巧

1. **启用详细日志**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **断点调试**
   ```python
   # 在关键位置添加断点
   import pdb; pdb.set_trace()
   ```

3. **数值稳定性检查**
   ```python
   # 检查数值是否合理
   assert np.all(np.isfinite(velocity)), "Velocity contains NaN or Inf"
   ```
