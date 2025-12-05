# Developer Guide

## Project Structure

```
PingPong Simulation System/
├── doc/                    # Documentation directory
│   ├── architecture.md     # Architecture documentation
│   ├── physics_model.md    # Physics model details
│   ├── user_guide.md       # User guide
│   ├── api_reference.md    # API reference
│   └── developer_guide.md  # Developer guide
├── src/                    # Source code directory (recommended)
│   ├── __init__.py
│   ├── constants.py
│   ├── ball_types.py
│   ├── physics.py
│   ├── simulation.py
│   ├── racket_control.py
│   ├── scenarios.py
│   └── visualization.py
├── tests/                  # Test directory
│   ├── test_physics.py
│   ├── test_simulation.py
│   └── conftest.py
├── examples/               # Example scripts
│   ├── basic_simulation.py
│   └── parameter_study.py
├── pingpong_main.py        # Main entry script
├── analyze_impact.py       # Analysis tool
├── requirements.txt        # Dependency list
├── setup.py               # Installation script
├── LICENSE                # License
├── README.md              # Project description
├── CONTRIBUTING.md        # Contribution guide
└── .gitignore            # Ignore file
```

## Development Environment Setup

### Environment Requirements

- Python 3.8+
- pip or conda package manager

### Install Development Dependencies

```bash
# Clone project
git clone https://github.com/username/pingpong-simulation.git
cd pingpong-simulation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e .[dev]
```

### Development Tool Configuration

Recommended tools:

- **Code Editor**: VSCode, PyCharm
- **Version Control**: Git
- **Code Formatting**: Black
- **Code Checking**: Flake8, MyPy
- **Testing Framework**: pytest
- **Documentation Generation**: Sphinx

## Code Standards

### Naming Conventions

- **Variables and functions**: snake_case
- **Classes**: PascalCase
- **Constants**: UPPER_CASE
- **Modules**: snake_case

### Type Hints

All new code must include complete type hints:

```python
from typing import List, Optional, Tuple
import numpy as np

def calculate_trajectory(
    initial_pos: np.ndarray,
    initial_vel: np.ndarray,
    time_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate trajectory"""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def aerodynamic_acceleration(
    velocity: np.ndarray,
    omega: np.ndarray
) -> np.ndarray:
    """Calculate aerodynamic acceleration.

    Calculate ball acceleration based on drag laws and Magnus effect.

    Args:
        velocity: Ball velocity vector (m/s)
        omega: Ball angular velocity vector (rad/s)

    Returns:
        Acceleration vector (m/s²)

    Examples:
        >>> vel = np.array([5.0, 0.0, 2.0])
        >>> omega = np.array([0.0, 100.0, 0.0])
        >>> acc = aerodynamic_acceleration(vel, omega)
        >>> acc.shape
        (3,)
    """
```

## Test Development

### Test Structure

```python
# tests/test_physics.py
import numpy as np
import pytest
from src.physics import aerodynamic_acceleration

class TestAerodynamicAcceleration:
    def test_zero_velocity(self):
        """Test acceleration with zero velocity"""
        vel = np.zeros(3)
        omega = np.zeros(3)
        acc = aerodynamic_acceleration(vel, omega)

        # Should only have gravity
        expected = np.array([0.0, 0.0, -9.81])
        np.testing.assert_array_almost_equal(acc, expected)

    def test_drag_force(self):
        """Test drag force effect"""
        vel = np.array([10.0, 0.0, 0.0])  # 10 m/s horizontal velocity
        omega = np.zeros(3)
        acc = aerodynamic_acceleration(vel, omega)

        # Should have drag component (negative x direction)
        assert acc[0] < 0.0  # Drag deceleration
        assert acc[2] == -9.81  # Gravity unchanged
```

### Run Tests

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_physics.py::TestAerodynamicAcceleration::test_zero_velocity

# With coverage report
pytest --cov=src --cov-report=html
```

### Testing Best Practices

1. **Unit tests**: Test each function independently
2. **Integration tests**: Test inter-module interactions
3. **Performance tests**: Validate large-scale simulation performance
4. **Physics validation**: Compare with analytical solutions

## Performance Optimization

### Analyzing Performance Bottlenecks

```python
import cProfile
import pstats

# Performance profiling
profiler = cProfile.Profile()
profiler.enable()

# Run simulation
result = simulate(...)

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(20)  # Show top 20 time-consuming functions
```

### Optimization Strategies

1. **Vectorized Computation**: Use NumPy array operations
2. **Reduce Memory Allocation**: Reuse array objects
3. **Adaptive Time Stepping**: Adjust dt based on motion intensity
4. **Event-Driven**: Avoid unnecessary calculations

### Memory Optimization

```python
# Use memory-mapped arrays for large data
import numpy as np

def save_large_trajectory(filename: str, trajectory: np.ndarray):
    """Efficiently save large trajectory data"""
    fp = np.memmap(filename, dtype='float64',
                   mode='w+', shape=trajectory.shape)
    fp[:] = trajectory[:]
    fp.flush()
```

## Extension Development

### Adding New Physics Models

```python
# src/physics.py
def advanced_aerodynamic_model(
    velocity: np.ndarray,
    omega: np.ndarray,
    temperature: float = 20.0,
    humidity: float = 0.5
) -> np.ndarray:
    """Advanced aerodynamic model considering temperature and humidity"""
    # Air density changes with temperature
    rho = 1.225 * (273.15 / (temperature + 273.15))

    # Humidity effects
    # ... implementation details

    return acceleration
```

### Adding New Ball Sports

```python
# src/sports/tennis.py
from ..ball_types import BallState, Table
from ..constants import TENNIS_BALL_RADIUS, TENNIS_BALL_MASS

class TennisBall:
    def __init__(self):
        self.radius = TENNIS_BALL_RADIUS
        self.mass = TENNIS_BALL_MASS

    def simulate_rally(self, initial_state: BallState) -> SimulationResult:
        # Tennis-specific simulation logic
        pass
```

### Custom Visualization

```python
# src/visualization/advanced_plotting.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_velocity_field(result: SimulationResult):
    """Plot velocity field visualization"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Implement velocity vector field plotting
    # ...

    return fig
```

## Version Control

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-physics-model

# Commit atomic changes
git add -p
git commit -m "feat: add advanced aerodynamic model

- Implement temperature-dependent air density
- Add humidity effects on drag
- Update tests for new model"

# Push to remote
git push origin feature/new-physics-model

# Create Pull Request
```

### Commit Message Standards

```
type(scope): description

[body]

[footer]
```

Types include:
- `feat`: new feature
- `fix`: bug fix
- `docs`: documentation update
- `style`: code formatting
- `refactor`: refactoring
- `test`: testing
- `chore`: miscellaneous

## Release Process

### Version Number Management

Follow Semantic Versioning:

- **MAJOR**: Breaking API changes
- **MINOR**: Backward-compatible new features
- **PATCH**: Backward-compatible bug fixes

### Release Steps

1. **Update version number**
   ```python
   # setup.py or pyproject.toml
   version = "1.2.0"
   ```

2. **Update changelog**
   ```
   # CHANGELOG.md
   ## [1.2.0] - 2024-01-15
   ### Added
   - Advanced aerodynamic model
   - Tennis simulation support

   ### Fixed
   - Collision detection precision
   ```

3. **Create release tag**
   ```bash
   git tag -a v1.2.0 -m "Release version 1.2.0"
   git push origin v1.2.0
   ```

4. **Build distribution packages**
   ```bash
   python setup.py sdist bdist_wheel
   twine upload dist/*
   ```

## Quality Assurance

### Code Review Checklist

- [ ] Type hints complete
- [ ] Docstrings standardized
- [ ] Unit test coverage
- [ ] Performance tests pass
- [ ] Backward compatibility
- [ ] Code formatting check

### Continuous Integration

Recommended CI configuration (.github/workflows/ci.yml):

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

## Troubleshooting

### Common Development Issues

1. **Import errors**
   ```python
   # Wrong
   from ..physics import rk4_step

   # Correct (add __init__.py)
   from src.physics import rk4_step
   ```

2. **Type checking failures**
   ```bash
   mypy src/ --ignore-missing-imports
   ```

3. **Performance issues**
   ```bash
   python -m cProfile -s cumulative script.py
   ```

### Debugging Tips

1. **Enable verbose logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Breakpoint debugging**
   ```python
   # Add breakpoint at critical locations
   import pdb; pdb.set_trace()
   ```

3. **Numerical stability checks**
   ```python
   # Check if values are reasonable
   assert np.all(np.isfinite(velocity)), "Velocity contains NaN or Inf"
   ```
