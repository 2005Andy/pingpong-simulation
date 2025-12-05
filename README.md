# PingPong Simulation

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](doc/)
[![codecov](https://codecov.io/gh/2005Andy/pingpong-simulation/branch/master/graph/badge.svg)](https://codecov.io/gh/2005Andy/pingpong-simulation)

Physics-based 3D ping-pong ball trajectory simulation system, implementing complete aerodynamics, collision detection, and racket interaction.

## ✨ Core Features

- **🔬 Realistic Physics Model**: Air drag, Magnus effect, gravity, collision dynamics
- **🎯 Precise Simulation**: Based on ITTF standard parameters, supports various stroke techniques
- **📊 Complete Data Output**: CSV trajectory data + 3D visualization animations
- **🎮 Dual Player Interaction**: Intelligent racket AI, supports multi-rally gameplay
- **🔧 Modular Design**: Easy to extend to other ball sports

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Serve simulation
python pingpong_main.py --serve-mode fh_under

# Custom stroke
python pingpong_main.py --serve-mode custom --pos -1.2 0 0.9 --vel 8 0 3 --omega 0 150 0

# Trajectory analysis
python analyze_impact.py --speed 5.0 --angle 30.0 --spin 0 150 0
```

## 📖 Detailed Documentation

- [**User Guide**](doc/user_guide.md) - Complete usage tutorial and parameter description
- [**API Reference**](doc/api_reference.md) - Detailed function and class documentation
- [**Architecture Document**](doc/architecture.md) - System design and module description
- [**Physics Model**](doc/physics_model.md) - Mathematical model and formula derivation
- [**Developer Guide**](doc/developer_guide.md) - Code contribution and development standards

## 🎯 Supported Simulation Scenarios

| Scenario | Description |
|---------|-------------|
| `fh_under` | Forehand underspin serve |
| `fast_long` | Fast long serve |
| `custom` | Custom initial conditions |

## 📊 Output Formats

- **Trajectory Data**: CSV format (position, velocity, angular velocity, events)
- **Visualization**: 3D animations (MP4) + static plots
- **Analysis Reports**: Physics parameter statistics and performance metrics

## 🔧 Technology Stack

- **Language**: Python 3.8+
- **Core Libraries**: NumPy, Matplotlib, Pandas
- **Algorithms**: RK4 numerical integration, impulse-momentum collision model
- **Architecture**: Modular design with complete type hints

## 🤝 Contributing

Welcome to contribute code! Please check:

- [Contribution Guide](CONTRIBUTING.md) - How to participate in project development
- [Developer Guide](doc/developer_guide.md) - Code standards and best practices
- [Code of Conduct](CODE_OF_CONDUCT.md) - Community standards and expectations
- [Security Policy](SECURITY.md) - How to report security vulnerabilities
- [Issues](../../issues) - Report issues or suggest new features

## 📄 License

This project uses the [MIT License](LICENSE) open source license.

## 🙏 Acknowledgments

- Based on ITTF official standard parameters
- Physics model references published scientific literature
- Thanks to the open source community

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=2005Andy/pingpong-simulation&type=date&legend=top-left)](https://www.star-history.com/#2005Andy/pingpong-simulation&type=date&legend=top-left)

---

⭐ If this project is helpful to you, please give it a star!

