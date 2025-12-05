# PingPong Simulation

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](doc/)

基于物理原理的3D乒乓球运动轨迹仿真系统，实现完整的空气动力学、碰撞检测和球拍交互。

## ✨ 核心特性

- **🔬 真实物理模型**: 空气阻力、马格努斯效应、重力、碰撞动力学
- **🎯 精确仿真**: 基于ITTF标准参数，支持多种击球技术
- **📊 完整数据输出**: CSV轨迹数据 + 3D可视化动画
- **🎮 双人交互**: 智能球拍AI，支持多回合对打
- **🔧 模块化设计**: 易于扩展到其他球类运动

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```bash
# 发球仿真
python pingpong_main.py --serve-mode fh_under

# 自定义击球
python pingpong_main.py --serve-mode custom --pos -1.2 0 0.9 --vel 8 0 3 --omega 0 150 0

# 轨迹分析
python analyze_impact.py --speed 5.0 --angle 30.0 --spin 0 150 0
```

## 📖 详细文档

- [**用户指南**](doc/user_guide.md) - 完整使用教程和参数说明
- [**API参考**](doc/api_reference.md) - 详细的函数和类文档
- [**架构文档**](doc/architecture.md) - 系统设计和模块说明
- [**物理模型**](doc/physics_model.md) - 数学模型和公式推导
- [**开发指南**](doc/developer_guide.md) - 贡献代码和开发规范

## 🎯 支持的仿真场景

| 场景 | 描述 |
|------|------|
| `fh_under` | 正手underspin发球 |
| `fast_long` | 快速长球发球 |
| `custom` | 自定义初始条件 |

## 📊 输出格式

- **轨迹数据**: CSV格式 (位置、速度、角速度、事件)
- **可视化**: 3D动画 (MP4) + 静态图表
- **分析报告**: 物理参数统计和性能指标

## 🔧 技术栈

- **语言**: Python 3.8+
- **核心库**: NumPy, Matplotlib, Pandas
- **算法**: RK4数值积分、脉冲-动量碰撞模型
- **架构**: 模块化设计，类型提示完整

## 🤝 贡献

欢迎贡献代码！请查看：

- [贡献指南](CONTRIBUTING.md) - 如何参与项目开发
- [开发指南](doc/developer_guide.md) - 代码规范和最佳实践
- [Issues](../../issues) - 报告问题或建议新功能

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

## 🙏 致谢

- 基于ITTF官方标准参数
- 物理模型参考公开科学文献
- 感谢开源社区的支持

---

⭐ 如果这个项目对你有帮助，请给它一个星标！
