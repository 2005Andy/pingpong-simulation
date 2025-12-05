# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-XX

### Added
- 🎯 **完整3D乒乓球仿真系统**
  - 基于物理原理的球体运动轨迹计算
  - 空气动力学模型（阻力 + 马格努斯效应）
  - 精确的碰撞检测和响应

- 🏓 **真实球桌环境**
  - ITTF标准尺寸球桌和球网
  - 多种材质的物理属性（正胶、生胶、防弧胶）
  - 双侧球拍交互系统

- 📊 **丰富的数据输出**
  - CSV格式轨迹数据导出
  - 3D可视化动画生成
  - 轨迹分析工具

- 🎮 **智能仿真场景**
  - 预定义发球模式（fh_under, fast_long）
  - 自定义初始条件支持
  - 多回合对打模拟

- 📚 **完整文档系统**
  - 用户指南和API参考
  - 架构设计文档
  - 物理模型详解
  - 开发贡献指南

- 🔧 **模块化架构**
  - physics.py - 物理计算引擎
  - simulation.py - 仿真控制引擎
  - racket_control.py - 球拍控制逻辑
  - visualization.py - 可视化引擎

### Technical Details
- **数值方法**: 四阶龙格-库塔积分
- **碰撞模型**: 脉冲-动量法 + Coulomb摩擦
- **坐标系**: 右手笛卡尔坐标系，Z轴向上
- **单位**: 国际单位制 (SI)
- **时间精度**: 50微秒时间步长

### Dependencies
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.5.0

---

## [0.1.0] - 2024-XX-XX

### Added
- 初始单文件实现版本
- 基本物理模型和碰撞检测
- 简单可视化输出

### Known Issues
- 代码结构待优化
- 缺少完整测试覆盖
- 文档不完善

---

## Types of changes
- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` for vulnerability fixes
