# 贡献指南

感谢您对乒乓球仿真系统的兴趣！我们欢迎各种形式的贡献，无论是修复bug、添加新功能、改进文档还是报告问题。

## 快速开始

### 开发环境设置

1. **Fork 项目** 到您的GitHub账户
2. **克隆到本地**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pingpong-simulation.git
   cd pingpong-simulation
   ```
3. **创建虚拟环境**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate     # Windows
   ```
4. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]  # 安装开发依赖
   ```
5. **运行测试** 确保一切正常:
   ```bash
   pytest
   ```

### 创建功能分支

```bash
# 创建并切换到新分支
git checkout -b feature/amazing-new-feature

# 或者修复bug
git checkout -b fix/bug-description
```

## 贡献类型

### 🐛 报告 Bug

如果您发现bug，请：

1. **检查现有问题** 是否已被报告
2. **创建新问题**，包含：
   - 清晰的标题
   - 详细的描述
   - 重现步骤
   - 期望行为 vs 实际行为
   - 系统信息 (Python版本、操作系统等)

### ✨ 新功能请求

对于新功能建议，请：

1. **检查现有问题** 是否已有类似请求
2. **详细描述** 功能需求
3. **解释为什么** 这个功能有价值
4. **考虑替代方案**

### 🔧 代码贡献

#### 代码规范

我们遵循以下代码规范：

- **PEP 8** 代码风格
- **Google风格** 文档字符串
- **类型提示** 必须完整
- **Black** 代码格式化

#### 提交信息格式

```
type(scope): description

[详细描述]

[相关问题引用]
```

类型包括：
- `feat`: 新功能
- `fix`: 修复
- `docs`: 文档
- `style`: 格式调整
- `refactor`: 重构
- `test`: 测试
- `chore`: 杂项

示例：
```
feat(physics): add advanced aerodynamic model

Implement temperature-dependent air density calculation
and humidity effects on drag coefficient.

Closes #42
```

#### 测试要求

所有新代码必须包含相应的测试：

- **单元测试** 覆盖核心功能
- **集成测试** 验证模块间交互
- **性能测试** 确保没有性能回归

```bash
# 运行测试
pytest

# 带覆盖率
pytest --cov=src --cov-report=html
```

### 📚 文档贡献

- 修复文档错误
- 添加使用示例
- 改进API文档
- 翻译文档

### 🎨 设计贡献

- UI/UX 改进
- 可视化增强
- 图表美化

## 开发工作流

### 1. 选择任务

- 查看 [Issues](../../issues) 页面
- 选择适合您的任务
- 添加注释表明您正在处理

### 2. 实现功能

```bash
# 创建功能分支
git checkout -b feature/new-feature

# 实现功能
# ... 编写代码 ...

# 添加测试
# ... 编写测试 ...

# 运行测试
pytest

# 格式化代码
black src/
```

### 3. 提交更改

```bash
# 添加更改
git add .

# 提交 (使用规范的提交信息)
git commit -m "feat: add amazing new feature

Detailed description of what was implemented
and why it's useful.

Closes #123"
```

### 4. 创建 Pull Request

1. **推送分支** 到您的fork
   ```bash
   git push origin feature/new-feature
   ```

2. **创建PR** 从您的分支到主分支
   - 清晰描述更改
   - 引用相关issues
   - 请求审查

3. **等待审查**
   - 响应审查意见
   - 进行必要的修改

## 代码审查指南

### 审查者职责

- **功能正确性** 检查代码是否正确实现需求
- **代码质量** 确保遵循编码规范
- **测试覆盖** 验证有足够测试
- **性能影响** 检查是否有性能问题
- **向后兼容** 确保不破坏现有功能

### 被审查者职责

- **及时响应** 审查意见
- **解释决策** 为什么这样实现
- **接受建议** 虚心学习
- **改进代码** 根据建议优化

## 行为准则

### 我们的承诺

我们致力于为所有人提供一个无骚扰的合作环境，无论年龄、体型、残疾、民族、性别认同和表达、经验水平、国籍、外貌、人种、宗教或性认同和取向。

### 标准

有助于创造积极环境的行为包括：

- 使用欢迎和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同情

不可接受的行为包括：

- 使用性化语言或图像以及不受欢迎的性关注
-  trolling、侮辱性/贬损性评论
- 公开或私人骚扰
- 未经明确许可发布他人的私人信息
- 其他在专业环境中不适当的行为

## 许可证

通过贡献代码，您同意您的贡献将根据项目的许可证进行许可。

## 获得帮助

如果您需要帮助：

- 📧 **邮箱**: 发送邮件到维护者
- 💬 **讨论**: 使用GitHub Discussions
- 🐛 **问题**: 创建GitHub Issue
- 📖 **文档**: 查看详细文档

感谢您的贡献！🎾
