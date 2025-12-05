# Contribution Guide

Thank you for your interest in the PingPong simulation system! We welcome all forms of contributions, whether fixing bugs, adding new features, improving documentation, or reporting issues.

## Quick Start

### Development Environment Setup

1. **Fork the project** to your GitHub account
2. **Clone locally**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pingpong-simulation.git
   cd pingpong-simulation
   ```
3. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]  # Install development dependencies
   ```
5. **Run tests** to ensure everything works:
   ```bash
   pytest
   ```

### Create Feature Branch

```bash
# Create and switch to new branch
git checkout -b feature/amazing-new-feature

# or fix bug
git checkout -b fix/bug-description
```

## Contribution Types

### 🐛 Report Bug

If you find a bug, please:

1. **Check existing issues** to see if it has been reported
2. **Create a new issue** with:
   - Clear title
   - Detailed description
   - Reproduction steps
   - Expected behavior vs actual behavior
   - System information (Python version, OS, etc.)

### ✨ Feature Request

For new feature suggestions, please:

1. **Check existing issues** for similar requests
2. **Describe in detail** the feature requirements
3. **Explain why** this feature would be valuable
4. **Consider alternatives**

### 🔧 Code Contribution

#### Code Standards

We follow these coding standards:

- **PEP 8** code style
- **Google-style** docstrings
- **Type hints** must be complete
- **Black** code formatting

#### Commit Message Format

```
type(scope): description

[detailed description]

[issue references]
```

Types include:
- `feat`: new feature
- `fix`: bug fix
- `docs`: documentation
- `style`: formatting
- `refactor`: refactoring
- `test`: testing
- `chore`: miscellaneous

Example:
```
feat(physics): add advanced aerodynamic model

Implement temperature-dependent air density calculation
and humidity effects on drag coefficient.

Closes #42
```

#### Testing Requirements

All new code must include corresponding tests:

- **Unit tests** covering core functionality
- **Integration tests** validating inter-module interactions
- **Performance tests** ensuring no performance regressions

```bash
# Run tests
pytest

# With coverage
pytest --cov=src --cov-report=html
```

### 📚 Documentation Contribution

- Fix documentation errors
- Add usage examples
- Improve API documentation
- Translate documentation

### 🎨 Design Contribution

- UI/UX improvements
- Visualization enhancements
- Chart beautification

## Development Workflow

### 1. Choose Task

- View [Issues](../../issues) page
- Choose a task that suits you
- Add comment indicating you're working on it

### 2. Implement Feature

```bash
# Create feature branch
git checkout -b feature/new-feature

# Implement feature
# ... write code ...

# Add tests
# ... write tests ...

# Run tests
pytest

# Format code
black src/
```

### 3. Commit Changes

```bash
# Add changes
git add .

# Commit (use standard commit message)
git commit -m "feat: add amazing new feature

Detailed description of what was implemented
and why it's useful.

Closes #123"
```

### 4. Create Pull Request

1. **Push branch** to your fork
   ```bash
   git push origin feature/new-feature
   ```

2. **Create PR** from your branch to main branch
   - Clearly describe changes
   - Reference related issues
   - Request review

3. **Wait for review**
   - Respond to review comments
   - Make necessary modifications

## Code Review Guidelines

### Reviewer Responsibilities

- **Functional correctness**: Check if code correctly implements requirements
- **Code quality**: Ensure coding standards are followed
- **Test coverage**: Verify sufficient testing
- **Performance impact**: Check for performance issues
- **Backward compatibility**: Ensure no breaking changes

### Reviewee Responsibilities

- **Timely response** to review comments
- **Explain decisions**: Why implemented this way
- **Accept suggestions**: Learn humbly
- **Improve code**: Optimize based on suggestions

## Code of Conduct

### Our Commitment

We are committed to providing a harassment-free collaborative environment for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Standards

Behaviors that contribute to creating a positive environment include:

- Using welcoming and inclusive language
- Respecting different viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Unacceptable behaviors include:

- Use of sexualized language or imagery and unwelcome sexual attention
- Trolling, insulting/derogatory comments
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

## License

By contributing code, you agree that your contributions will be licensed under the project's license.

## Get Help

If you need help:

- 📧 **Email**: Send email to maintainers
- 💬 **Discussions**: Use GitHub Discussions
- 🐛 **Issues**: Create GitHub Issue
- 📖 **Documentation**: View detailed documentation

Thank you for your contributions! 🎾
