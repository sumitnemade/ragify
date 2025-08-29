# Contributing to Ragify

Thank you for your interest in contributing to Ragify! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- pip

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/ragify.git
cd ragify
   ```

2. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Run Tests**
   ```bash
   pytest
   ```

## 📝 Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

### Pre-commit Setup

Install pre-commit hooks:
```bash
pre-commit install
```

This will automatically format and check your code before commits.

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ragify

# Run specific test file
pytest tests/test_core.py

# Run with verbose output
pytest -v
```

### Writing Tests
- Place tests in the `tests/` directory
- Use descriptive test names
- Include both unit and integration tests
- Aim for high test coverage

## 📚 Documentation

### Building Documentation
```bash
pip install -e ".[docs]"
cd docs
make html
```

### Documentation Guidelines
- Keep documentation up to date with code changes
- Include examples for new features
- Use clear and concise language
- Add type hints to all functions

## 🔧 Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Write your code
- Add tests
- Update documentation
- Follow the code style guidelines

### 3. Commit Your Changes
```bash
git add .
git commit -m "feat: add new feature description"
```

### 4. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

## 📋 Pull Request Guidelines

### Before Submitting
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation is updated
- [ ] New features have tests
- [ ] Breaking changes are documented

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## 🐛 Reporting Bugs

### Bug Report Template
```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Ubuntu 20.04]
- Python Version: [e.g., 3.9.7]
- Ragify Version: [e.g., 0.1.0]

## Additional Information
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template
```markdown
## Feature Description
Clear description of the requested feature

## Use Case
Why this feature is needed

## Proposed Solution
How you think it should be implemented

## Alternatives Considered
Other approaches you've considered
```

## 🤝 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Help others learn
- Provide constructive feedback
- Follow the project's coding standards

### Communication
- Use GitHub Issues for bug reports and feature requests
- Use GitHub Discussions for general questions
- Be clear and concise in your communication

## 📄 License

By contributing to Ragify, you agree that your contributions will be licensed under the MIT License.

## 🙏 Acknowledgments

Thank you for contributing to Ragify! Your contributions help make this project better for everyone.
