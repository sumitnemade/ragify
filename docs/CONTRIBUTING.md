# Contributing to Ragify

Thank you for your interest in contributing to Ragify! This document provides comprehensive guidelines and information for contributors to help build the future of intelligent context orchestration.

## 🎯 What We're Building

Ragify is a Python framework that combines data from multiple sources (docs, APIs, databases, real-time) and resolves conflicts. Built specifically for **LLM-powered applications** that need accurate, current information.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- pip
- Basic understanding of RAG systems and LLM applications

### Development Setup

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub first, then:
   git clone https://github.com/your-username/ragify.git
   cd ragify
   
   # Add upstream remote
   git remote add upstream https://github.com/sumitnemade/ragify.git
   ```

2. **Install Development Dependencies**
   ```bash
   # Install in development mode
   pip install -e ".[dev]"
   
   # Or install manually if dev extras aren't configured
   pip install -r requirements-dev.txt
   pip install -e .
   ```

3. **Set Python Path**
   ```bash
   # Add src to Python path for development
   export PYTHONPATH=src:$PYTHONPATH
   ```

4. **Run Tests**
   ```bash
   pytest
   ```

## 🧪 Testing Your Setup

### Test the RAG Chat Assistant Example
One of the best ways to understand Ragify is to run our featured example:

```bash
cd examples/rag_chat_assistant
python test_setup.py
streamlit run rag_chat_assistant.py
```

This will help you:
- Verify your development environment is working
- See Ragify in action with a real application
- Understand how the framework components work together

## 📝 Code Style & Quality

We use the following tools to maintain code quality:

- **Black**: Code formatting (line length: 88 characters)
- **isort**: Import sorting and organization
- **flake8**: Linting and style checking
- **mypy**: Type checking and validation
- **pytest**: Testing framework

### Pre-commit Setup

Install pre-commit hooks for automatic code quality checks:
```bash
pip install pre-commit
pre-commit install
```

This will automatically:
- Format your code with Black
- Sort imports with isort
- Check for style issues with flake8
- Validate types with mypy
- Run tests before commits

### Code Style Guidelines

- **Type Hints**: Use type hints for all function parameters and return values
- **Docstrings**: Follow Google-style docstrings for all public functions
- **Naming**: Use descriptive names, follow Python naming conventions
- **Error Handling**: Use custom exceptions from `ragify.exceptions`
- **Async**: Use async/await for I/O operations

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ragify --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run with verbose output
pytest -v

# Run tests in parallel
pytest -n auto

# Run only fast tests
pytest -m "not slow"
```

### Writing Tests
- Place tests in the `tests/` directory
- Use descriptive test names that explain the scenario
- Include both unit and integration tests
- Aim for high test coverage (>90%)
- Use fixtures for common test data
- Mock external dependencies

### Test Structure Example
```python
import pytest
from ragify import ContextOrchestrator

class TestContextOrchestrator:
    """Test suite for ContextOrchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a test orchestrator instance."""
        return ContextOrchestrator(
            vector_db_url="memory://",
            privacy_level="private"
        )
    
    def test_initialization(self, orchestrator):
        """Test orchestrator initializes correctly."""
        assert orchestrator is not None
        assert orchestrator.privacy_level == "private"
    
    @pytest.mark.asyncio
    async def test_add_source(self, orchestrator):
        """Test adding a data source."""
        # Test implementation
        pass
```

## 📚 Documentation

### Building Documentation
```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build HTML documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

### Documentation Guidelines
- Keep documentation up to date with code changes
- Include practical examples for new features
- Use clear and concise language
- Add type hints to all functions
- Include code examples in docstrings
- Update README.md for significant changes

### Documentation Structure
- **README.md**: Project overview and quick start
- **docs/**: Comprehensive documentation
- **examples/**: Working examples and demos
- **docstrings**: Inline code documentation

## 🔧 Development Workflow

### 1. Stay Updated
```bash
# Fetch latest changes from upstream
git fetch upstream
git checkout main
git merge upstream/main
```

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 3. Make Changes
- Write your code following style guidelines
- Add comprehensive tests
- Update relevant documentation
- Ensure all tests pass

### 4. Commit Your Changes
```bash
# Stage changes
git add .

# Commit with conventional commit message
git commit -m "feat: add intelligent conflict resolution"
git commit -m "fix: resolve vector database connection issue"
git commit -m "docs: update API reference with examples"
```

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

## 📋 Pull Request Guidelines

### Before Submitting
- [ ] Code follows style guidelines (Black, isort, flake8, mypy)
- [ ] All tests pass (unit, integration, and manual)
- [ ] Documentation is updated
- [ ] New features have comprehensive tests
- [ ] Breaking changes are documented
- [ ] Code is self-reviewed

### Pull Request Template
```markdown
## Description
Brief description of changes and why they're needed

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring (no functional changes)

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] RAG Chat Assistant example still works (if applicable)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or properly documented)
- [ ] Added tests for new functionality
- [ ] All CI checks pass

## Related Issues
Closes #(issue number)
```

## 🐛 Reporting Bugs

### Bug Report Template
```markdown
## Bug Description
Clear description of the bug and expected behavior

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens (include error messages)

## Environment
- OS: [e.g., Ubuntu 20.04, macOS 12.0, Windows 11]
- Python Version: [e.g., 3.9.7, 3.10.4]
- Ragify Version: [e.g., 0.1.0, commit hash]
- Dependencies: [relevant package versions]

## Additional Information
- Screenshots if applicable
- Logs or error traces
- Any workarounds you've found
```

## 💡 Feature Requests

### Feature Request Template
```markdown
## Feature Description
Clear description of the requested feature

## Use Case
Why this feature is needed and how it would be used

## Proposed Solution
How you think it should be implemented

## Alternatives Considered
Other approaches you've considered

## Impact
How this feature would benefit the Ragify ecosystem
```

## 🏗️ Project Structure

Understanding the project structure helps with contributions:

```
ragify/
├── src/ragify/           # Core framework code
│   ├── core.py          # Main orchestrator
│   ├── engines/         # Core engines (fusion, scoring, storage)
│   ├── sources/         # Data source implementations
│   ├── storage/         # Storage backends
│   └── models.py        # Data models and types
├── examples/            # Working examples
│   └── rag_chat_assistant/  # Featured RAG example
├── tests/               # Test suite
├── docs/                # Documentation
└── requirements*.txt    # Dependencies
```

## 🤝 Community Guidelines

### Code of Conduct
- Be respectful and inclusive to all contributors
- Help others learn and grow
- Provide constructive feedback
- Follow the project's coding standards
- Welcome newcomers and help them get started

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community chat
- **Pull Requests**: Code contributions and reviews
- **Documentation**: Help improve guides and examples

### Getting Help
- Check existing issues and discussions first
- Be specific about your problem or question
- Include relevant code snippets and error messages
- Be patient and respectful

## 🚀 Advanced Development

### Working with Examples
- Test your changes against the RAG Chat Assistant example
- Ensure examples still work after modifications
- Add new examples for new features
- Update example documentation

### Performance Considerations
- Profile code for bottlenecks
- Use async operations for I/O
- Implement caching where appropriate
- Consider memory usage for large datasets

### Security & Privacy
- Follow security best practices
- Test privacy controls thoroughly
- Validate input data
- Handle sensitive information appropriately

## 📄 License

By contributing to Ragify, you agree that your contributions will be licensed under the MIT License.

## 🙏 Acknowledgments

Thank you for contributing to Ragify! Your contributions help make this project better for everyone in the LLM and RAG community.

### Special Thanks To
- Contributors who help with code, documentation, and examples
- Users who provide feedback and bug reports
- The open-source community for inspiration and tools

---

**Ready to contribute? Start by exploring the [RAG Chat Assistant example](examples/rag_chat_assistant/) to see Ragify in action!**
