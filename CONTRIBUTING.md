# Contributing to MetaFed-FL

Thank you for your interest in contributing to MetaFed-FL! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/MetaFed-FL.git
   ```
3. Create a virtual environment:
   ```bash
   python -m venv metafed-env
   source metafed-env/bin/activate  # On Windows: metafed-env\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
5. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## How to Contribute

### Reporting Bugs

Before submitting a bug report, please check if the issue has already been reported. If not, create a new issue with:

- A clear and descriptive title
- A detailed description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- Environment details (OS, Python version, package versions)

### Suggesting Enhancements

To suggest a new feature or enhancement:

1. Check if there's already an open issue or pull request for the feature
2. Create a new issue with:
   - A clear and descriptive title
   - A detailed explanation of the proposed feature
   - Use cases and benefits
   - Potential implementation approaches (if applicable)

### Code Contributions

1. Choose an issue to work on or create a new one
2. Comment on the issue to indicate you're working on it
3. Create a new branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Make your changes following the coding standards
5. Add tests if applicable
6. Update documentation as needed
7. Commit your changes with a clear message
8. Push your branch and create a pull request

## Development Workflow

### Branch Naming

- `feature/feature-name` for new features
- `bugfix/issue-name` for bug fixes
- `docs/documentation-topic` for documentation improvements
- `refactor/component-name` for refactoring work

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification:

```
type(scope): description

body (optional)

footer (optional)
```

Types include:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality. Install them with:

```bash
pre-commit install
```

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints where possible
- Maintain consistent naming conventions:
  - `snake_case` for variables and functions
  - `PascalCase` for classes
  - `UPPER_CASE` for constants

### Code Organization

- Follow the existing project structure
- Keep modules focused and cohesive
- Use meaningful variable and function names
- Add docstrings to all public functions, classes, and modules
- Keep functions small and focused on a single task

### Dependencies

- Only add dependencies that are necessary
- Specify version ranges in `requirements.txt` and `setup.py`
- Update `requirements-dev.txt` for development dependencies

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/metafed

# Run specific test file
pytest tests/unit/test_client.py
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test function names
- Follow the Arrange-Act-Assert pattern
- Test edge cases and error conditions
- Use pytest fixtures where appropriate

## Documentation

### Docstrings

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings):

```python
def example_function(param1: int, param2: str) -> bool:
    """Example function with types documented in the docstring.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.
    """
```

### README Updates

Update the README.md when:
- Adding new features
- Changing installation instructions
- Modifying usage examples
- Updating project badges or links

## Pull Request Process

1. Ensure your code follows the coding standards
2. Add tests for new functionality
3. Update documentation as needed
4. Run all tests to ensure nothing is broken
5. Squash related commits for clarity
6. Submit a pull request with:
   - A clear title
   - Detailed description of changes
   - Reference to related issues
   - Screenshots or examples if applicable

### Pull Request Review

- All pull requests require review from maintainers
- Address all review comments
- Be responsive to feedback
- Be patient during the review process

## Community

### Communication

- Join our [Discussions](https://github.com/afrilab/MetaFed-FL/discussions) for general questions and community support
- Check [Issues](https://github.com/afrilab/MetaFed-FL/issues) for bug reports and feature requests

### Recognition

Contributors will be recognized in:
- Release notes
- Contributor list in documentation
- Project README (for significant contributions)

Thank you for contributing to MetaFed-FL!