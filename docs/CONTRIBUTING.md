# Contributing to MUSE

We welcome contributions to the MUSE project! This document provides guidelines for contributing.

## Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest tests/`
6. Format code: `black muse_v3_advanced/`
7. Submit a pull request

## Code Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Add docstrings to all functions and classes
- Keep functions focused and small

## Testing

- Write unit tests for new functions
- Include integration tests for major features
- Ensure 80%+ code coverage
- Test both original MUSE and v3 components

## Documentation

- Update relevant documentation
- Add examples for new features
- Keep README.md current
- Document API changes

## Reporting Issues

- Use GitHub Issues for bug reports
- Include reproducible examples
- Specify environment details
- Check existing issues first
