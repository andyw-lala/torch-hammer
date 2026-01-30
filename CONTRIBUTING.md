# Contributing to Torch Hammer

Thank you for your interest in contributing to Torch Hammer! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Community Communication](#community-communication)
- [Developer Certificate of Origin](#developer-certificate-of-origin)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Other Ways to Contribute](#other-ways-to-contribute)
- [References](#references)

## Code of Conduct

This project adheres to our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [github@hpe.com](mailto:github@hpe.com).

## Community Communication

We encourage community participation and welcome your contributions!

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions, ideas, and general community discussion
- **Pull Requests**: For code contributions (see [Contributing Code](#contributing-code))

For general questions or to reach maintainers directly:
- **Email**: [github@hpe.com](mailto:github@hpe.com)

## Developer Certificate of Origin

Torch Hammer requires the Developer Certificate of Origin (DCO) process for all contributions. The DCO is a lightweight way for contributors to certify that they wrote or otherwise have the right to submit the code they are contributing.

### What is the DCO?

The DCO is a declaration that you have the right to contribute the code and that you agree to it being used under the project's license.

**Developer Certificate of Origin Version 1.1**

```
Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Read the full [DCO text](https://developercertificate.org/).

### How to Sign Off

Every commit must include a `Signed-off-by` line with your name and email:

```
Signed-off-by: Your Name <your.email@example.com>
```

**Using git:**
```bash
# Add sign-off to a single commit
git commit -s -m "Your commit message"

# Add sign-off to all commits interactively (if you forgot)
git rebase HEAD~n --signoff  # where n is the number of commits
```

**Note**: Use your real name (no pseudonyms) and a working email address.

### Fixing Unsigned Commits

If you've already made commits without signing:

```bash
# Amend the last commit
git commit --amend --signoff

# For multiple commits, use interactive rebase
git rebase -i HEAD~N  # where N is the number of commits
# Then amend each commit with --signoff

# Force push after fixing (only on your feature branch!)
git push --force-with-lease
```

### Why DCO?

The DCO provides a clear audit trail for contributions. It ensures that:
- You have the right to submit the code
- You understand the project's open source license
- The contribution is made in accordance with the license terms

## Getting Started

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/torch-hammer.git
   cd torch-hammer
   ```

3. **Set up the development environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   # For telemetry support:
   pip install nvidia-ml-py  # NVIDIA
   pip install amdsmi        # AMD
   # For testing:
   pip install pytest pytest-cov
   ```

4. **Configure git sign-off** (for DCO):
   ```bash
   git config user.name "Your Name"
   git config user.email "your.email@example.com"
   ```

## How to Contribute

### Reporting Bugs

Before creating a bug report:
- Check the existing issues to avoid duplicates
- Collect relevant information (hardware, OS, PyTorch version, error output)

When creating an issue:
- Use a clear, descriptive title
- Provide detailed steps to reproduce
- Include expected vs actual behavior
- Attach relevant logs (use `--verbose` flag)

### Suggesting Features

We welcome feature suggestions! Please:
- Check existing issues/discussions first
- Describe the use case and benefit
- Consider if it fits the project's scope (portable micro-benchmarks)

### Contributing Code

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our [Coding Standards](#coding-standards)

3. **Test your changes** (see [Testing](#testing))

4. **Commit with sign-off**:
   ```bash
   git commit -s -m "Add feature X"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**

## Pull Request Process

1. **Before submitting**:
   - Ensure all commits are signed (DCO)
   - Run the benchmarks to verify no regressions
   - Run the test suite: `pytest tests/`
   - Update documentation if needed
   - Keep PRs focused and reasonably sized

2. **PR Description should include**:
   - Clear description of what the PR does
   - Why the change is needed
   - How it was tested
   - Any breaking changes

3. **Review process**:
   - A maintainer will review your PR
   - Address any feedback
   - Once approved, a maintainer will merge

4. **After merging**:
   - Delete your feature branch
   - Sync your fork with upstream

## Coding Standards

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines
- Use meaningful variable and function names
- Add docstrings for public functions
- Keep functions focused and reasonably sized

### Project-Specific Guidelines

- **Single-file design**: Core functionality stays in `torch-hammer.py` for portability
- **Telemetry pattern**: Follow the `TelemetryBase` abstract class pattern for new backends
- **Benchmark pattern**: Follow existing benchmark structure (warmup, timed iterations, summary)
- **Timer usage**: Always use the `Timer` context manager for GPU timing
- **Error handling**: Fail fast with clear error messages

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Keep the first line under 72 characters
- Reference issues when applicable: `Fixes #123`

Example:
```
Add support for Intel dGPU telemetry

- Implement IntelTelemetry class using Level Zero API
- Add detection logic to make_telemetry() factory
- Update README with Intel GPU instructions

Fixes #42

Signed-off-by: Your Name <your.email@example.com>
```

## Testing

Torch Hammer uses pytest for testing. We encourage you to add tests for new functionality and run existing tests before submitting PRs.

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_parsing.py

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing
```

### Test Categories

- **test_parsing.py**: Argument parsing and configuration tests
- **test_utilities.py**: Timer, VerbosePrinter, and utility function tests
- **test_telemetry.py**: Telemetry class and factory tests
- **test_smoke.py**: End-to-end benchmark execution tests (CPU-only)

### Running Benchmarks Manually

```bash
# Basic test on default GPU
./torch-hammer.py --batched-gemm --verbose

# Test on all GPUs
./torch-hammer.py --batched-gemm --all-gpus

# Test specific precision
./torch-hammer.py --batched-gemm --precision-gemm float64
```

### What to Test

When contributing new features:
- Run affected benchmarks on available hardware
- Verify telemetry output is sensible
- Check for any regressions in existing functionality
- Test on multiple platforms if possible (NVIDIA, AMD, CPU)
- Add unit tests for new functionality

### Syntax Check

```bash
python3 -m py_compile torch-hammer.py
```

## Other Ways to Contribute

Code contributions are not the only way to help! Here are other valuable contributions:

- **Documentation**: Improve README, add examples, fix typos
- **Testing**: Test on different hardware configurations and report results
- **Bug Reports**: Report issues with detailed reproduction steps
- **Feature Ideas**: Suggest new benchmarks or improvements
- **Spread the Word**: Star the repository, share with colleagues
- **Answer Questions**: Help others in GitHub Discussions

## References

- [Developer Certificate of Origin](https://developercertificate.org/)
- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

By contributing to Torch Hammer, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE.md).

Thank you for contributing to Torch Hammer! ��⚡
