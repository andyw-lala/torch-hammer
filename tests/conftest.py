# Copyright 2024-2026 Hewlett Packard Enterprise Development LP
# SPDX-License-Identifier: Apache-2.0
"""
Pytest fixtures and configuration for torch-hammer tests.
"""
import sys
import os
from pathlib import Path

import pytest

# Add parent directory to path so we can import torch-hammer
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import the main module (handles the hyphen in filename)
import importlib.util
spec = importlib.util.spec_from_file_location("torch_hammer", ROOT_DIR / "torch-hammer.py")
torch_hammer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(torch_hammer)


@pytest.fixture
def parser():
    """Provide a fresh argument parser for testing."""
    return torch_hammer.build_parser()


@pytest.fixture
def default_args(parser):
    """Provide default parsed arguments."""
    return parser.parse_args([])


@pytest.fixture
def th():
    """Provide the torch-hammer module for testing."""
    return torch_hammer
