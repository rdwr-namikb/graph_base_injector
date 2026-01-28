"""
Interface module for AutoInjector.

This module provides user interfaces for the application:
    - CLI: Command-line interface using Typer
    - main: Main entry point
"""

from .cli import app, main
from .main import main as main_entry

__all__ = [
    "app",
    "main",
    "main_entry",
]
