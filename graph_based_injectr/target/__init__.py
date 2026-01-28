"""
Target module for AutoInjector.

This module provides the client for communicating with target LLM servers:
    - TargetClient: HTTP client for target communication
    - TargetResponse: Response dataclass

The target is the LLM system being tested for prompt injection vulnerabilities.
"""

from .client import TargetClient, TargetResponse

__all__ = [
    "TargetClient",
    "TargetResponse",
]
