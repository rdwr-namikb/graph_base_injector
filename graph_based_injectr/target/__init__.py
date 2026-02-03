"""
Target module for AutoInjector.

This module provides clients for communicating with target LLM servers:
    - TargetClient: HTTP client for generic target communication
    - TargetResponse: Response dataclass for generic targets
    - GandalfClient: Specialized client for Lakera AI's Gandalf challenge
    - GandalfResponse: Response dataclass for Gandalf

The target is the LLM system being tested for prompt injection vulnerabilities.
"""

from .client import TargetClient, TargetResponse
from .gandalf_client import GandalfClient, GandalfResponse

__all__ = [
    "TargetClient",
    "TargetResponse",
    "GandalfClient",
    "GandalfResponse",
]
