"""
Configuration module for Graph-Based Injectr.

This module provides centralized configuration management including:
    - Settings: Application settings from environment/files
    - Constants: Static configuration values

Usage:
    from graph_based_injectr.config import Settings, get_settings
    from graph_based_injectr.config.constants import AGENT_MAX_ITERATIONS
"""

from .constants import (
    AGENT_MAX_ITERATIONS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TARGET_ENDPOINT,
    DEFAULT_TARGET_URL,
    DEFAULT_TEMPERATURE,
    KNOWLEDGE_PATH,
    LOOT_PATH,
    NOTES_PATH,
    ORCHESTRATOR_MAX_ITERATIONS,
    REPORTS_PATH,
)
from .settings import Settings, get_settings

__all__ = [
    # Settings
    "Settings",
    "get_settings",
    # Constants
    "AGENT_MAX_ITERATIONS",
    "ORCHESTRATOR_MAX_ITERATIONS",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_TARGET_URL",
    "DEFAULT_TARGET_ENDPOINT",
    "KNOWLEDGE_PATH",
    "LOOT_PATH",
    "REPORTS_PATH",
    "NOTES_PATH",
]
