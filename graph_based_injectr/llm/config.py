"""
LLM Configuration for AutoInjector

This module defines the configuration for LLM interactions,
including model parameters, retry settings, and rate limiting.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """
    Configuration for LLM model parameters.
    
    Attributes:
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
        max_context_tokens: Maximum tokens in context window
        top_p: Nucleus sampling parameter
        frequency_penalty: Repetition penalty
        presence_penalty: Topic penalty
        max_retries: Maximum retry attempts on failure
        retry_delay: Base delay between retries (seconds)
        timeout: Request timeout (seconds)
    """
    
    temperature: float = 0.7
    max_tokens: Optional[int] = None  # None means no limit
    max_context_tokens: int = 128000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 60.0
