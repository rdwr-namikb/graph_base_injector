"""
LLM module for Graph-Based Injectr.

This module provides LangChain chat model integration for the agent:
    - create_chat_model: Factory function for creating chat models
    - LLM: Backward-compatible wrapper class
    - LLMResponse: Response dataclass
    - ModelConfig: Configuration for model parameters

For LangGraph, use create_chat_model() to get a LangChain chat model
that can be bound to tools and used in graph nodes.
"""

from .config import ModelConfig
from .llm import LLM, LLMResponse, create_chat_model, create_chat_model_with_fallback

__all__ = [
    # Primary function for LangGraph
    "create_chat_model",
    "create_chat_model_with_fallback",
    # Backward compatibility
    "LLM",
    "LLMResponse",
    "ModelConfig",
]
