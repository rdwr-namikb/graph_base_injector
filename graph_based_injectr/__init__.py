"""
Graph-Based Injectr - AI-Powered Prompt Injection Testing Framework

This package provides tools for automated prompt injection and jailbreaking
testing against LLM-based systems. It uses a LangGraph-based architecture where
an orchestrator LLM plans and executes injection attacks against a target LLM.

Built with LangGraph for declarative graph-based agent orchestration.

Main Components:
    - agents: LangGraph agent implementations (StateGraph-based)
    - llm: LangChain chat model wrapper
    - target: Client for communicating with target LLM
    - payloads: Injection payload templates and mutations
    - tools: LangChain tools available to the agent
    - knowledge: RAG-based knowledge retrieval
"""

__version__ = "0.1.0"
__author__ = "Security Researcher"

from .config.settings import Settings

# Lazy imports to avoid circular dependencies
__all__ = [
    "Settings",
    "__version__",
]
