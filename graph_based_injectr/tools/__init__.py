"""
Tools module for Graph-Based Injectr.

This module provides all LangChain tools available to the injection agent:
    - send_prompt: Send payloads to the target LLM
    - analyze: Analyze target responses
    - mutate: Create payload variations
    - notes: Save and retrieve findings
    - finish: Complete task with summary
    - get_jailbreaks: Retrieve jailbreak templates

Tools are registered at import time and available through the registry.
All tools are LangChain BaseTool instances compatible with LangGraph's ToolNode.
"""

from .registry import (
    register_tool,
    get_tool,
    get_all_tools,
    clear_registry,
    set_target_client,
    get_target_client,
)

# Import tools to register them
from . import send_prompt
from . import analyze
from . import mutate
from . import notes
from . import finish
from . import get_jailbreaks

# Re-export TaskPlan for agent usage
from .finish import TaskPlan

# Re-export individual tools for direct access
from .send_prompt import send_prompt as send_prompt_tool
from .analyze import analyze as analyze_tool
from .mutate import mutate as mutate_tool
from .notes import notes as notes_tool, get_all_notes_sync
from .finish import finish as finish_tool
from .get_jailbreaks import get_jailbreaks as get_jailbreaks_tool

__all__ = [
    # Registry functions
    "register_tool",
    "get_tool",
    "get_all_tools",
    "clear_registry",
    "set_target_client",
    "get_target_client",
    # TaskPlan
    "TaskPlan",
    # Individual tools
    "send_prompt_tool",
    "analyze_tool",
    "mutate_tool",
    "notes_tool",
    "finish_tool",
    "get_jailbreaks_tool",
    # Utility
    "get_all_notes_sync",
]
