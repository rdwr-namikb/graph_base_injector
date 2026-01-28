"""
Tool Registry for Graph-Based Injectr (LangGraph/LangChain)

This module provides tool creation and registration using LangChain's
tool abstractions. Tools are created as LangChain BaseTool instances
that can be used with LangGraph's ToolNode.

The registry pattern allows tools to be:
    - Created with injected dependencies (target_client)
    - Discovered dynamically
    - Used with LangGraph's ToolNode
"""

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..target import TargetClient


# Global target client reference (set during tool initialization)
_target_client: Optional["TargetClient"] = None


def set_target_client(client: "TargetClient") -> None:
    """Set the global target client for tools that need it."""
    global _target_client
    _target_client = client


def get_target_client() -> Optional["TargetClient"]:
    """Get the global target client."""
    return _target_client


# Global tool registry
_tool_registry: Dict[str, BaseTool] = {}


def register_tool(tool: BaseTool) -> None:
    """
    Register a tool in the global registry.
    
    Args:
        tool: The LangChain tool to register
    """
    _tool_registry[tool.name] = tool


def get_tool(name: str) -> Optional[BaseTool]:
    """
    Get a tool by name.
    
    Args:
        name: The tool name
        
    Returns:
        The tool, or None if not found
    """
    return _tool_registry.get(name)


def get_all_tools() -> List[BaseTool]:
    """
    Get all registered tools.
    
    Returns:
        List of all registered LangChain tools
    """
    return list(_tool_registry.values())


def clear_registry() -> None:
    """Clear all registered tools."""
    _tool_registry.clear()


def create_structured_tool(
    name: str,
    description: str,
    func: Callable,
    args_schema: Optional[type] = None,
    coroutine: Optional[Callable] = None,
) -> StructuredTool:
    """
    Create a LangChain StructuredTool.
    
    This is a helper for creating tools with the LangChain pattern.
    
    Args:
        name: Tool name
        description: Tool description
        func: Synchronous function (can be a dummy if coroutine provided)
        args_schema: Pydantic model for arguments
        coroutine: Async function implementation
        
    Returns:
        A LangChain StructuredTool
    """
    return StructuredTool(
        name=name,
        description=description,
        func=func,
        coroutine=coroutine,
        args_schema=args_schema,
    )
