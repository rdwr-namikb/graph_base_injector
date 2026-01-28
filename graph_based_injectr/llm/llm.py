"""
LLM Integration for Graph-Based Injectr (LangGraph/LangChain)

This module provides LangChain chat model creation for use with LangGraph.
It supports multiple providers through LangChain's unified interface.

For LangGraph, the chat model is used directly with the graph nodes,
bound to tools via model.bind_tools().

Example:
    chat_model = create_chat_model(model="gpt-4o")
    model_with_tools = chat_model.bind_tools(tools)
    response = await model_with_tools.ainvoke(messages)
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from ..config import DEFAULT_MODEL, DEFAULT_MAX_CONTEXT_TOKENS, get_settings
from .config import ModelConfig


@dataclass
class LLMResponse:
    """
    Response from an LLM generation call.
    
    Kept for backward compatibility with any code that may reference it.
    
    Attributes:
        content: The text content of the response
        tool_calls: List of tool calls requested by the model
        usage: Token usage statistics
        model: The model that generated the response
        finish_reason: Why generation stopped
    """
    content: Optional[str]
    tool_calls: Optional[List[Any]]
    usage: Optional[Dict[str, int]]
    model: str = ""
    finish_reason: str = ""


def _setup_langsmith_tracing() -> None:
    """Configure LangSmith tracing if enabled."""
    settings = get_settings()
    
    if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes") or settings.langchain_tracing_v2:
        tracing_env = {
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_API_KEY": settings.langchain_api_key,
            "LANGCHAIN_PROJECT": settings.langchain_project,
        }
        
        for key, value in tracing_env.items():
            if value:
                os.environ[key] = str(value)


def create_chat_model(
    model: Optional[str] = None,
    config: Optional[ModelConfig] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Create a LangChain chat model for use with LangGraph.
    
    This is the primary function for getting a chat model. It automatically
    selects the appropriate LangChain chat model class based on the model name.
    
    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-3-opus-20240229")
        config: Model configuration (uses defaults if not provided)
        **kwargs: Additional arguments passed to the chat model
        
    Returns:
        A LangChain BaseChatModel instance
        
    Example:
        # Create a chat model
        chat_model = create_chat_model(model="gpt-4o")
        
        # Bind tools for function calling
        model_with_tools = chat_model.bind_tools(tools)
        
        # Use in LangGraph
        response = await model_with_tools.ainvoke(messages)
    """
    # Setup tracing
    _setup_langsmith_tracing()
    
    model = model or DEFAULT_MODEL
    config = config or ModelConfig()
    settings = get_settings()
    
    # Determine provider from model name
    model_lower = model.lower()
    
    # Build common config
    common_config = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        **kwargs,
    }
    
    if "claude" in model_lower or "anthropic" in model_lower:
        # Use Anthropic
        from langchain_openai import ChatOpenAI
        
        # Use LiteLLM for Anthropic models via OpenAI-compatible interface
        return ChatOpenAI(
            model=f"anthropic/{model}" if not model.startswith("anthropic/") else model,
            api_key=settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url="https://api.anthropic.com/v1",
            **common_config,
        )
    
    elif "deepseek" in model_lower:
        # Use DeepSeek via OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            model=model,
            api_key=settings.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=settings.deepseek_base_url or "https://api.deepseek.com/v1",
            **common_config,
        )
    
    else:
        # Default to OpenAI
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key or os.getenv("OPENAI_API_KEY"),
            **common_config,
        )


def create_chat_model_with_fallback(
    primary_model: str,
    fallback_model: str,
    config: Optional[ModelConfig] = None,
) -> BaseChatModel:
    """
    Create a chat model with automatic fallback.
    
    If the primary model fails, automatically switches to the fallback.
    
    Args:
        primary_model: Primary model identifier
        fallback_model: Fallback model identifier
        config: Model configuration
        
    Returns:
        A chat model with fallback behavior
    """
    from langchain_core.runnables import RunnableWithFallbacks
    
    primary = create_chat_model(primary_model, config)
    fallback = create_chat_model(fallback_model, config)
    
    return primary.with_fallbacks([fallback])


class LLM:
    """
    Backward-compatible LLM wrapper.
    
    This class wraps a LangChain chat model to maintain compatibility
    with existing code that uses the old interface. New code should
    use create_chat_model() directly.
    
    Attributes:
        model: The model identifier
        config: Model configuration
        chat_model: The underlying LangChain chat model
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        config: Optional[ModelConfig] = None,
    ):
        """
        Initialize the LLM wrapper.
        
        Args:
            model: Model identifier
            config: Model configuration
        """
        self.model = model or DEFAULT_MODEL
        self.config = config or ModelConfig()
        self.chat_model = create_chat_model(self.model, self.config)
    
    async def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Any]] = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        This method maintains backward compatibility with the old interface.
        
        Args:
            system_prompt: The system prompt
            messages: List of conversation messages
            tools: Optional list of tools (LangChain BaseTool instances)
            
        Returns:
            LLMResponse with content and/or tool calls
        """
        from langchain_core.messages import (
            HumanMessage,
            AIMessage,
            SystemMessage,
            ToolMessage,
        )
        
        # Convert messages to LangChain format
        lc_messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                # Handle tool calls in assistant messages
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    lc_messages.append(AIMessage(
                        content=content or "",
                        tool_calls=[
                            {
                                "id": tc.get("id", tc["id"]) if isinstance(tc, dict) else tc.id,
                                "name": tc.get("function", {}).get("name") if isinstance(tc, dict) else tc.function.name,
                                "args": tc.get("function", {}).get("arguments") if isinstance(tc, dict) else tc.function.arguments,
                            }
                            for tc in tool_calls
                        ] if tool_calls else None,
                    ))
                else:
                    lc_messages.append(AIMessage(content=content))
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                lc_messages.append(ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                ))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))
        
        # Bind tools if provided
        model = self.chat_model
        if tools:
            model = model.bind_tools(tools)
        
        # Generate response
        response = await model.ainvoke(lc_messages)
        
        # Parse response
        return LLMResponse(
            content=response.content if isinstance(response.content, str) else str(response.content),
            tool_calls=response.tool_calls if hasattr(response, "tool_calls") else None,
            usage=response.response_metadata.get("usage") if hasattr(response, "response_metadata") else None,
            model=self.model,
            finish_reason=response.response_metadata.get("finish_reason", "") if hasattr(response, "response_metadata") else "",
        )
    
    async def simple_generate(self, prompt: str) -> str:
        """
        Simple text generation without tools.
        
        Args:
            prompt: The user prompt
            
        Returns:
            The generated text
        """
        response = await self.generate(
            system_prompt="You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content or ""
