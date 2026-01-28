"""
Settings Management for Graph-Based Injectr

This module handles application configuration using Pydantic Settings.
Configuration can be loaded from:
    1. Environment variables
    2. .env file in the project root
    3. Default values

The Settings class provides a centralized way to access all configuration
values throughout the application.

Example:
    from graph_based_injectr.config import Settings
    
    settings = Settings()
    print(settings.target_url)
    print(settings.model)
"""

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file into os.environ at module level
# This ensures that external libraries like LiteLLM can see the variables
env_path = Path(__file__).parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

from .constants import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TARGET_ENDPOINT,
    DEFAULT_TARGET_URL,
    DEFAULT_TEMPERATURE,
)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Environment variables can be prefixed with AUTOINJECTOR_ for clarity,
    though unprefixed versions also work for common OpenAI variables.
    
    Attributes:
        openai_api_key: API key for OpenAI (orchestrator LLM)
        model: Model to use for the orchestrator agent
        temperature: Temperature for LLM generation
        max_context_tokens: Maximum tokens in context
        target_url: URL of the target LLM server
        target_endpoint: API endpoint on the target server
        debug: Enable debug mode for verbose logging
    """
    
    # Configure Pydantic settings to read from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AUTOINJECTOR_",
        extra="ignore",  # Ignore extra env vars
    )
    
    # ----- API Keys -----
    # OpenAI API key (for orchestrator LLM)
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for the orchestrator LLM",
        alias="OPENAI_API_KEY",
    )
    
    # Anthropic API key (alternative orchestrator)
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key (optional, for Claude models)",
        alias="ANTHROPIC_API_KEY",
    )
    
    # DeepSeek API key
    deepseek_api_key: Optional[str] = Field(
        default=None,
        description="DeepSeek API key (optional)",
        alias="DEEPSEEK_API_KEY",
    )

    # DeepSeek Base URL
    deepseek_base_url: Optional[str] = Field(
        default="https://api.deepseek.com/v1",
        description="DeepSeek API base URL",
        alias="DEEPSEEK_BASE_URL",
    )
    
    # ----- LLM Configuration -----
    model: str = Field(
        default=DEFAULT_MODEL,
        description="Model to use for the orchestrator agent",
    )
    
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="Temperature for LLM generation (0.0-1.0)",
        ge=0.0,
        le=2.0,
    )
    
    max_context_tokens: int = Field(
        default=DEFAULT_MAX_CONTEXT_TOKENS,
        description="Maximum tokens in the context window",
    )
    
    # ----- Target Configuration -----
    target_url: str = Field(
        default=DEFAULT_TARGET_URL,
        description="URL of the target LLM server",
    )
    
    target_endpoint: str = Field(
        default=DEFAULT_TARGET_ENDPOINT,
        description="API endpoint for chat on the target server",
    )
    
    # ----- LangSmith Configuration -----
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Enable LangChain tracing",
        alias="LANGCHAIN_TRACING_V2",
    )
    
    langchain_api_key: Optional[str] = Field(
        default=None,
        description="LangChain API key",
        alias="LANGCHAIN_API_KEY",
    )
    
    langchain_project: str = Field(
        default="graph_based_injectr",
        description="LangChain project name",
        alias="LANGCHAIN_PROJECT",
    )
    
    # ----- Debug Settings -----
    debug: bool = Field(
        default=False,
        description="Enable debug mode for verbose logging",
    )
    
    def get_full_target_url(self) -> str:
        """
        Get the complete target URL including endpoint.
        
        Returns:
            Full URL string (e.g., "http://localhost:8000/chat")
        """
        base = self.target_url.rstrip("/")
        endpoint = self.target_endpoint
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        return f"{base}{endpoint}"


# Global settings instance (lazily initialized)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Creates the instance on first call (lazy initialization).
    This allows for testing with different configurations.
    
    Returns:
        The Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
