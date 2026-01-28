"""
Configuration Constants for AutoInjector

This module defines all the constant values used throughout the application.
These include default settings, limits, and other configuration values that
rarely change but need to be accessible across the codebase.

Constants:
    - Agent behavior limits (max iterations, timeouts)
    - Default model configurations
    - Target defaults
    - Path configurations
"""

from pathlib import Path

# =============================================================================
# Agent Configuration
# =============================================================================

# Maximum iterations for the main agent loop before forcing termination.
# This prevents infinite loops if the agent gets stuck.
AGENT_MAX_ITERATIONS = 50

# Maximum iterations for the crew orchestrator when managing multiple workers.
ORCHESTRATOR_MAX_ITERATIONS = 30

# Timeout for individual tool executions (seconds)
TOOL_EXECUTION_TIMEOUT = 60

# Maximum conversation history length (messages) before truncation
MAX_CONVERSATION_HISTORY = 100

# =============================================================================
# LLM Configuration
# =============================================================================

# Default model for the orchestrator agent (the one planning attacks)
DEFAULT_MODEL = "gpt-4o"

# Maximum tokens in context window (used for conversation truncation)
DEFAULT_MAX_CONTEXT_TOKENS = 128000

# Default temperature for generation
DEFAULT_TEMPERATURE = 0.7

# =============================================================================
# Target Configuration
# =============================================================================

# Default target URL (the LLM server we're testing)
DEFAULT_TARGET_URL = "http://localhost:8000"

# Default API endpoint for chat
DEFAULT_TARGET_ENDPOINT = "/chat"

# Timeout for target requests (seconds)
TARGET_REQUEST_TIMEOUT = 30

# Maximum retries for failed requests
TARGET_MAX_RETRIES = 3

# =============================================================================
# Knowledge Base Configuration
# =============================================================================

# Path to knowledge sources relative to package
KNOWLEDGE_PATH = Path(__file__).parent.parent / "knowledge" / "sources"

# Default embedding model for RAG
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Number of top results to retrieve from RAG
RAG_TOP_K = 5

# =============================================================================
# Payload Configuration
# =============================================================================

# Maximum payload length (characters)
MAX_PAYLOAD_LENGTH = 10000

# Maximum number of mutation attempts
MAX_MUTATION_ATTEMPTS = 10

# =============================================================================
# Output Configuration
# =============================================================================

# Path to store loot (results, logs, etc.)
LOOT_PATH = Path(__file__).parent.parent.parent / "loot"

# Path to store reports
REPORTS_PATH = LOOT_PATH / "reports"

# Path to store notes
NOTES_PATH = LOOT_PATH / "notes.json"
