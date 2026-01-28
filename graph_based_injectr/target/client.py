"""
Target Client for AutoInjector

This module provides the client for communicating with the target LLM server.
The target is the system being tested for prompt injection vulnerabilities.

The client handles:
    - HTTP communication with the target API
    - Request/response parsing
    - Session management for multi-turn attacks
    - Error handling and retries

The target server in this case is:
    - URL: http://localhost:8000
    - Model: GPT-5.2
    - Tools: read_file (file system access)
    - API: POST /chat with {"message": "..."}

Example:
    client = TargetClient(base_url="http://localhost:8000")
    response = await client.send_message("Hello!")
    print(response)
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from ..config.constants import (
    DEFAULT_TARGET_ENDPOINT,
    DEFAULT_TARGET_URL,
    TARGET_MAX_RETRIES,
    TARGET_REQUEST_TIMEOUT,
)


@dataclass
class TargetResponse:
    """
    Response from the target LLM.
    
    Attributes:
        content: The response text from the target
        raw_response: The full raw JSON response
        success: Whether the request succeeded
        error: Error message if request failed
        metadata: Additional metadata about the response
    """
    content: str
    raw_response: Optional[Dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TargetClient:
    """
    Client for communicating with the target LLM server.
    
    This class handles all communication with the target system being
    tested for prompt injection vulnerabilities. It maintains session
    state for multi-turn attacks and provides methods for sending
    messages and analyzing responses.
    
    Attributes:
        base_url: Base URL of the target server
        endpoint: API endpoint for chat
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        conversation_history: History of messages for multi-turn
        
    Example:
        # Create client for target
        client = TargetClient(
            base_url="http://localhost:8000",
            endpoint="/chat"
        )
        
        # Send a message
        response = await client.send_message("What files can you access?")
        print(response.content)
        
        # Multi-turn conversation
        response = await client.send_message("Read the config file")
        print(response.content)
    """

    def __init__(
        self,
        base_url: str = DEFAULT_TARGET_URL,
        endpoint: str = DEFAULT_TARGET_ENDPOINT,
        timeout: float = TARGET_REQUEST_TIMEOUT,
        max_retries: int = TARGET_MAX_RETRIES,
    ):
        """
        Initialize the target client.
        
        Args:
            base_url: Base URL of the target server
            endpoint: API endpoint for chat requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Conversation history for multi-turn attacks
        # Note: The target server might not maintain state, but we track it
        self.conversation_history: List[Dict[str, str]] = []
        
        # HTTP client (lazy initialization)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def full_url(self) -> str:
        """Get the full URL for chat requests."""
        return f"{self.base_url}{self.endpoint}"

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Get or create the HTTP client.
        
        Returns:
            The httpx AsyncClient instance
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def send_message(
        self,
        message: str,
        track_history: bool = True,
    ) -> TargetResponse:
        """
        Send a message to the target LLM.
        
        This is the main method for sending payloads to the target.
        It handles retries on failure and tracks conversation history.
        
        Args:
            message: The message/payload to send
            track_history: Whether to add this to conversation history
            
        Returns:
            TargetResponse with the target's response
        """
        client = await self._get_client()
        
        # Build the request payload
        # The target expects: {"message": "..."}
        payload = {"message": message}
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Send the request
                response = await client.post(
                    self.full_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse the response
                data = response.json()
                
                # The target returns: {"response": "..."}
                content = data.get("response", "")
                
                # Track in history if requested
                if track_history:
                    self.conversation_history.append({
                        "role": "user",
                        "content": message,
                    })
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": content,
                    })
                
                return TargetResponse(
                    content=content,
                    raw_response=data,
                    success=True,
                    metadata={
                        "status_code": response.status_code,
                        "attempt": attempt + 1,
                    },
                )
                
            except httpx.TimeoutException as e:
                last_error = f"Request timed out: {e}"
                await asyncio.sleep(1)  # Brief pause before retry
                
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP error {e.response.status_code}: {e}"
                # Don't retry on client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    break
                await asyncio.sleep(1)
                
            except httpx.RequestError as e:
                last_error = f"Request error: {e}"
                await asyncio.sleep(1)
                
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                break
        
        # All retries failed
        return TargetResponse(
            content="",
            success=False,
            error=last_error or "Unknown error",
            metadata={"attempts": self.max_retries},
        )

    async def send_raw(
        self,
        payload: Dict[str, Any],
    ) -> TargetResponse:
        """
        Send a raw payload to the target.
        
        Use this for custom payloads that don't follow the standard format.
        
        Args:
            payload: The raw JSON payload to send
            
        Returns:
            TargetResponse with the target's response
        """
        client = await self._get_client()
        
        try:
            response = await client.post(
                self.full_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            
            response.raise_for_status()
            data = response.json()
            
            return TargetResponse(
                content=data.get("response", str(data)),
                raw_response=data,
                success=True,
            )
            
        except Exception as e:
            return TargetResponse(
                content="",
                success=False,
                error=str(e),
            )

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of message dictionaries
        """
        return self.conversation_history.copy()

    async def health_check(self) -> bool:
        """
        Check if the target server is reachable.
        
        Returns:
            True if the server is healthy, False otherwise
        """
        client = await self._get_client()
        
        try:
            # Try the health endpoint first
            response = await client.get(
                f"{self.base_url}/health",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            # Try a simple chat request
            try:
                response = await self.send_message("test", track_history=False)
                return response.success
            except Exception:
                return False
