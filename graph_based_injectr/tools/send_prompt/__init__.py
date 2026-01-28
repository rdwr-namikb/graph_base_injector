"""
Send Prompt Tool for Graph-Based Injectr

This tool sends a prompt (injection payload) to the target LLM and returns
the response. It's the primary tool for interacting with the target.
"""

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..registry import get_target_client, register_tool


class SendPromptInput(BaseModel):
    """Input schema for send_prompt tool."""
    prompt: str = Field(description="The prompt/message to send to the target LLM")


@tool("send_prompt", args_schema=SendPromptInput)
async def send_prompt(prompt: str) -> str:
    """Send a prompt/message to the target LLM and get its response.

    Use this tool to:
    - Send injection payloads to the target
    - Test different attack techniques
    - Observe the target's behavior
    - Build multi-turn conversations

    The target has file reading capabilities that may be exploitable.
    """
    if not prompt:
        return "Error: No prompt provided"
    
    target_client = get_target_client()
    if target_client is None:
        return "Error: Target client not initialized"
    
    # Send to target
    response = await target_client.send_message(prompt)
    
    if response.success:
        return f"Target Response:\n{response.content}"
    else:
        return f"Error sending prompt: {response.error}"


# Register the tool
register_tool(send_prompt)
