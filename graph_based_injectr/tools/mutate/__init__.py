"""
Mutate Tool for Graph-Based Injectr

This tool applies mutations to payloads to create variations that might
bypass filters or succeed where the original failed.
"""

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..registry import register_tool
from ...payloads import (
    mutate_payload,
    generate_variations,
    encode_base64,
    encode_rot13,
    encode_leetspeak,
)


class MutateInput(BaseModel):
    """Input schema for mutate tool."""
    payload: str = Field(description="The payload to mutate")
    strategy: Optional[str] = Field(
        default=None,
        description="Mutation strategy (politeness, urgency, authority, question, steps, debug, test, research, reasoning, partial_encode)"
    )
    count: int = Field(
        default=1,
        description="Number of variations to generate (1-10)"
    )


@tool("mutate", args_schema=MutateInput)
def mutate(payload: str, strategy: Optional[str] = None, count: int = 1) -> str:
    """Apply mutations to a payload to create variations.

    Use this tool when:
    - A payload was blocked and you want to try variations
    - You want to obfuscate a payload
    - You need encoded versions of a payload

    Strategies available:
    - politeness: Add polite phrases
    - urgency: Add urgency indicators
    - authority: Add authority claims
    - question: Rephrase as a question
    - steps: Break into numbered steps
    - debug/test/research: Add context wrappers
    - reasoning: Add justification
    - partial_encode: Partially encode the payload
    """
    if not payload:
        return "Error: No payload provided to mutate"
    
    # Ensure count is reasonable
    count = min(max(int(count), 1), 10)
    
    results = []
    
    if count == 1:
        # Single mutation
        mutated = mutate_payload(payload, strategy)
        results.append(f"**Mutated payload:**\n{mutated}")
    else:
        # Multiple variations
        variations = generate_variations(payload, count)
        results.append(f"**Generated {len(variations)} variations:**")
        for i, var in enumerate(variations, 1):
            results.append(f"\n{i}. {var}")
    
    # Also provide encoded versions
    results.append("\n\n**Encoded versions:**")
    results.append(f"- Base64: {encode_base64(payload)}")
    results.append(f"- ROT13: {encode_rot13(payload)}")
    results.append(f"- Leetspeak: {encode_leetspeak(payload)}")
    
    return "\n".join(results)


# Register the tool
register_tool(mutate)
