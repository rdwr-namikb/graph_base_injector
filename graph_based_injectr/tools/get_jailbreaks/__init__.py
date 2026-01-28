"""
Get Jailbreaks Tool for Graph-Based Injectr

This tool retrieves jailbreak templates from the payload library.
It helps the agent discover and use pre-built injection techniques.
"""

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..registry import register_tool
from ...payloads import (
    JailbreakCategory,
    get_all_templates,
    get_templates_by_category,
    get_templates_by_effectiveness,
    JAILBREAK_TEMPLATES,
)


class GetJailbreaksInput(BaseModel):
    """Input schema for get_jailbreaks tool."""
    category: Optional[str] = Field(
        default=None,
        description="Filter by category (role_playing, context_manipulation, encoding, etc.)"
    )
    min_effectiveness: int = Field(
        default=0,
        description="Minimum effectiveness score (1-10)"
    )
    template_name: Optional[str] = Field(
        default=None,
        description="Get a specific template by name"
    )


@tool("get_jailbreaks", args_schema=GetJailbreaksInput)
def get_jailbreaks(
    category: Optional[str] = None,
    min_effectiveness: int = 0,
    template_name: Optional[str] = None
) -> str:
    """Retrieve jailbreak templates from the payload library.

    Use this tool to:
    - Discover available injection techniques
    - Get templates for specific attack categories
    - Find effective techniques to try

    Categories:
    - role_playing: Persona-based jailbreaks (DAN, developer mode)
    - context_manipulation: Change conversation context
    - encoding: Encoded/obfuscated payloads
    - multi_turn: Multi-message attack sequences
    - tool_abuse: Exploit tool/function usage
    - instruction_override: Override system instructions
    - hypothetical: Frame as fiction/education
    """
    # Get specific template
    if template_name:
        template = JAILBREAK_TEMPLATES.get(template_name)
        if template:
            return f"""**{template.name}**
Category: {template.category.value}
Effectiveness: {template.effectiveness}/10
Description: {template.description}

**Template:**
```
{template.template}
```

**Usage:** Replace {{action}} and {{target}} with your objective."""
        else:
            return f"Template '{template_name}' not found. Use get_jailbreaks without template_name to list all."
    
    # Get by category
    if category:
        try:
            cat_enum = JailbreakCategory(category)
            templates = get_templates_by_category(cat_enum)
        except ValueError:
            return f"Invalid category. Valid categories: {[c.value for c in JailbreakCategory]}"
    else:
        templates = get_all_templates()
    
    # Filter by effectiveness
    if min_effectiveness > 0:
        templates = [t for t in templates if t.effectiveness >= min_effectiveness]
    
    if not templates:
        return "No templates match the criteria"
    
    # Format output
    lines = [f"**Found {len(templates)} jailbreak templates:**\n"]
    
    for t in sorted(templates, key=lambda x: -x.effectiveness):
        lines.append(f"- **{t.name}** [{t.category.value}] (effectiveness: {t.effectiveness}/10)")
        lines.append(f"  {t.description}")
    
    lines.append("\n*Use template_name argument to get the full template.*")
    
    return "\n".join(lines)


# Register the tool
register_tool(get_jailbreaks)
