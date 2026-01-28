"""
Notes Tool for Graph-Based Injectr

This tool allows the agent to save and retrieve notes about findings,
successful techniques, and progress during the attack session.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..registry import register_tool
from ...config.constants import NOTES_PATH


class NotesInput(BaseModel):
    """Input schema for notes tool."""
    action: str = Field(
        description="The action: save, read, list, or delete"
    )
    key: Optional[str] = Field(
        default=None,
        description="Unique identifier for the note"
    )
    content: Optional[str] = Field(
        default=None,
        description="The note content (required for save)"
    )
    category: str = Field(
        default="info",
        description="Note category (finding, success, failure, info)"
    )


# In-memory notes storage (persisted to file)
_notes: Dict[str, Dict[str, Any]] = {}


def _load_notes() -> None:
    """Load notes from file."""
    global _notes
    
    if NOTES_PATH.exists():
        try:
            with open(NOTES_PATH, 'r') as f:
                _notes = json.load(f)
        except Exception:
            _notes = {}
    else:
        _notes = {}


def _save_notes() -> None:
    """Save notes to file."""
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(NOTES_PATH, 'w') as f:
        json.dump(_notes, f, indent=2)


def get_all_notes_sync() -> Dict[str, Dict[str, Any]]:
    """
    Get all notes synchronously.
    
    Used by agents to include notes in system prompt.
    
    Returns:
        Dictionary of all notes
    """
    global _notes
    
    if not _notes:
        _load_notes()
    
    return _notes.copy()


@tool("notes", args_schema=NotesInput)
def notes(
    action: str,
    key: Optional[str] = None,
    content: Optional[str] = None,
    category: str = "info"
) -> str:
    """Save and retrieve notes about findings, techniques, and progress.

    Use this tool to:
    - Record successful injection techniques
    - Log discovered vulnerabilities
    - Track failed approaches to avoid repeating
    - Document findings for the final report

    Categories:
    - finding: Discovered vulnerabilities or behaviors
    - success: Successful injection techniques
    - failure: Failed approaches (for reference)
    - info: General observations

    Actions:
    - save: Save a new note (requires key, content)
    - read: Read a specific note (requires key)
    - list: List all saved notes
    - delete: Delete a note (requires key)
    """
    global _notes
    
    # Load notes if not loaded
    if not _notes:
        _load_notes()
    
    if action == "save":
        if not key or not content:
            return "Error: 'key' and 'content' required for save"
        
        _notes[key] = {
            "content": content,
            "category": category,
        }
        _save_notes()
        return f"Saved note '{key}' in category '{category}'"
    
    elif action == "read":
        if not key:
            return "Error: 'key' required for read"
        
        note = _notes.get(key)
        if not note:
            return f"Note '{key}' not found"
        
        return f"**{key}** [{note.get('category', 'info')}]:\n{note.get('content', '')}"
    
    elif action == "list":
        if not _notes:
            return "No notes saved yet"
        
        lines = ["**Saved notes:**"]
        for note_key, note_data in _notes.items():
            cat = note_data.get("category", "info") if isinstance(note_data, dict) else "info"
            preview = note_data.get("content", str(note_data))[:50] if isinstance(note_data, dict) else str(note_data)[:50]
            lines.append(f"- [{cat}] {note_key}: {preview}...")
        
        return "\n".join(lines)
    
    elif action == "delete":
        if not key:
            return "Error: 'key' required for delete"
        
        if key in _notes:
            del _notes[key]
            _save_notes()
            return f"Deleted note '{key}'"
        else:
            return f"Note '{key}' not found"
    
    else:
        return f"Unknown action: {action}. Use: save, read, list, or delete"


# Register the tool
register_tool(notes)
