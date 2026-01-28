"""
Finish Tool for Graph-Based Injectr

This tool is used to complete a task and generate a summary report.
It signals to the agent loop that the task is complete.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..registry import register_tool


@dataclass
class TaskPlan:
    """
    Tracks the task plan and progress.
    
    Used to ensure the agent completes objectives before finishing.
    
    Attributes:
        steps: List of planned steps
        completed: List of completed step indices
    """
    steps: List[str] = field(default_factory=list)
    completed: List[int] = field(default_factory=list)
    
    def clear(self) -> None:
        """Clear the plan."""
        self.steps.clear()
        self.completed.clear()
    
    def add_step(self, step: str) -> None:
        """Add a step to the plan."""
        self.steps.append(step)
    
    def complete_step(self, index: int) -> None:
        """Mark a step as complete."""
        if index not in self.completed:
            self.completed.append(index)
    
    def is_complete(self) -> bool:
        """Check if all steps are complete."""
        if not self.steps:
            return True  # No plan means can finish
        return len(self.completed) >= len(self.steps)
    
    def get_progress(self) -> str:
        """Get progress summary."""
        if not self.steps:
            return "No plan defined"
        
        lines = []
        for i, step in enumerate(self.steps):
            status = "✓" if i in self.completed else "○"
            lines.append(f"{status} {i+1}. {step}")
        
        return "\n".join(lines)


class FinishInput(BaseModel):
    """Input schema for finish tool."""
    summary: str = Field(description="Summary of what was accomplished")
    findings: Optional[str] = Field(
        default=None,
        description="Key security findings"
    )
    recommendations: Optional[str] = Field(
        default=None,
        description="Recommendations for the target system"
    )


@tool("finish", args_schema=FinishInput)
def finish(
    summary: str,
    findings: Optional[str] = None,
    recommendations: Optional[str] = None
) -> str:
    """Complete the current task and generate a summary report.

    Use this tool when:
    - You have achieved the objective
    - You have exhausted reasonable attack attempts
    - You need to document final findings

    The report should include:
    - What was attempted
    - What succeeded or failed
    - Key findings and vulnerabilities
    - Recommendations for the target system
    """
    report_parts = [
        "=" * 60,
        "TASK COMPLETION REPORT",
        "=" * 60,
        "",
        "## Summary",
        summary,
    ]
    
    if findings:
        report_parts.extend([
            "",
            "## Key Findings",
            findings,
        ])
    
    if recommendations:
        report_parts.extend([
            "",
            "## Recommendations",
            recommendations,
        ])
    
    report_parts.extend([
        "",
        "=" * 60,
        "Task marked as COMPLETE",
        "=" * 60,
    ])
    
    return "\n".join(report_parts)


# Register the tool
register_tool(finish)
