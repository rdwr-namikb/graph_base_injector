"""
Agents module for Graph-Based Injectr.

This module provides LangGraph-based agent implementations for prompt injection testing:
    - create_agent_graph: Factory function for creating StateGraph agents
    - create_injection_agent: Factory function for the injection-specific agent
    - create_recon_agent: Factory function for the reconnaissance agent
    - get_manager_decision: Function to get strategic oversight from Manager Agent
    - AgentState: TypedDict state schema for LangGraph
    - create_initial_state: Helper to create initial agent state

The agents use LangGraph's StateGraph pattern to implement the ReAct workflow
(Reason + Act) with declarative graph-based orchestration.

Agent Hierarchy:
    Manager Agent (Strategic Oversight)
         │
         ├── Approves/modifies injection strategies
         ├── Critiques analyzer assessments
         └── Maintains campaign-level view
         │
    Injection Agent ←→ Analyzer Agent (Tactical)
         │                    │
         └── Sends payloads   └── Evaluates responses
"""

from .base_agent import create_agent_graph, run_agent_graph
from .injection_agent import (
    create_injection_agent,
    create_injection_initial_state,
    INJECTION_AGENT_PROMPT,
)
from .recon_agent import (
    create_recon_agent,
    create_recon_initial_state,
    create_recon_tools,
    parse_recon_results,
    get_recommended_injection_phase,
    RECON_AGENT_PROMPT,
    RECON_PROBES,
)
from .manager_agent import (
    get_manager_decision,
    should_persist_on_vector,
    update_state_with_manager_feedback,
    ManagerVerdict,
    ConfidenceAdjustment,
    ManagerDecision,
    NEAR_MISS_INDICATORS,
    SUCCESS_INDICATORS,
    DETECTION_INDICATORS,
)
from .state import (
    AgentState,
    TargetInfo,
    AttackAttempt,
    ReconInfo,
    InjectionPhase,
    InjectionGuidance,
    ManagerFeedback,
    BypassStrategy,
    ParsedReconIntelligence,
    create_initial_state,
    get_last_message,
    increment_iteration,
    add_attack_attempt,
    mark_finished,
    update_recon_info,
    mark_recon_complete,
    update_injection_phase,
    update_injection_guidance,
    parse_recon_for_attack,
    format_recon_for_injection_agent,
    generate_bypass_strategies,
)

__all__ = [
    # Graph factory functions
    "create_agent_graph",
    "run_agent_graph",
    "create_injection_agent",
    "create_injection_initial_state",
    "create_recon_agent",
    "create_recon_initial_state",
    "create_recon_tools",
    # Manager agent
    "get_manager_decision",
    "should_persist_on_vector",
    "update_state_with_manager_feedback",
    "ManagerVerdict",
    "ConfidenceAdjustment",
    "ManagerDecision",
    # State types and helpers
    "AgentState",
    "TargetInfo",
    "AttackAttempt",
    "ReconInfo",
    "InjectionPhase",
    "InjectionGuidance",
    "ManagerFeedback",
    "BypassStrategy",
    "ParsedReconIntelligence",
    "create_initial_state",
    "get_last_message",
    "increment_iteration",
    "add_attack_attempt",
    "mark_finished",
    "update_recon_info",
    "mark_recon_complete",
    "update_injection_phase",
    "update_injection_guidance",
    # Recon intelligence
    "parse_recon_for_attack",
    "format_recon_for_injection_agent",
    "generate_bypass_strategies",
    "parse_recon_results",
    "get_recommended_injection_phase",
    # Detection indicators
    "NEAR_MISS_INDICATORS",
    "SUCCESS_INDICATORS",
    "DETECTION_INDICATORS",
    # Prompts
    "INJECTION_AGENT_PROMPT",
    "RECON_AGENT_PROMPT",
    "RECON_PROBES",
]
