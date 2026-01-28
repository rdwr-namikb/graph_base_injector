"""
Base Agent Graph for Graph-Based Injectr (LangGraph)

This module provides the StateGraph-based agent implementation using LangGraph.
Instead of a manual ReAct loop, the agent is defined as a graph with nodes
for reasoning (agent) and tool execution (tools).

The graph follows the ReAct pattern:
    1. agent node: Generate response (think/reason)
    2. Conditional edge: Route to tools or END
    3. tools node: Execute tools (act)
    4. Edge back to agent: Observe results and continue

Example:
    graph = create_agent_graph(
        chat_model=model,
        tools=tools,
        system_prompt_fn=get_system_prompt,
    )
    
    async for event in graph.astream(initial_state):
        print(event)
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from .state import AgentState, increment_iteration, mark_finished


def create_agent_graph(
    chat_model: BaseChatModel,
    tools: List[BaseTool],
    system_prompt_fn: Callable[[AgentState], str],
    max_iterations: int = 50,
    checkpointer: Optional[Any] = None,
) -> StateGraph:
    """
    Create a LangGraph StateGraph for the agent.
    
    This creates a ReAct-style agent graph with:
    - An agent node that generates responses using the chat model
    - A tool node that executes tool calls
    - Conditional routing based on whether tools are called
    - Automatic termination on "finish" tool or max iterations
    
    Args:
        chat_model: LangChain chat model (will be bound to tools)
        tools: List of LangChain tools available to the agent
        system_prompt_fn: Function that takes state and returns system prompt
        max_iterations: Maximum iterations before forced stop
        checkpointer: Optional LangGraph checkpointer for persistence
        
    Returns:
        Compiled StateGraph that can be invoked or streamed
        
    Example:
        graph = create_agent_graph(
            chat_model=ChatOpenAI(model="gpt-4o"),
            tools=[send_prompt, analyze, mutate, notes, finish],
            system_prompt_fn=lambda state: f"Objective: {state['objective']}",
        )
        
        result = await graph.ainvoke({
            "messages": [HumanMessage(content="Start testing")],
            "objective": "Read /etc/passwd",
            ...
        })
    """
    
    # Bind tools to the model
    model_with_tools = chat_model.bind_tools(tools)
    
    # Create the tool node
    tool_node = ToolNode(tools)
    
    # Define the agent node
    def agent_node(state: AgentState) -> Dict[str, Any]:
        """
        Agent reasoning node.
        
        Generates a response from the chat model based on current state.
        The system prompt is dynamically generated from state.
        """
        # Check iteration limit
        iteration = state.get("iteration", 0)
        if iteration >= max_iterations:
            return {
                "messages": [AIMessage(content=f"Maximum iterations ({max_iterations}) reached. Stopping.")],
                "finished": True,
            }
        
        # Build messages with dynamic system prompt
        system_prompt = system_prompt_fn(state)
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        messages.extend(state.get("messages", []))
        
        # Generate response
        response = model_with_tools.invoke(messages)
        
        # Increment iteration counter
        return {
            "messages": [response],
            "iteration": iteration + 1,
        }
    
    # Define the routing function
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """
        Determine whether to continue to tools or end.
        
        Routes to:
        - "end" if no tool calls or if "finish" tool was called
        - "tools" if there are tool calls to execute
        """
        # Check if already finished
        if state.get("finished", False):
            return "end"
        
        # Get the last message
        messages = state.get("messages", [])
        if not messages:
            return "end"
        
        last_message = messages[-1]
        
        # Check for tool calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # Check if "finish" tool was called
            for tool_call in last_message.tool_calls:
                tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else getattr(tool_call, "name", "")
                if tool_name == "finish":
                    # Don't route to tools - we want to execute finish and end
                    return "tools"
            return "tools"
        
        return "end"
    
    # Define a post-tool check for finish
    def check_finish(state: AgentState) -> Dict[str, Any]:
        """
        Check if the finish tool was called and mark as finished.
        
        This node runs after tools and checks if we should terminate.
        """
        messages = state.get("messages", [])
        
        # Look for recent finish tool result
        for msg in reversed(messages[-5:]):  # Check last 5 messages
            if isinstance(msg, ToolMessage):
                # Check if this was from the finish tool
                if "TASK COMPLETION REPORT" in msg.content:
                    return {"finished": True}
        
        return {}
    
    # Build the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("check_finish", check_finish)
    
    # Set entry point
    graph.set_entry_point("agent")
    
    # Add conditional edge from agent
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )
    
    # After tools, check if finish was called, then back to agent
    graph.add_edge("tools", "check_finish")
    graph.add_conditional_edges(
        "check_finish",
        lambda state: "end" if state.get("finished", False) else "continue",
        {
            "end": END,
            "continue": "agent",
        }
    )
    
    # Compile the graph
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


async def run_agent_graph(
    graph: StateGraph,
    initial_state: AgentState,
    stream: bool = True,
) -> Union[AgentState, Any]:
    """
    Run an agent graph and return the final state.
    
    Args:
        graph: Compiled StateGraph
        initial_state: Initial state to start with
        stream: Whether to stream events (returns async iterator if True)
        
    Returns:
        Final state if not streaming, async iterator of events if streaming
    """
    if stream:
        return graph.astream(initial_state)
    else:
        return await graph.ainvoke(initial_state)


# Backward compatibility exports
# These allow existing code to still import from this module
__all__ = [
    "create_agent_graph",
    "run_agent_graph",
    "AgentState",
]
