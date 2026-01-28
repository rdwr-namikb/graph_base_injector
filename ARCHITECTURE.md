# Graph-Based Injectr Architecture

## Overview

Graph-Based Injectr is an AI-powered prompt injection and jailbreaking testing framework built with **LangGraph** for declarative graph-based agent orchestration.

## Key Differences from Traditional Agents

This implementation uses LangGraph's `StateGraph` pattern instead of manual ReAct loops:

- **Declarative Graph**: Agent workflow defined as a graph with nodes and edges
- **Built-in State Management**: State flows through the graph automatically
- **Conditional Routing**: Branching logic defined as conditional edges
- **Native Tool Support**: LangChain tools integrated via `ToolNode`
- **Streaming**: Built-in support for streaming events from the graph

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Graph-Based Injectr System                              │
└─────────────────────────────────────────────────────────────────────────────────┘

                                    ┌─────────────────┐
                                    │   CLI / TUI     │
                                    │   Interface     │
                                    └────────┬────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         LangGraph Orchestration Layer                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        StateGraph (InjectionAgent)                       │    │
│  │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                │    │
│  │  │   START     │────▶│    agent    │────▶│ conditional │                │    │
│  │  │             │     │   (think)   │     │   (route)   │                │    │
│  │  └─────────────┘     └─────────────┘     └──────┬──────┘                │    │
│  │                                                  │                       │    │
│  │                           ┌──────────────────────┼──────────────────────┐│    │
│  │                           │                      │                      ││    │
│  │                           ▼                      ▼                      ││    │
│  │                    ┌─────────────┐         ┌───────────┐               ││    │
│  │                    │   tools     │         │    END    │               ││    │
│  │                    │  (execute)  │         │           │               ││    │
│  │                    └──────┬──────┘         └───────────┘               ││    │
│  │                           │                                             ││    │
│  │                           ▼                                             ││    │
│  │                    ┌─────────────┐                                      ││    │
│  │                    │check_finish │───────────────────────────────────────┘│    │
│  │                    └─────────────┘                                       │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  AgentState (TypedDict):                                                        │
│  • messages: Annotated[Sequence[BaseMessage], add_messages]                     │
│  • objective, target_info, attack_history, notes, iteration, finished           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Knowledge Layer (RAG)                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────┐  ┌────────────────────┐  ┌─────────────────────────┐    │
│  │  Jailbreak DB      │  │  Technique Docs    │  │   Attack Patterns       │    │
│  │  ──────────────────│  │  ──────────────────│  │  ─────────────────────  │    │
│  │  • Known bypasses  │  │  • Methodology     │  │  • Success patterns     │    │
│  │  • DAN prompts     │  │  • Best practices  │  │  • Failure analysis     │    │
│  │  • Model-specific  │  │  • Research papers │  │  • Adaptive strategies  │    │
│  └────────────────────┘  └────────────────────┘  └─────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         LangChain Integration Layer                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────┐      ┌─────────────────────────────────────┐    │
│  │    ChatModel (LangChain)   │      │         Target Client               │    │
│  │  ──────────────────────────│      │  ─────────────────────────────────  │    │
│  │  • ChatOpenAI              │      │  • HTTP API communication           │    │
│  │  • model.bind_tools()      │      │  • Response parsing                 │    │
│  │  • Streaming support       │      │  • Session management               │    │
│  │                            │      │                                     │    │
│  │  (GPT-4o, Claude, etc.)    │      │  Target: localhost:8000             │    │
│  └────────────────────────────┘      │  Model: GPT-5.2 w/ file read tool   │    │
│                                      └─────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         LangChain Tools Layer                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐  │
│  │ send_prompt │ │  analyze    │ │  mutate     │ │   notes     │ │  finish   │  │
│  │ ─────────── │ │ ─────────── │ │ ─────────── │ │ ─────────── │ │ ───────── │  │
│  │ @tool       │ │ @tool       │ │ @tool       │ │ @tool       │ │ @tool     │  │
│  │ Send to     │ │ Evaluate    │ │ Transform   │ │ Log results │ │ Complete  │  │
│  │ target LLM  │ │ response    │ │ payloads    │ │ & findings  │ │ task      │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘  │
│                                                                                  │
│  All tools use LangChain's @tool decorator for automatic schema generation      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow (LangGraph)

```
1. User Input → CLI Interface
       │
       ▼
2. Create Initial State (AgentState TypedDict)
       │
       ▼
3. graph.astream(initial_state)
       │
       ▼
┌──────────────────────────────────────────────────────┐
│             LangGraph Execution Loop                  │
│                                                       │
│  4. agent node → LLM generates response with tools    │
│         │                                             │
│         ▼                                             │
│  5. conditional edge → Check for tool calls           │
│         │                                             │
│    ┌────┴────┐                                        │
│    ▼         ▼                                        │
│  tools    END (no tools)                              │
│    │                                                  │
│    ▼                                                  │
│  6. tools node → ToolNode executes LangChain tools    │
│         │                                             │
│         ▼                                             │
│  7. check_finish → If finish tool called, END         │
│         │                                             │
│         ▼                                             │
│  8. Loop back to agent node                           │
│                                                       │
└──────────────────────────────────────────────────────┘
       │
       ▼
9. Yield events via astream() for real-time updates
       │
       ▼
10. Final state contains all messages and results
```

## Key Components

### 1. StateGraph Factory (agents/base_agent.py)
- `create_agent_graph()`: Creates a compiled StateGraph
- Defines agent node, tool node, and conditional routing
- Handles iteration limits and finish detection

### 2. InjectionAgent Factory (agents/injection_agent.py)
- `create_injection_agent()`: Creates injection-specific graph
- Dynamic system prompt with RAG context and notes
- `create_injection_initial_state()`: Helper for initial state

### 3. AgentState (agents/state.py)
- TypedDict schema for LangGraph state
- Uses `Annotated[Sequence[BaseMessage], add_messages]` for message accumulation
- Tracks objective, target_info, attack_history, notes, etc.

### 4. LangChain Tools (tools/)
- All tools use `@tool` decorator from `langchain_core.tools`
- Automatic Pydantic schema generation for arguments
- Compatible with LangGraph's `ToolNode`

### 5. Chat Model Integration (llm/llm.py)
- `create_chat_model()`: Factory for LangChain chat models
- Supports OpenAI, Anthropic, DeepSeek via LangChain
- Models bound to tools via `model.bind_tools()`

### 6. Target Client (target/client.py)
- Communicates with target LLM API
- Handles the target at localhost:8000
- Parses responses and tracks conversation state

### 7. RAG Knowledge Base (knowledge/)
- Indexed jailbreak techniques
- Model-specific vulnerabilities
- Success/failure patterns

## Target System

The target is an LLM assistant running at:
- **URL**: http://localhost:8000
- **Model**: GPT-5.2
- **Tools Available**: read_file (file system access)
- **API Endpoint**: POST /chat with {"message": "..."}

## LangGraph Benefits

1. **Declarative Workflow**: Define agent logic as a graph, not imperative code
2. **Built-in State Management**: Automatic state accumulation and persistence
3. **Streaming**: Native `astream()` for real-time event handling
4. **Checkpointing**: Optional persistence for long-running sessions
5. **Debugging**: Graph visualization and LangSmith tracing
6. **Extensibility**: Easy to add new nodes and routing logic

## Security Notes

This tool is for **authorized security testing only**.
Always obtain proper authorization before testing any system.
