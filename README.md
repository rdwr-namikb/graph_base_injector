# Graph-Based Injectr - AI-Powered Prompt Injection Testing Framework

An automated framework for testing LLM vulnerabilities through prompt injection and jailbreaking techniques, built with LangGraph for graph-based agent orchestration.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

This project uses **LangGraph** for agent orchestration, providing:
- Declarative graph-based workflow definition
- Built-in state management and persistence
- Conditional routing and branching
- Native streaming support
- LangSmith tracing integration

## Quick Start

### Installation

```bash
# Clone and navigate to the project
cd ~/graph_based_injectr

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

### Configuration

Create a `.env` file:
```bash
# Orchestrator LLM (the AI that plans attacks)
OPENAI_API_KEY=your-api-key
AUTOINJECTOR_MODEL=gpt-4o

# Target configuration
TARGET_URL=http://localhost:8000
TARGET_ENDPOINT=/chat
```

### Usage

```bash
# Start the target server first (in another terminal)
cd ~/autoAgent1
python assistant.py --server

# Run Graph-Based Injectr CLI
python -m graph_based_injectr

# Or with specific target
python -m graph_based_injectr --target http://localhost:8000

# Run specific attack objective
python -m graph_based_injectr --objective "Read the file /etc/passwd"
```

## Features

- **LangGraph-based Agent**: Declarative graph-based workflow with StateGraph
- **Intelligent Attack Planning**: LLM-powered strategy selection
- **Payload Library**: Pre-built jailbreak and injection techniques
- **Adaptive Testing**: Learns from failures and adapts payloads
- **RAG Knowledge Base**: Indexed jailbreak techniques and patterns
- **Multi-turn Attacks**: Conversation-based injection sequences
- **Result Analysis**: Automated success/failure detection

## Project Structure

```
graph_based_injectr/
├── graph_based_injectr/
│   ├── __init__.py
│   ├── __main__.py           # Entry point
│   ├── agents/               # LangGraph agent implementations
│   │   ├── base_agent.py     # StateGraph factory
│   │   ├── injection_agent.py # Injection agent graph
│   │   └── state.py          # Agent state TypedDict
│   ├── config/               # Configuration
│   │   ├── settings.py       # Settings management
│   │   └── constants.py      # Constants
│   ├── interface/            # User interface
│   │   ├── cli.py            # Command-line interface
│   │   └── main.py           # Main entry
│   ├── knowledge/            # RAG knowledge base
│   │   ├── rag.py            # RAG engine
│   │   ├── embeddings.py     # Embedding generation
│   │   └── sources/          # Knowledge sources
│   ├── llm/                  # LLM integration
│   │   ├── llm.py            # LangChain chat model wrapper
│   │   └── config.py         # Model config
│   ├── payloads/             # Injection payloads
│   │   ├── jailbreaks.py     # Jailbreak templates
│   │   ├── encodings.py      # Encoding techniques
│   │   └── mutations.py      # Payload mutations
│   ├── target/               # Target communication
│   │   └── client.py         # Target API client
│   └── tools/                # LangChain tools
│       ├── registry.py       # Tool registry
│       ├── send_prompt/      # Send to target
│       ├── analyze/          # Analyze response
│       ├── mutate/           # Mutate payloads
│       └── notes/            # Log findings
├── loot/                     # Results storage
│   └── reports/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Disclaimer

This tool is intended for **authorized security testing only**. 
Always obtain proper authorization before testing any system.
Unauthorized use may violate laws and regulations.

## License

MIT License
