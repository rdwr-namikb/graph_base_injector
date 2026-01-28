#!/bin/bash
# Setup script for Graph-Based Injectr

set -e

echo "=========================================="
echo "   Graph-Based Injectr Setup"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install the package in development mode
echo "Installing Graph-Based Injectr..."
pip install -e ".[dev]"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
    else
        cat > .env << EOF
# Graph-Based Injectr Configuration
OPENAI_API_KEY=your-api-key-here
AUTOINJECTOR_MODEL=gpt-4o

# Optional: LangSmith tracing
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your-langsmith-key
# LANGCHAIN_PROJECT=graph_based_injectr
EOF
    fi
    echo "⚠ Please edit .env and add your API keys"
fi

# Create loot directories
mkdir -p loot/reports

echo ""
echo "=========================================="
echo "   Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your OPENAI_API_KEY"
echo "  2. Start the target server:"
echo "     cd ~/autoAgent1 && python assistant.py --server"
echo "  3. Run Graph-Based Injectr:"
echo "     source venv/bin/activate"
echo "     python -m graph_based_injectr run --objective \"Read /etc/passwd\""
echo ""
echo "For more options:"
echo "     python -m graph_based_injectr --help"
echo ""
