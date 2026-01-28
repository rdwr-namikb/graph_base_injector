#!/bin/bash
# Run script for Graph-Based Injectr

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run Graph-Based Injectr with passed arguments
python -m graph_based_injectr "$@"
