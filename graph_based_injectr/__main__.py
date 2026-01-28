"""
Graph-Based Injectr entry point for `python -m graph_based_injectr`.

This module allows running Graph-Based Injectr as a Python module:
    python -m graph_based_injectr [options]

It delegates to the main CLI interface.
"""

from graph_based_injectr.interface.main import main

if __name__ == "__main__":
    main()
