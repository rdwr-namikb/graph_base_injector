"""
Knowledge module for AutoInjector.

This module provides the RAG (Retrieval Augmented Generation) system
for retrieving relevant injection techniques based on context:
    - RAGEngine: Main search engine (static knowledge)
    - AdaptiveRAG: Learning-capable RAG for agents (dynamic knowledge)
    - Document: Knowledge chunk dataclass
    - Embeddings: Vector generation functions
"""

from .rag import RAGEngine, Document
from .embeddings import (
    get_embeddings,
    get_embeddings_local,
    cosine_similarity,
    cosine_similarity_matrix,
)
from .adaptive_rag import (
    AdaptiveRAG,
    LearnedTechnique,
    NearMissConversion,
    VendorProfile,
    ObjectiveType,
    EffectivenessRating,
    ReconInsight,
    ManagerInsight,
    InjectionInsight,
    AnalyzerInsight,
    get_adaptive_rag,
)

__all__ = [
    # Static RAG
    "RAGEngine",
    "Document",
    # Embeddings
    "get_embeddings",
    "get_embeddings_local",
    "cosine_similarity",
    "cosine_similarity_matrix",
    # Adaptive RAG
    "AdaptiveRAG",
    "LearnedTechnique",
    "NearMissConversion",
    "VendorProfile",
    "ObjectiveType",
    "EffectivenessRating",
    "ReconInsight",
    "ManagerInsight",
    "InjectionInsight",
    "AnalyzerInsight",
    "get_adaptive_rag",
]
