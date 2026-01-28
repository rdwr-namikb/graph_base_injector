"""
Embeddings for AutoInjector Knowledge Base

This module provides functions for generating embeddings from text,
used by the RAG engine for similarity search.
"""

import os
from typing import List, Optional

import numpy as np


def get_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small",
) -> np.ndarray:
    """
    Get embeddings from OpenAI API.
    
    Args:
        texts: List of texts to embed
        model: OpenAI embedding model name
        
    Returns:
        Numpy array of embeddings (n_texts x embedding_dim)
    """
    try:
        import openai
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # OpenAI embeddings API
        response = client.embeddings.create(
            model=model,
            input=texts,
        )
        
        # Extract embeddings
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
        
    except ImportError:
        raise ImportError("openai package required for embeddings")
    except Exception as e:
        raise RuntimeError(f"Error getting embeddings: {e}")


def get_embeddings_local(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Get embeddings using local sentence-transformers model.
    
    Fallback when OpenAI API is not available.
    
    Args:
        texts: List of texts to embed
        model_name: Sentence transformers model name
        
    Returns:
        Numpy array of embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=False)
        return np.array(embeddings)
        
    except ImportError:
        raise ImportError(
            "sentence-transformers required for local embeddings. "
            "Install with: pip install sentence-transformers"
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity (-1 to 1)
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cosine_similarity_matrix(
    queries: np.ndarray,
    corpus: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity matrix between queries and corpus.
    
    Args:
        queries: Query embeddings (n_queries x dim)
        corpus: Corpus embeddings (n_corpus x dim)
        
    Returns:
        Similarity matrix (n_queries x n_corpus)
    """
    # Normalize
    queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    corpus_norm = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    
    # Compute similarities
    return np.dot(queries_norm, corpus_norm.T)
