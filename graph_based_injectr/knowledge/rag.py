"""
RAG (Retrieval Augmented Generation) Engine for AutoInjector

This module provides the RAG engine for retrieving relevant injection
techniques from the knowledge base based on semantic similarity.

The RAG engine:
    1. Indexes knowledge sources (techniques, patterns, etc.)
    2. Generates embeddings for documents
    3. Retrieves relevant context for the agent
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .embeddings import get_embeddings, get_embeddings_local, cosine_similarity_matrix
from ..config.constants import KNOWLEDGE_PATH, RAG_TOP_K


@dataclass
class Document:
    """
    A chunk of knowledge for retrieval.
    
    Attributes:
        content: The text content
        source: Where this content came from
        metadata: Additional metadata
        embedding: Vector embedding (set during indexing)
        doc_id: Unique document identifier
    """
    content: str
    source: str
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[np.ndarray] = None
    doc_id: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.doc_id is None:
            self.doc_id = f"{hash(self.content)}_{hash(self.source)}"


class RAGEngine:
    """
    RAG engine for knowledge retrieval.
    
    Indexes documents from the knowledge directory and provides
    semantic search capabilities for the agent.
    
    Attributes:
        knowledge_path: Path to knowledge sources
        embedding_model: Model to use for embeddings
        use_local_embeddings: Whether to use local model
        documents: Indexed documents
        
    Example:
        rag = RAGEngine()
        rag.index()
        
        results = rag.search("bypass safety filters")
        for result in results:
            print(result)
    """

    def __init__(
        self,
        knowledge_path: Path = None,
        embedding_model: str = "text-embedding-3-small",
        use_local_embeddings: bool = False,
    ):
        """
        Initialize the RAG engine.
        
        Args:
            knowledge_path: Path to knowledge directory
            embedding_model: Model for embeddings
            use_local_embeddings: Use local sentence-transformers
        """
        self.knowledge_path = knowledge_path or KNOWLEDGE_PATH
        self.embedding_model = embedding_model
        self.use_local_embeddings = use_local_embeddings
        
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self._indexed = False

    def index(self, force: bool = False) -> None:
        """
        Index all documents in the knowledge directory.
        
        Reads files from the knowledge directory, chunks them,
        and generates embeddings.
        
        Args:
            force: Force re-indexing even if already indexed
        """
        if self._indexed and not force:
            return
        
        chunks = []
        
        # Process knowledge directory
        if self.knowledge_path.exists():
            for file in self.knowledge_path.rglob("*"):
                if not file.is_file():
                    continue
                
                try:
                    if file.suffix in [".txt", ".md"]:
                        content = file.read_text(encoding="utf-8", errors="ignore")
                        file_chunks = self._chunk_text(content, source=str(file))
                        chunks.extend(file_chunks)
                    
                    elif file.suffix == ".json":
                        data = json.loads(file.read_text(encoding="utf-8"))
                        if isinstance(data, list):
                            for item in data:
                                chunks.append(Document(
                                    content=json.dumps(item, indent=2),
                                    source=str(file),
                                    metadata=item if isinstance(item, dict) else {"data": item},
                                ))
                        else:
                            chunks.append(Document(
                                content=json.dumps(data, indent=2),
                                source=str(file),
                                metadata=data if isinstance(data, dict) else {"data": data},
                            ))
                            
                except Exception as e:
                    print(f"[RAG] Error processing {file}: {e}")
        
        self.documents = chunks
        
        # Generate embeddings
        if chunks:
            texts = [doc.content for doc in chunks]
            
            try:
                if self.use_local_embeddings:
                    self.embeddings = get_embeddings_local(texts)
                else:
                    self.embeddings = get_embeddings(texts, self.embedding_model)
                
                # Store embeddings in documents
                for i, doc in enumerate(self.documents):
                    doc.embedding = self.embeddings[i]
                    
            except Exception as e:
                print(f"[RAG] Error generating embeddings: {e}")
                self.embeddings = None
        
        self._indexed = True

    def _chunk_text(
        self,
        text: str,
        source: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> List[Document]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            source: Source identifier
            chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks
            
        Returns:
            List of Document chunks
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > chunk_size:
                if current_chunk:
                    chunks.append(Document(
                        content=current_chunk.strip(),
                        source=source,
                    ))
                current_chunk = para
            else:
                current_chunk += "\n\n" + para
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(Document(
                content=current_chunk.strip(),
                source=source,
            ))
        
        return chunks

    def search(
        self,
        query: str,
        top_k: int = None,
    ) -> List[str]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant document contents
        """
        if not self._indexed:
            self.index()
        
        if not self.documents or self.embeddings is None:
            return []
        
        top_k = top_k or RAG_TOP_K
        
        try:
            # Get query embedding
            if self.use_local_embeddings:
                query_embedding = get_embeddings_local([query])
            else:
                query_embedding = get_embeddings([query], self.embedding_model)
            
            # Compute similarities
            similarities = cosine_similarity_matrix(
                query_embedding,
                self.embeddings,
            )[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Return documents
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Relevance threshold
                    results.append(self.documents[idx].content)
            
            return results
            
        except Exception as e:
            print(f"[RAG] Search error: {e}")
            return []

    def add_document(self, content: str, source: str = "manual") -> None:
        """
        Add a document to the index.
        
        Args:
            content: Document content
            source: Source identifier
        """
        doc = Document(content=content, source=source)
        
        try:
            if self.use_local_embeddings:
                embedding = get_embeddings_local([content])[0]
            else:
                embedding = get_embeddings([content], self.embedding_model)[0]
            
            doc.embedding = embedding
            self.documents.append(doc)
            
            # Update embeddings array
            if self.embeddings is not None:
                self.embeddings = np.vstack([self.embeddings, embedding])
            else:
                self.embeddings = embedding.reshape(1, -1)
                
        except Exception as e:
            print(f"[RAG] Error adding document: {e}")

    @property
    def document_count(self) -> int:
        """Get number of indexed documents."""
        return len(self.documents)
