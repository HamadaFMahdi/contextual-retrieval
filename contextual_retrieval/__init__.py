"""
contextual_retrieval

An open-source Python library for Contextual Retrieval.

This library provides tools for implementing Contextual Retrieval,
a method that significantly improves the retrieval step in 
Retrieval-Augmented Generation (RAG) systems.

Main components:
- ContextualRetrieval: The main class for performing contextual retrieval
- EmbeddingModel: Handles text embedding
- ContextGenerator: Generates context for chunks
- BM25Retriever: Implements BM25 retrieval
- Reranker: Reranks retrieved chunks
- VectorStore: Manages vector storage and retrieval
- chunk_documents: Utility function for document chunking

For more information, visit: https://github.com/yourusername/contextual_retrieval
"""

from .retrieval import ContextualRetrieval
from .embedding_models import EmbeddingModel
from .context_generator import ContextGenerator
from .bm25 import BM25Retriever
from .reranker import Reranker
from .vector_store import VectorStore
from .utils import chunk_documents

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

# Convenience function for quick setup
def create_retriever(mode='contextual_embedding', **kwargs):
    """
    Create and return a ContextualRetrieval instance with the specified mode.

    Parameters:
    - mode (str): Retrieval mode. Options are 'contextual_embedding', 'contextual_bm25', 'rerank'.
    - **kwargs: Additional keyword arguments to pass to ContextualRetrieval constructor.

    Returns:
    - ContextualRetrieval instance
    """
    return ContextualRetrieval(mode=mode, **kwargs)