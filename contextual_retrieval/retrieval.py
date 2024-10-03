"""
retrieval.py

Core functionality for Contextual Retrieval.
"""

from typing import List, Tuple, Union, Optional
import numpy as np
from .embedding_models import EmbeddingModel
from .context_generator import ContextGenerator
from .vector_store import VectorStore
from .bm25 import BM25Retriever
from .reranker import Reranker
from .utils import chunk_documents

class ContextualRetrieval:
    def __init__(
        self,
        mode: str = 'contextual_embedding',
        embedding_model: Optional[EmbeddingModel] = None,
        context_generator: Optional[ContextGenerator] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        reranker: Optional[Reranker] = None,
        vector_store: Optional[VectorStore] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the ContextualRetrieval system.

        Parameters:
        - mode (str): Retrieval mode. Options are 'contextual_embedding', 'contextual_bm25', 'rerank'.
        - embedding_model (EmbeddingModel): Embedding model to use.
        - context_generator (ContextGenerator): Model for generating context.
        - bm25_retriever (BM25Retriever): BM25 retriever instance.
        - reranker (Reranker): Reranker instance.
        - vector_store (VectorStore): Vector store for embeddings.
        - chunk_size (int): Size of text chunks.
        - chunk_overlap (int): Overlap between chunks.
        """
        if mode not in ['contextual_embedding', 'contextual_bm25', 'rerank']:
            raise ValueError("Invalid mode. Choose from 'contextual_embedding', 'contextual_bm25', or 'rerank'.")

        self.mode = mode
        self.embedding_model = embedding_model or EmbeddingModel()
        self.context_generator = context_generator or ContextGenerator()
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.vector_store = vector_store or VectorStore()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if self.mode in ['contextual_bm25', 'rerank'] and not bm25_retriever:
            self.bm25_retriever = BM25Retriever()
        if self.mode == 'rerank' and not reranker:
            self.reranker = Reranker()

    def index_documents(self, documents: List[str]) -> None:
        """
        Index the given documents.

        Parameters:
        - documents (List[str]): List of documents to index.
        """
        if not documents:
            raise ValueError("No documents provided for indexing.")

        # Chunk documents and get parent document for each chunk
        chunk_tuples = chunk_documents(documents, self.chunk_size, self.chunk_overlap)
        chunks, parent_docs = zip(*chunk_tuples)

        # Generate context for each chunk
        contextualized_chunks = self.context_generator.generate_context(chunks, parent_docs)

        # Compute embeddings
        embeddings = self.embedding_model.encode(contextualized_chunks)

        # Add embeddings to vector store
        self.vector_store.add_embeddings(embeddings, contextualized_chunks)

        # Index chunks for BM25 if applicable
        if self.mode in ['contextual_bm25', 'rerank']:
            self.bm25_retriever.index_chunks(contextualized_chunks)

    def query(self, query_text: str, top_k: int = 5) -> List[Union[str, Tuple[str, float]]]:
        """
        Retrieve relevant chunks for the given query.

        Parameters:
        - query_text (str): The user's query.
        - top_k (int): Number of top results to return.

        Returns:
        - List of relevant chunks or tuples (chunk, score) if scores are requested.
        """
        if not query_text:
            raise ValueError("Query text cannot be empty.")

        # Retrieve using embeddings
        emb_results = self._retrieve_embeddings(query_text, top_k * 3)

        # Retrieve using BM25 if applicable
        if self.mode in ['contextual_bm25', 'rerank']:
            bm25_results = self._retrieve_bm25(query_text, top_k * 3)
            # Combine and deduplicate results
            combined_results = self._combine_results(emb_results, bm25_results)
        else:
            combined_results = emb_results

        # Rerank if mode is 'rerank'
        if self.mode == 'rerank':
            top_results = self.reranker.rerank(query_text, [chunk for chunk, _ in combined_results], return_scores=True)[:top_k]
        else:
            top_results = combined_results[:top_k]

        return top_results

    def _retrieve_embeddings(self, query_text: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Retrieve using embeddings.

        Parameters:
        - query_text (str): The user's query.
        - top_k (int): Number of top results to return.

        Returns:
        - List of tuples (chunk, score).
        """
        query_embedding = self.embedding_model.encode([query_text])[0]
        results = self.vector_store.search(query_embedding, top_k=top_k)
        return results

    def _retrieve_bm25(self, query_text: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Retrieve using BM25.

        Parameters:
        - query_text (str): The user's query.
        - top_k (int): Number of top results to return.

        Returns:
        - List of tuples (chunk, score).
        """
        results = self.bm25_retriever.retrieve(query_text, top_k=top_k)
        return results

    def _combine_results(self, emb_results: List[Tuple[str, float]], bm25_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Combine and deduplicate results from embeddings and BM25.

        Parameters:
        - emb_results (List[Tuple[str, float]]): Results from embeddings retrieval.
        - bm25_results (List[Tuple[str, float]]): Results from BM25 retrieval.

        Returns:
        - Combined and deduplicated list of results.
        """
        results_dict = {}
        for chunk, score in emb_results:
            results_dict[chunk] = score

        for chunk, score in bm25_results:
            if chunk in results_dict:
                results_dict[chunk] += score  # Combine scores
            else:
                results_dict[chunk] = score

        # Sort by combined score descending
        sorted_results = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def get_indexed_document_count(self) -> int:
        """
        Get the number of indexed documents.

        Returns:
        - int: The number of indexed documents.
        """
        return self.vector_store.get_document_count()

    def clear_index(self) -> None:
        """
        Clear all indexed documents.
        """
        self.vector_store.clear()
        if self.bm25_retriever:
            self.bm25_retriever = BM25Retriever()