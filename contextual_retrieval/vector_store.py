"""
vector_store.py

Handles the vector database for embeddings.
"""

from typing import List, Tuple
import numpy as np
import faiss
import pickle

class VectorStore:
    def __init__(self, dimension: int = 384):
        """
        Initialize the vector store.

        Parameters:
        - dimension (int): Dimension of the embeddings.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[str] = []

    def add_embeddings(self, embeddings: List[np.ndarray], documents: List[str]) -> None:
        """
        Add embeddings and corresponding documents to the store.

        Parameters:
        - embeddings (List[np.ndarray]): The embeddings to add.
        - documents (List[str]): The corresponding documents.

        Raises:
        - ValueError: If the number of embeddings and documents don't match.
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings and documents must match.")
        
        embeddings_array = np.array(embeddings).astype('float32')
        if embeddings_array.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {embeddings_array.shape[1]}")
        
        self.index.add(embeddings_array)
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar embeddings.

        Parameters:
        - query_embedding (np.ndarray): The query embedding.
        - top_k (int): Number of top results to return.

        Returns:
        - List[Tuple[str, float]]: List of tuples (document, score).

        Raises:
        - ValueError: If the query embedding dimension doesn't match the index dimension.
        """
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query embedding dimension mismatch. Expected {self.dimension}, got {query_embedding.shape[0]}")
        
        query_embedding = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                score = 1 / (1 + dist)  # Convert distance to similarity score
                results.append((self.documents[idx], score))
        return results

    def get_document_count(self) -> int:
        """
        Get the number of documents in the store.

        Returns:
        - int: The number of documents.
        """
        return len(self.documents)

    def clear(self) -> None:
        """
        Clear all embeddings and documents from the store.
        """
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents.clear()

    def save(self, index_file_path: str, documents_file_path: str) -> None:
        """
        Save the index and documents to files.

        Parameters:
        - index_file_path (str): Path to save the index.
        - documents_file_path (str): Path to save the documents.
        """
        faiss.write_index(self.index, index_file_path)
        with open(documents_file_path, 'wb') as f:
            pickle.dump(self.documents, f)

    def load(self, index_file_path: str, documents_file_path: str) -> None:
        """
        Load the index and documents from files.

        Parameters:
        - index_file_path (str): Path to load the index from.
        - documents_file_path (str): Path to load the documents from.
        """
        self.index = faiss.read_index(index_file_path)
        with open(documents_file_path, 'rb') as f:
            self.documents = pickle.load(f)
        if self.index.d != self.dimension:
            raise ValueError(f"Loaded index dimension ({self.index.d}) doesn't match current dimension ({self.dimension})")