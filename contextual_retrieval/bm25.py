"""
bm25.py

BM25 retrieval functionality.
"""

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from typing import List, Tuple

class BM25Retriever:
    def __init__(self, tokenizer=None):
        """
        Initialize the BM25 retriever.

        Parameters:
        - tokenizer: A custom tokenizer function. If None, word_tokenize from NLTK is used.
        """
        self.bm25 = None
        self.documents = []
        self.tokenized_corpus = []
        self.tokenizer = tokenizer or word_tokenize

    def index_chunks(self, chunks: List[str]) -> None:
        """
        Index the chunks using BM25.

        Parameters:
        - chunks (List[str]): The text chunks to index.
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing.")
        
        self.documents = chunks
        self.tokenized_corpus = [self._tokenize(doc) for doc in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve relevant documents using BM25.

        Parameters:
        - query (str): The query text.
        - top_k (int): Number of top results to return.

        Returns:
        - List[Tuple[str, float]]: List of tuples (document, score).

        Raises:
        - ValueError: If no documents have been indexed.
        """
        if not self.bm25:
            raise ValueError("No documents have been indexed. Call index_chunks() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_n = scores.argsort()[-top_k:][::-1]
        results = [(self.documents[i], scores[i]) for i in top_n]
        return results

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text.

        Parameters:
        - text (str): The text to tokenize.

        Returns:
        - List[str]: The tokenized text.
        """
        return self.tokenizer(text.lower())

    def get_document_count(self) -> int:
        """
        Get the number of indexed documents.

        Returns:
        - int: The number of indexed documents.
        """
        return len(self.documents)

    def get_average_document_length(self) -> float:
        """
        Get the average length of indexed documents.

        Returns:
        - float: The average document length.

        Raises:
        - ValueError: If no documents have been indexed.
        """
        if not self.tokenized_corpus:
            raise ValueError("No documents have been indexed.")
        return sum(len(doc) for doc in self.tokenized_corpus) / len(self.tokenized_corpus)