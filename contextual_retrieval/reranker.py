"""
reranker.py

Reranking retrieved chunks using transformer-based models.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Union, Tuple

class Reranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2', device: str = 'cpu'):
        """
        Initialize the reranker model.

        Parameters:
        - model_name (str): Name of the reranker model.
        - device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.device = device
        self.model.eval()  # Set the model to evaluation mode

    def rerank(self, query: str, documents: List[str], return_scores: bool = False) -> List[Union[str, Tuple[str, float]]]:
        """
        Rerank the documents based on the query.

        Parameters:
        - query (str): The query text.
        - documents (List[str]): Documents to rerank.
        - return_scores (bool): If True, return tuples of (document, score) instead of just documents.

        Returns:
        - List of documents sorted by relevance, or list of (document, score) tuples if return_scores is True.
        """
        if not documents:
            return []

        scores = []
        for doc in documents:
            score = self._score_document(query, doc)
            scores.append((doc, score))

        # Sort documents by score in descending order
        sorted_results = sorted(scores, key=lambda x: x[1], reverse=True)
        
        if return_scores:
            return sorted_results
        else:
            return [doc for doc, _ in sorted_results]

    def _score_document(self, query: str, document: str) -> float:
        """
        Score a single document based on the query.

        Parameters:
        - query (str): The query text.
        - document (str): The document to score.

        Returns:
        - float: The relevance score of the document.
        """
        inputs = self.tokenizer.encode_plus(query, document, return_tensors='pt', max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        score = logits.cpu().numpy()[0][0]
        return float(score)

    def batch_rerank(self, query: str, documents: List[str], batch_size: int = 32, return_scores: bool = False) -> List[Union[str, Tuple[str, float]]]:
        """
        Rerank documents in batches for improved efficiency.

        Parameters:
        - query (str): The query text.
        - documents (List[str]): Documents to rerank.
        - batch_size (int): Number of documents to process in each batch.
        - return_scores (bool): If True, return tuples of (document, score) instead of just documents.

        Returns:
        - List of documents sorted by relevance, or list of (document, score) tuples if return_scores is True.
        """
        if not documents:
            return []

        scores = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_scores = self._batch_score_documents(query, batch)
            scores.extend(zip(batch, batch_scores))

        # Sort documents by score in descending order
        sorted_results = sorted(scores, key=lambda x: x[1], reverse=True)
        
        if return_scores:
            return sorted_results
        else:
            return [doc for doc, _ in sorted_results]

    def _batch_score_documents(self, query: str, documents: List[str]) -> List[float]:
        """
        Score a batch of documents based on the query.

        Parameters:
        - query (str): The query text.
        - documents (List[str]): The documents to score.

        Returns:
        - List[float]: The relevance scores of the documents.
        """
        inputs = self.tokenizer.batch_encode_plus(
            [(query, doc) for doc in documents],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        scores = logits.cpu().numpy()[:, 0]
        return scores.tolist()